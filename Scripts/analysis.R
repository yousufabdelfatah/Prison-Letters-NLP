####### CSV's created after OCR'ing image files

# Set Up ------------------------------------------------------------------

# load packages
library(tidyverse)
library(quanteda)
library(quanteda.textstats)
library(quanteda.textplots)
library(conText)
library(data.table)
library(topicmodels)
library(tidytext)
library(tidyr)
library(ragg)


# Build Egypt Corpus ------------------------------------------------------

# load csv with Egypt data
egypt_text <- 
  read_csv("../Data/egypt_letters_text.csv")

# rename text column so it's easier to work with
egypt_text <- 
  egypt_text %>% 
  rename(text = "0")


# make corpus
egypt_corpus <-
  corpus(egypt_text)

# arabic stopwords
arabic_stop <- 
  readtext::readtext("../Data/Arabic_Stop.txt")

arabic_stop <- 
  strsplit(arabic_stop[['text']], "\n")[[1]]

# tokenize
egypt_tokens <- 
  tokens(egypt_corpus, remove_punct = TRUE, remove_numbers = TRUE) %>% 
  tokens_wordstem(language = "ara") %>% 
  tokens_remove(arabic_stop)


# document feature matrix
egypt_dfm <- 
  egypt_tokens %>% 
  dfm()


# Build Palestine Corpus --------------------------------------------------

# load Palestine text
pal_text <- read_csv("../Data/palestine_letters_text.csv", col_names = FALSE) 

pal_text <- 
  pal_text %>% 
  select(X2) %>% 
  rename(text = "X2")

# build corpus
pal_corpus <-
  corpus(pal_text)

# tokenize
pal_tokens <- 
  tokens(pal_corpus, remove_punct = TRUE, remove_numbers = TRUE) %>% 
  tokens_wordstem(language = "ara") %>% 
  tokens_remove(pattern = arabic_stop) 

# document feature matrix
pal_dfm <- 
  pal_tokens %>% 
  dfm()


# Summary Stats -----------------------------------------------------------

# how long is the average document in each corpus
summary_egy <- 
  textstat_summary(egypt_corpus)

summary_pali <- 
  textstat_summary(pal_corpus)


# egypt token stats
mean(summary_egy$tokens)
max(summary_egy$tokens)
min(summary_egy$tokens)
median(summary_egy$tokens)

# palestine token stats
mean(summary_pali$tokens)
max(summary_pali$tokens)
min(summary_pali$tokens)
median(summary_pali$tokens)


# Create a data frame that combines both sets of tokens with an identifier
data <- 
  data.frame(
  tokens = c(summary_egy$tokens, summary_pali$tokens),
  corpus = c(rep("Egypt", length(summary_egy$tokens)), 
             rep("Palestine", length(summary_pali$tokens)))
  )

# Plot the histograms using ggplot
ggplot(data, aes(x = tokens, fill = corpus)) + 
  geom_density(position = "identity", alpha = 0.5, binwidth = 7) +
  scale_fill_manual(values = c("blue", "red")) +
  labs(title = "Token Distribution", x = "Number of Tokens", y = "Frequency") +
  theme_minimal()



# Word Frequencies --------------------------------------------------------

## Some info here but also a sanity check


# Calculate frequencies Egypt
word_freq <- textstat_frequency(egypt_dfm)

# Get the top 10 words
head(word_freq, 20)

# Calculate frequencies Palestine
pal_freq <- textstat_frequency(pal_dfm)

# Get the top 10 words
head(pal_freq, 20)


# using ragg:app_png here to make sure the Arabic text in the plot reads RTL
# plot the frequencies for both Palestine and Egypt

agg_png('../Plots/palestine_freq.png', width = 600, height = 350, res = 120)

head(pal_freq, 10) %>% 
  ggplot(aes(y = reorder(feature, frequency), x = frequency)) + 
  geom_col(fill = "dark blue") +
  xlab("Frequency") +
  ylab("Word") +
  theme_classic()

dev.off()


agg_png('../Plots/egypt_freq.png', width = 600, height = 350, res = 120)

head(word_freq, 10) %>% 
  ggplot(aes(x = frequency, y = reorder(feature, frequency))) + 
  geom_col(fill = "dark blue") +
  xlab("Frequency") +
  ylab("Word") +
  theme_classic()


dev.off()


# plot wordclouds
agg_png('../Plots/egypt_wordcloud.png', width = 600, height = 350, res = 120)

textplot_wordcloud(egypt_dfm, 
                   min_count = 20,
                   min_size = 0.5,
                   max_size = 5)

dev.off()

agg_png('../Plots/palestine_wordcloud.png', width = 600, height = 350, res = 120)

textplot_wordcloud(pal_dfm, 
                   min_count = 20,
                   min_size = 0.5,
                   max_size = 4)


# Topic Modeling LDA ----------------------------------------------------------

##### egypt topics

egypt_topics <- 
  LDA(egypt_dfm, k = 5, method = "Gibbs", control = list(seed = 1234))

egypt_topics_tidy <- tidy(egypt_topics, matrix = "beta")

head(egypt_topics_tidy)

# top_topics 
top_topics <- topics(egypt_topics, k=2)

# number of documents per top 5 topics
data_frame(topics = seq(1,5), num_documents = table(top_topics)) %>% 
  arrange(desc(num_documents))

## top terms for each topic
egypt_top_terms <- egypt_topics_tidy %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

egypt_top_terms


### Make barcharts with top terms for each topic
topics <- unique(egypt_top_terms$topic)

# Define the number of rows and columns for the layout
n_rows <- ceiling(sqrt(length(topics)))
n_cols <- ceiling(length(topics) / n_rows)

# Start the ragg device
agg_png('../Plots/egypt_topics_terms.png', width = 800, height = 600, res = 120)

# Set up the layout for the faceted plot with outer margins
par(mfrow = c(n_rows, n_cols), mar = c(4, 4, 2, 1), oma = c(5, 5, 0, 0))

# Loop through each topic and create a bar plot
for (i in seq_along(topics)) {
  topic_data <- subset(egypt_top_terms, topic == topics[i])
  barplot(topic_data$beta,
          names.arg = topic_data$term,
          las = 2, 
          main = paste("Topic", topics[i]), 
          col = "light blue",
          border = "white", 
  )
}

# Adding common x and y labels
mtext("Term", side = 1, line = 2, outer = TRUE) 
mtext("Probability", side = 2, line = 2, outer = TRUE) 

# Close the ragg device to save the file
dev.off()


##### Palestine Topics

## get rid of rows with no entries
# Calculate the sum of feature counts for each document
doc_sums <- rowSums(as.matrix(pal_dfm))

# Identify documents with all zero counts
zero_docs <- which(doc_sums == 0)

# Remove these documents from the DFM
pal_dfm <- pal_dfm[-zero_docs, ]

# LDA on Palestinian letters
pal_topics <- 
  LDA(pal_dfm, k = 5, method = "Gibbs", control = list(seed = 1234))


pal_topics_tidy <- tidy(pal_topics, matrix = "beta")

# top_topics 
top_topics <- topics(pal_topics, k=2)

# topic counts
data_frame(topics = seq(1,5), num_documents = table(top_topics)) %>% 
  arrange(desc(num_documents))

## top terms for each topic
pal_top_terms <- pal_topics_tidy %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

pal_top_terms


### Make barcharts with top terms for each topic
topics <- unique(pal_top_terms$topic)

# Define the number of rows and columns for the layout
n_rows <- ceiling(sqrt(length(topics)))
n_cols <- ceiling(length(topics) / n_rows)

# Start the ragg device
agg_png('../Plots/palestine_topics_terms.png', width = 800, height = 600, res = 120)

# Set up the layout for the faceted plot with outer margins
par(mfrow = c(n_rows, n_cols), mar = c(4, 4, 2, 1), oma = c(5, 5, 0, 0))

# Loop through each topic and create a bar plot
for (i in seq_along(topics)) {
  topic_data <- subset(pal_top_terms, topic == topics[i])
  barplot(topic_data$beta,
          names.arg = topic_data$term,
          las = 2, 
          main = paste("Topic", topics[i]), 
          col = "light blue", 
          border = "white", 
  )
}

# Adding common x and y labels
mtext("Term", side = 1, line = 2, outer = TRUE) 
mtext("Probability", side = 2, line = 2, outer = TRUE) 

# Close the ragg device to save the file
dev.off()


# Embeddings Based Analsysis ---------------------------------------------------
# based on template code from Rodriguez, Spirling, Stewart and Wirsching:
# http://alcembeddings.org/alccode
# EMBEDDINGS AND TRANSFORMATION MATRIX NOT IN REPO- AVAILABLE AT ALC SITE

# Transformation matrix 
transform <- readRDS("../Models/fasttext_transform_arwiki_15.rds")

# fastText pretrained embeddings
not_all_na <- function(x) any(!is.na(x))
fasttext <-  setDT(read_delim("../Models/fasttext_vectors_arwiki.vec",
                              delim = " ",
                              quote = "",
                              skip = 1,
                              col_names = F,
                              col_types = cols())) %>%
  dplyr::select(where(not_all_na)) # remove last column which is all NA
word_vectors <-  as.matrix(fasttext, rownames = 1)
colnames(word_vectors) = NULL
rm(fasttext)


# combine both corpuses

# Rename document names to ensure uniqueness
docnames(egypt_corpus) <- paste("EGYPT_", docnames(egypt_corpus), sep = "")
docnames(pal_corpus) <- paste("PAL_", docnames(pal_corpus), sep = "")

# Combine the two corpora
combined_corpus <- corpus(c(egypt_corpus, pal_corpus))

# Create a new document variable 'Egypt'
# It will be 1 for documents from egypt_corpus, 0 for pal_corpus
docvars(combined_corpus, "Egypt") <- 
  c(rep(1, ndoc(egypt_corpus)), rep(0, ndoc(pal_corpus)))

# tokenize without stops
toks <- tokens(combined_corpus, remove_punct = TRUE, remove_numbers = TRUE)

toks_nostop <- toks %>% 
  tokens_remove(arabic_stop) %>% 
  tokens_wordstem(language = "ara")

# only use features that appear at least 10 times in the corpus
feats <- dfm(toks_nostop) %>%
  dfm_trim(min_termfreq = 10) %>% 
  featnames()

toks_nostop <- tokens_select(toks_nostop, feats, padding = TRUE)

# assign english target tokens so r is easier to work with
detainee <- "معتقل"
nation <- "وطن"
prison <- "سجن"

# nearest neighbors for our 3 topics
# for detainee
target_detainee <- 
  tokens_context(x = toks_nostop, pattern = detainee, window = 5L)

feats <- featnames(dfm(target_detainee))

detainee_nns_Egypt <- get_nns(x = target_detainee, N = 10,
                              groups = docvars(target_detainee, 'Egypt'),
                              candidates = feats,
                              pre_trained = word_vectors,
                              transform = TRUE,
                              transform_matrix = transform,
                              bootstrap = F) %>% 
  lapply(., "[[",2) %>% 
  do.call(rbind, .) %>% 
  as.data.frame()

# for nation
target_nation <- 
  tokens_context(x = toks_nostop, pattern = nation, window = 5L)

feats <- featnames(dfm(target_nation))

nation_nns_Egypt <- get_nns(x = target_nation, N = 10,
                            groups = docvars(target_nation, 'Egypt'),
                            candidates = feats,
                            pre_trained = word_vectors,
                            transform = TRUE,
                            transform_matrix = transform,
                            bootstrap = F) %>% 
  lapply(., "[[",2) %>% 
  do.call(rbind, .) %>% 
  as.data.frame()

# for prison
target_prison <- 
  tokens_context(x = toks_nostop, pattern = prison, window = 5L)

feats <- featnames(dfm(target_prison))

prison_nns_Egypt <- get_nns(x = target_prison, N = 10,
                            groups = docvars(target_prison, 'Egypt'),
                            candidates = feats,
                            pre_trained = word_vectors,
                            transform = TRUE,
                            transform_matrix = transform,
                            bootstrap = F) %>% 
  lapply(., "[[",2) %>% 
  do.call(rbind, .) %>% 
  as.data.frame()

# embedding regressions on Prison
set.seed(2021L)
model_prison <- conText(formula = "سجن" ~ Egypt,
                        data = toks,
                        pre_trained = word_vectors,
                        transform = TRUE, 
                        transform_matrix = transform,
                        bootstrap = TRUE,
                        num_bootstraps = 100,
                        confidence_level = 0.95,
                        stratify = FALSE,
                        permute = TRUE, num_permutations = 10,
                        window = 6, case_insensitive = TRUE,
                        verbose = FALSE)

# embedding regression on detainee
set.seed(2021L)
model_detainee <- conText(formula = "معتقل" ~ Egypt,
                          data = toks,
                          pre_trained = word_vectors,
                          transform = TRUE, 
                          transform_matrix = transform,
                          bootstrap = TRUE,
                          num_bootstraps = 100,
                          confidence_level = 0.95,
                          stratify = FALSE,
                          permute = TRUE, num_permutations = 10,
                          window = 6, case_insensitive = TRUE,
                          verbose = FALSE)

# embedding regression on nation
set.seed(2021L)
model_nation <- conText(formula = "*وطن" ~ Egypt,
                        data = toks,
                        pre_trained = word_vectors,
                        transform = TRUE, 
                        transform_matrix = transform,
                        bootstrap = TRUE,
                        num_bootstraps = 100,
                        confidence_level = 0.95,
                        stratify = FALSE,
                        permute = TRUE, num_permutations = 10,
                        window = 6, case_insensitive = TRUE,
                        verbose = FALSE)

# nearest neighbor ratios

ratio_prison <- get_nns_ratio(x = target_prison,
                              N = 20,
                              groups = docvars(target_prison, 'Egypt'),
                              candidates = feats,
                              pre_trained = word_vectors,
                              transform = TRUE,
                              transform_matrix = transform,
                              bootstrap = T,
                              num_bootstraps = 100,
                              permute = T,
                              num_permutations = 100,
                              verbose = FALSE) %>% 
  filter(group != "shared")

ratio_detainee <- get_nns_ratio(x = target_detainee,
                                N = 20,
                                groups = docvars(target_detainee, 'Egypt'),
                                candidates = feats,
                                pre_trained = word_vectors,
                                transform = TRUE,
                                transform_matrix = transform,
                                bootstrap = T,
                                num_bootstraps = 100,
                                permute = T,
                                num_permutations = 100,
                                verbose = FALSE) %>% 
  filter(group != "shared")

ratio_nation <- get_nns_ratio(x = target_nation,
                              N = 20,
                              groups = docvars(target_nation, 'Egypt'),
                              candidates = feats,
                              pre_trained = word_vectors,
                              transform = TRUE,
                              transform_matrix = transform,
                              bootstrap = T,
                              num_bootstraps = 100,
                              permute = T,
                              num_permutations = 100,
                              verbose = FALSE) %>% 
  filter(group != "shared")




