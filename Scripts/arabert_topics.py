""" Text Analysis of Egyptian and Palestinian Prison Letters"""
# Set Up
# ---------------------

# import packages
import re
import string

import arabic_reshaper
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from ar_wordcloud import ArabicWordCloud
from bertopic import BERTopic
from bidi.algorithm import get_display
from flair.embeddings import TransformerDocumentEmbeddings
from sklearn.feature_extraction.text import CountVectorizer
from snowballstemmer import stemmer

####### CSV's created after OCR'ing image files
# load Egypt Letters
df = pd.read_csv('../Data/egypt_letters_text.csv', header=None, index_col=0, names=['Text'])
df.head()

# load Palestine letters
df2 = pd.read_csv('../Data/palestine_letters_text.csv', index_col=0, names=['Text'])
df2.head()

# Egypt Data Cleaning
# ---------------------

# this can be accomplished with camel-tools package but this way gives me a little bit more control
ARABIC_PUNCTUATIONS = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
ENGLISH_PUNCTUATIONS = string.punctuation
PUNCTUATIONS_LIST = ARABIC_PUNCTUATIONS + ENGLISH_PUNCTUATIONS

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                        #  """, re.VERBOSE)

# this helps with "orthofrpahic ambiguity by dealing with spelling inconsistencies"
def normalize_arabic(text):
    """ Standardizes characters and spellings"""
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_diacritics(text):
    """ Removes diacritics from text"""
    text = re.sub(arabic_diacritics, '', text)
    return text


def remove_punctuations(text):
    """ Removes punctuations from text"""
    translator = str.maketrans('', '', PUNCTUATIONS_LIST)
    return text.translate(translator)


def remove_repeating_char(text):
    """ Removes repeating characters"""
    return re.sub(r'(.)\1+', r'\1', text)

df["Text"] = df["Text"].apply(normalize_arabic)\
    .apply(remove_diacritics).\
        apply(remove_punctuations).\
            apply(remove_repeating_char)

# replace any new lines characters
df['Text'] = df['Text'].replace('\n', '', regex=True)

# stopwords
stopwords = open('../Data/Arabic_Stop.txt', "r", encoding='utf-8')
stopwords = stopwords.read().replace('\n', ' ').split(' ')
len(stopwords)

stopwords = [normalize_arabic(item) for item in stopwords]
stopwords = [remove_diacritics(item) for item in stopwords]
stopwords = [remove_punctuations(item) for item in stopwords]
stopwords = [remove_repeating_char(item) for item in stopwords]

# Create tokenizer that includes stemming

class StemTokenizer:
    """ Tokenizer using Stemming"""
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.ar = stemmer("arabic")
    def __call__(self, doc):
        return [self.ar.stemWord(t)
                for t in nltk.tokenize.word_tokenize(doc)
                if t not in self.ignore_tokens]

tokenizer=StemTokenizer()

vectorizer = CountVectorizer(stop_words= stopwords, # lemmatized stop words list
                    lowercase=True, # make everything lower case
                    tokenizer = tokenizer, # tokenizer we created above
                    ngram_range=(1,2), # return bigrams and unigrams
                    min_df=5, # ignore words that appear in less than 5 documents
                    max_df=0.8) # ignore words that appear in more than 80% of documents

# lemmatize the stopwords list
stopwords = tokenizer(' '.join(stopwords))

# apply to text
X = vectorizer.fit_transform(df["Text"])
dtm = vectorizer.transform(df["Text"])
dtm = pd.DataFrame(dtm.toarray(), columns = vectorizer.get_feature_names())
freq = dtm.sum(axis = 0).sort_values(ascending = False)

# Sanity Check
# ---------------------

# most frequent words
most_freq = pd.DataFrame(freq[:30]).reset_index().rename(columns={'index':'Word', 0:'Count'})

print(most_freq)

# wordcloud
awc = ArabicWordCloud(background_color="black")

freq = pd.Series(freq, name="Number")

freq = freq[1:20]

my_dict = freq.to_dict()

dict_wc = awc.from_dict(my_dict, ignore_stopwords=True)
awc.plot(dict_wc, width=15, height=15)
dict_wc.to_file("wordcloud.jpg")

freq_index_fixed = []

for i in list(freq.index.astype(str)):
    X = arabic_reshaper.reshape(i)
    y = get_display(X)
    freq_index_fixed.append(y)

# keywords barchart
plt.rcParams["figure.figsize"] = (12,10)
plt.bar(freq_index_fixed, freq.values)
plt.xlabel("Keywords")
plt.ylabel("Frequency")
plt.savefig("keywords_barchart.jpg")

# Palesine Data Cleaning
# ---------------------

# apply what we created above
df2["Text"] = df2["Text"].apply(str)

df2["Text"] = df2["Text"].apply(normalize_arabic)\
    .apply(remove_diacritics).\
        apply(remove_punctuations).\
            apply(remove_repeating_char)

X = vectorizer.fit_transform(df2["Text"])
dtm = vectorizer.transform(df2["Text"])
dtm = pd.DataFrame(dtm.toarray(), columns = vectorizer.get_feature_names())
freq = dtm.sum(axis = 0).sort_values(ascending = False)

# Sanity Check
# ---------------------

# most frequent words
most_freq = pd.DataFrame(freq[:30]).reset_index().rename(columns={'index':'Word', 0:'Count'})

print(most_freq)

# wordcloud

awc = ArabicWordCloud(background_color="black")

freq = pd.Series(freq, name = "Number")

freq = freq[1:20]

my_dict = freq.to_dict()

dict_wc = awc.from_dict(my_dict, ignore_stopwords=True)
awc.plot(dict_wc, width=15, height=15)
dict_wc.to_file("wordcloud.jpg")

freq_index_fixed = []

for i in list(freq.index.astype(str)):
    X = arabic_reshaper.reshape(i)
    y = get_display(X)
    freq_index_fixed.append(y)

# keywords barchart
plt.rcParams["figure.figsize"] = (12,10)
plt.bar(freq_index_fixed, freq.values)
plt.xlabel("Keywords")
plt.ylabel("Frequency")
plt.savefig("keywords_barchart.jpg")

# Topic Modeling
# ---------------------

# arabert embeddings
arabert = TransformerDocumentEmbeddings('aubmindlab/bert-base-arabertv02')

### EGYPT ###

# topic modeling using BERTopic
topic_model= BERTopic(embedding_model=arabert, n_gram_range=(2,3), nr_topics='auto', verbose=True)

topics, probabilities = topic_model.fit_transform(list(df['Text']))

# print topics
topic_model.get_topic_info()

# visualize the top words of each topic
topic_model.visualize_barchart()

# spatial visualization of topics
topic_model.visualize_topics()

# limit to 10 topics
topic_model= BERTopic(embedding_model=arabert, n_gram_range=(2,3), nr_topics=10, verbose=True)

topics, probabilities = topic_model.fit_transform(list(df['Text']))

# print the topics
topic_model.get_topic_info()

# visualize the top words of each topic
topic_model.visualize_barchart()

# spatial visualization of topics
topic_model.visualize_topics()

### PALESTINE ###

topic_model= BERTopic(embedding_model=arabert, n_gram_range=(2,3), nr_topics='auto', verbose=True)

topics, probabilities = topic_model.fit_transform(list(df2['Text']))

# print the topics
topic_model.get_topic_info()

# visualize the top words of each topic
topic_model.visualize_barchart()
### not great

# limit to 10
topic_model= BERTopic(embedding_model=arabert, n_gram_range=(2,3), nr_topics=10, verbose=True)

topics, probabilities = topic_model.fit_transform(list(df2['Text']))

# print the topics to json
topic_model.get_topic_info()

# visualize the top words of each topic
topic_model.visualize_barchart()
