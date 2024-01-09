""" OCR Code for Egyptian and Palestinian Prison Letters
Directory Undefined as raw image files are not included in this repo"""

# import libraries
import os
import tempfile

from google.cloud import vision
from PyPDF2 import PdfFileReader, PdfFileWriter
import pandas as pd

# OCR For Egypt Letters
# ---------------------

egypt_letters_text = []

# egypt letters are in pdf format so we need to split them up

def split_pdf(letters_path, start_page, end_page):
    """Splits a PDF into multiple pages."""
    pdf_reader = PdfFileReader(letters_path)
    pdf_writer = PdfFileWriter()
    for page in range(start_page, end_page):
        pdf_writer.addPage(pdf_reader.getPage(page))
    temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    with open(temp_pdf.name, 'wb') as f:
        pdf_writer.write(f)
    return temp_pdf.name

def detect_text(letters_path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    pdf_reader = PdfFileReader(letters_path)
    total_pages = pdf_reader.getNumPages()
    for start_page in range(0, total_pages, 5):
        end_page = min(start_page + 5, total_pages)
        temp_pdf = split_pdf(letters_path, start_page, end_page)

        with open(temp_pdf, 'rb') as pdf_file:
            input_config = {
                'mime_type': 'application/pdf',
                'content': pdf_file.read(),
            }
        features = [{'type_': vision.Feature.Type.DOCUMENT_TEXT_DETECTION}]
        requests = [{'input_config': input_config, 'features': features}]

        response = client.batch_annotate_files(requests=requests)

        for image_response in response.responses[0].responses:
            egypt_letters_text.append(image_response.full_text_annotation.text)


# directory =

egypt_files = []

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    egypt_files.append(file_path)

for files in egypt_files[1:]:
    detect_text(files)

len(egypt_letters_text)

# makes egypt_letters_text into a dataframe and saves it as a csv file
df = pd.DataFrame(egypt_letters_text)
df.to_csv('egypt_letters_text.csv')


# OCR For Palestine Letters
# ---------------------

palestine_letters_text = []

def detect_text_2(letter_path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    pdf_reader = PdfFileReader(letter_path)
    total_pages = pdf_reader.getNumPages()
    for start_page in range(0, total_pages, 5):
        end_page = min(start_page + 5, total_pages)
        temp_pdf = split_pdf(letter_path, start_page, end_page)

        with open(temp_pdf, 'rb') as pdf_file:
            input_config = {
                'mime_type': 'application/pdf',
                'content': pdf_file.read(),
            }
        features = [{'type_': vision.Feature.Type.DOCUMENT_TEXT_DETECTION}]
        requests = [{'input_config': input_config, 'features': features}]

        response = client.batch_annotate_files(requests=requests)

        for image_response in response.responses[0].responses:
            palestine_letters_text.append(image_response.full_text_annotation.text)


# directory =

palestine_files = []

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    palestine_files.append(file_path)

for files in palestine_files:
    try:
        detect_text_2(files)
    except:
        print("failed")

df = pd.DataFrame(palestine_letters_text)
df.to_csv('palestine_letters_text.csv')
