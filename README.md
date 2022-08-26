# Summarizer

Flask server that can be used to summarize Government Scheme PDFs. 

The PDFs are first converted to images and then the text is extracted from them using OCR. Then the text is processed to select relevant information. 

We extract the following information from the document:
- The subject of the document
- Date of scheme release
- The list of ministries involved
- One-two line summary of the pdf
- 3-5 important sentences from the document that conveys the gist of the document.

This has been done using the following technologies
- Regular Expressions
- Pegasus state-of-the-art Deep Learning Model
- LSA Extractive Summariser

## Dependencies

- Flask ( for the server)
- Sumy, NLTK ( for summarisation)
- PIL, pdf2image, poppler ( to convert pdf to image to run OCR)
- Pytesseract ( to run OCR)
- Transformers (for the DL model)
- Re ( for regular expressions)
- Requests (to download the PDF using from the URL) 

(requirements.txt has been created to install all the dependencies)
