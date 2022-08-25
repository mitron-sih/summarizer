from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
import requests

from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import os    
from PIL import Image
from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
import re
import nltk

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


model_name = "google/pegasus-xsum"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)

@app.route('/', methods=['POST'])
@cross_origin()
def Summary():

    baseUrl = "http://127.0.0.1:8000"

    text = request.json["text"]
    scheme_name = request.json["scheme_name"]
    state = request.json["state"]
    disability_benefits_criteria = request.json["disability_benefits_criteria"]
    benefit_types = request.json["benefit_types"]
    document_link = request.json["document_link"]
    disability_type = request.json["disability_type"]

    print(benefit_types)

    if request.method == "POST":

        if(len(text.split())<50):
            return {
                'summary': text,
                'highlights': '',
                'scheme_name': scheme_name,
                'state' : state,
                'disability_benefits_criteria': disability_benefits_criteria,
                'benefit_types': benefit_types,
                'document_link': document_link,
                'disability_type': disability_type
            }
        
        parser=PlaintextParser.from_string(text,Tokenizer('english'))
        lsa_summarizer=LsaSummarizer()
        lsa_summary= lsa_summarizer(parser.document,5)
        summary={}
        summary["highlights"]=""
        for sentence in lsa_summary:
          summary['highlights']+=str(sentence)+"\n"

        tokens = pegasus_tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
        encoded_summary = pegasus_model.generate(**tokens)
        decoded_summary = pegasus_tokenizer.decode(
              encoded_summary[0],
              skip_special_tokens=True
        )

        summary['summary'] = decoded_summary
        summary['scheme_name'] = scheme_name
        summary['state'] = state
        summary['disability_benefits_criteria'] = disability_benefits_criteria
        summary['benefit_types'] = benefit_types
        summary['document_link'] = document_link
        summary['disability_type'] = disability_type

        requests.post(baseUrl + '/schemes/', data=summary)

        return summary

if __name__ == "__main__":

    app.run()

