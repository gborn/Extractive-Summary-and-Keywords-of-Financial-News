from transformers import pipeline
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import re
import requests
import json
import os

# token to authorize with Hugging Face
API_URL = "https://api-inference.huggingface.co/models/gborn/autonlp-news-summarization-483413089"

with open('.env') as f:
    API_TOKEN=str(f.read()).strip('\n')
    
headers = {"Authorization": f"{API_TOKEN}"}

def query(payload, API_URL=API_URL, headers=headers):
    response = requests.post(API_URL, headers=headers, json=payload)
    output =  json.loads(response.content.decode("utf-8"))
    return output

class Model:
    def __init__(self, ):
        # To generate title, we use AutoNLP model trained on Reuters dataset present in our data folder
        # Instead of loading model, we will use accelerated inference from HuggingFace
        # ping model to wake it up
        output = query({"inputs": "The answer to the universe is"})
        if isinstance(output, dict):
            print("Estimated time to load glad's summarizer app", output)


        # sentiment analysis
        self.classifier = pipeline("sentiment-analysis")
        print('Loaded sentiment classification model')

        # we use Google’s T5 model which was pre-trained on a multi-task mixed dataset.
        self.summarizer = pipeline("summarization")
        print('Loaded summarizer')

        # KeyBert uses BERT-embeddings and simple cosine similarity to find the sub-phrases in a document that are the most similar to the document itself.
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.kw_model = KeyBERT(sentence_model)

    def get_sentiment(self, text):
        result = self.classifier(text)[0]
        return result['label'], result['score']

    def get_keywords(self, text, with_highlight=False):
        # generates list of keyword phrases with their cosine-similarity
        keywords_with_scores = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=5, highlight=with_highlight)
        
        keywords = [kw[0] for kw in keywords_with_scores]
        scores = [kw[1] for kw in keywords_with_scores]
        return keywords, scores

    def get_title(self, text):
        # The model eats a lot of memory, so we will use only first 1000 characters
        text = text[:1000]
        output = query({"inputs": text})
        return output[0]['generated_text']

    def get_summary(self, text):
        # T5 uses a max_length of 512 so we cut the article to 512 tokens.
        # inputs = self.summarize_tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        # outputs = self.summarizer.generate(
        #      inputs["input_ids"], 
        #      max_length=150, 
        #      min_length=40, 
        #      length_penalty=2.0, 
        #      num_beams=4, 
        #      early_stopping=True
        # )
        output = self.summarizer(text, max_length=512, min_length=30, do_sample=False)
        return output[0]['summary_text']

    def clean_text(self, text):
        text = text.replace(u"\ufffd", "")
        #Remove multiple white spaces with one
        text = re.sub('[\s]+', ' ', text) 
        text = text.strip('\'"').replace("|","") 
        #replace non ascii characters
        text = re.sub('[^\\x00-\\xff]','', text)
        text = text.replace("\\xe2","").replace("\\x80","").replace("\\x99","").replace("\\xf0","").replace("\\x9f","").replace("\\x98","").replace("\xad","").replace("\\xa6","").replace("\\x9f","")
        return text