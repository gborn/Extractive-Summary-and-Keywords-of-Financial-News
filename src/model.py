from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from transformers import pipeline 
import re

class Model:
    def __init__(self, ):
        # we use Google’s T5 model which was pre-trained on a multi-task mixed dataset (including CNN / Daily Mail)
        self.summarizer = TFAutoModelForSeq2SeqLM.from_pretrained("t5-base")
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")

        # KeyBert uses BERT-embeddings and simple cosine similarity to find the sub-phrases in a document that are the most similar to the document itself.
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.kw_model = KeyBERT(sentence_model)

        # sentiment analysis
        self.classifier = pipeline("sentiment-analysis")

    def get_sentiment(self, text):
        result = self.classifier(text)[0]
        return result['label'], result['score']

    def get_keywords(self, text, with_highlight=False):
        # generates list of keyword phrases with their cosine-similarity
        keywords_with_scores = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=5, highlight=with_highlight)
        
        keywords = [kw[0] for kw in keywords_with_scores]
        scores = [kw[1] for kw in keywords_with_scores]
        return keywords, scores

    def get_summary(self, text):
        # T5 uses a max_length of 512 so we cut the article to 512 tokens.
        inputs = self.tokenizer("summarize: " + text, return_tensors="tf", max_length=512)
        outputs = self.summarizer.generate(
                        inputs["input_ids"], 
                        max_length=150, 
                        min_length=40, 
                        length_penalty=2.0, 
                        num_beams=4, 
                        early_stopping=True
                    )
        
        return self.tokenizer.decode(outputs[0])

    def clean_text(self, text):
        text = text.replace(u"\ufffd", "")
        #Remove multiple white spaces with one
        text = re.sub('[\s]+', ' ', text) 
        text = text.strip('\'"').replace("|","") 
        #replace non ascii characters
        text = re.sub('[^\\x00-\\xff]','', text)
        text = text.replace("\\xe2","").replace("\\x80","").replace("\\x99","").replace("\\xf0","").replace("\\x9f","").replace("\\x98","").replace("\\xad","").replace("\\xa6","").replace("\\x9f","")
        return text