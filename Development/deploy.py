#Importing the libraries
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import contractions
import os

#Initializing Flask app
app = Flask(__name__)

#Path to model and vectorizer files
model_path = 'stacked_sentiment_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

#Load model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

#Downloading required NLTK resources
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

#Dictionaries for emojis and abbreviations
emojis = {
    ':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', ':-(': 'sad', ':-<': 'sad', 
    ':P': 'raspberry', ':O': 'surprised', ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed', 
    ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy', '@@': 'eyeroll', 
    ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused', '<(-_-)>': 'robot', 'd[-_-]b': 'dj', 
    ":'-)": 'sadsmile', ';)': 'wink', ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'
}

abbreviations = {
    "$": " dollar ", "€": " euro ", "4ao": "for adults only", "a.m": "before midday", "afaik": "as far as I know", 
    "app": "application", "asap": "as soon as possible", "atm": "at the moment", "brb": "be right back", 
    "btw": "by the way", "cu": "see you", "faq": "frequently asked questions", "fyi": "for your information", 
    "idk": "I do not know", "imho": "in my humble opinion", "lol": "laughing out loud", "omg": "oh my god", 
    "omw": "on my way", "ppl": "people", "smh": "shake my head", "tbh": "to be honest", "ttyl": "talk to you later", 
    "u": "you", "w/": "with", "w/o": "without", "wtf": "what the f***", "wyd": "what you doing"
}

#Data cleaning class to clean the input data
class DataCleaning:
    def __init__(self):
        self.stop_word = set(stopwords.words('english'))
        #Stemming
        self.stemmer = PorterStemmer()  
        #Lemmatization
        self.lemmatizer = WordNetLemmatizer()  
    
    #Function for expanding
    def expand_contractions(self, text):
        return contractions.fix(text)
    
    #Function to replace emojis with their meanings
    def replace_emojis(self, text):
        for emoji, meaning in emojis.items():
            text = text.replace(emoji, meaning)
        return text
    
    #function to replace abbreviations with their meanings
    def replace_abbreviations(self, text):
        for abbr, full_form in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full_form, text)
        return text
    
    #Data cleaing
    def clean_text(self, text):
        text = text.lower()
        text = self.expand_contractions(text)
        text = self.replace_emojis(text)
        text = self.replace_abbreviations(text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_word]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)

#Sentiment prediction function
def predict_sentiment(texts):
    cleaner = DataCleaning()
    cleaned_texts = [cleaner.clean_text(text) for text in texts]
    X_new = vectorizer.transform(cleaned_texts)
    predictions_proba = model.predict_proba(X_new)[:, 1]
    
    #Classifing based on probability
    sentiment_labels = []
    for prob in predictions_proba:
        if prob < 0.50:
            #Negative sentiment
            sentiment_labels.append(0)  
        else:
            #Positive sentiment
            sentiment_labels.append(1)  
    return sentiment_labels, predictions_proba

#Home route to welcome users and provide API instructions
@app.route('/')
def home():
    return """
    <h1>Welcome to the Sentiment Analysis API</h1>
    <p>Use the /predict endpoint to analyze sentiment from reviews stored in a CSV file.</p>
    <p>How to use:</p>
    <ul>
        <li>Send a POST request to /predict with a CSV file containing a "text" column.</li>
        <li>API will return sentiment labels (0 for negative, 1 for positive/neutral) and polarity scores.</li>
    </ul>
    <p>Example usage: POST a file to /predict endpoint to get sentiment predictions.</p>
    """

#API endpoint for file upload and sentiment analysis
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    #Checking if the file is a CSV
    if file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            if 'text' not in df.columns:
                return jsonify({"error": "CSV file must contain a 'text' column"}), 400
            
            #Performing sentiment analysis on the uploaded file
            texts = df['text'].tolist()
            sentiment_labels, polarity_scores = predict_sentiment(texts)
            
            #Preparin response
            result = {
                "sentiments": sentiment_labels,
                "polarity_scores": polarity_scores.tolist()
            }
            #Returning the output on json foramt
            return jsonify(result)
        #Expetion hanlding
        except Exception as e:
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
    else:
        return jsonify({"error": "File must be a CSV"}), 400

#Runing the Flask app
if __name__ == "__main__":
    app.run(debug=True)
