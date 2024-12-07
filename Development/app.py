import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import contractions
import plotly.graph_objects as go

# Downloading required NLTK resources
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt_tab')

# Dictionaries for emojis and abbreviations
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

# Data cleaning class to clean the input data
class DataCleaning:
    def __init__(self):
        self.stop_word = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()  # For stemming
        self.lemmatizer = WordNetLemmatizer()  # For lemmatization
    
    def expand_contractions(self, text):
        return contractions.fix(text)
    
    def replace_emojis(self, text):
        for emoji, meaning in emojis.items():
            text = text.replace(emoji, meaning)
        return text
    
    def replace_abbreviations(self, text):
        for abbr, full_form in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full_form, text)
        return text
    
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

# Sentiment predictor class
class SentimentPredictor:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        self.cleaner = DataCleaning()  
    
    def predict_sentiment(self, new_texts):
        if isinstance(new_texts, str):
            new_texts = [new_texts]
        
        cleaned_texts = [self.cleaner.clean_text(text) for text in new_texts]
        X_new = self.vectorizer.transform(cleaned_texts)
        predictions = self.model.predict(X_new)
        prediction_proba = self.model.predict_proba(X_new)[:, 1]
        return predictions, prediction_proba

# Cache loading functions separately
@st.cache_data
def load_model(model_path):
    return joblib.load(model_path)

@st.cache_data
def load_vectorizer(vectorizer_path):
    return joblib.load(vectorizer_path)

# Streamlit application
@st.cache_data
def display_speedometer(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Sentiment Polarity"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "green"}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}
        }
    ))
    st.plotly_chart(fig)


# Streamlit UI
if __name__ == "__main__":
    st.title("Sentiment Analysis with Speedometer")

    model_path = 'sentiment_model.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'
    
    # Load model and vectorizer with caching
    model = load_model(model_path)
    vectorizer = load_vectorizer(vectorizer_path)
    
    sentiment_predictor = SentimentPredictor(model, vectorizer)
    
    user_input = st.text_area("Enter Text for Sentiment Analysis", "I love this product!")
    
    if st.button("Predict Sentiment"):
        predictions, probabilities = sentiment_predictor.predict_sentiment(user_input)
        
        sentiment = "Positive" if predictions[0] == 1 else "Negative"
        probability = probabilities[0]
        
        st.write(f"Predicted Sentiment: {sentiment}")
        st.write(f"Probability of Positive Sentiment: {probability:.2f}")
        
        display_speedometer(probability)