import joblib
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import contractions

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
class data_cleaning:
    def __init__(self):
        self.stop_word = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()  #For stemming
        self.lemmatizer = WordNetLemmatizer()  #For lemmatization
    
    def expand_contractions(self, text):
        #Expand contractions with contraction library
        return contractions.fix(text)
    
    def replace_emojis(self, text):
        #Replace emojis with their meanings in text for better understanding
        for emoji, meaning in emojis.items():
            text = text.replace(emoji, meaning)
        return text
    
    def replace_abbreviations(self, text):
        #Replace abbreviations with their full forms
        for abbr, full_form in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full_form, text)
        return text
    
    def clean_text(self, text):
        #Normalizing text (convert to lowercase)
        text = text.lower()
        
        #Expanding contractions
        text = self.expand_contractions(text)
        
        #Replacing emojis with meanings
        text = self.replace_emojis(text)
        
        #Replacing abbreviations with full forms
        text = self.replace_abbreviations(text)
        
        #Removing URLs
        text = re.sub(r'http\S+', '', text)
        
        #Removing mentions (e.g., @username)
        text = re.sub(r'@\w+', '', text)
        
        #Removing special characters and punctuation (leaving letters and spaces)
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenizing the text
        tokens = word_tokenize(text)
        
        #Removing stopwords
        tokens = [word for word in tokens if word not in self.stop_word]
        
        #Lemmatization
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        #Stemming
        tokens = [self.stemmer.stem(word) for word in tokens]
        
        #Join tokens back into a cleaned string
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text

#Sentiment predictor class
class SentimentPredictor:
    def __init__(self, model_path, vectorizer_path):
        #Loading the pre-trained model and vectorizer
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        #Initializing the data cleaning object
        self.cleaner = data_cleaning()  
    
    def predict_sentiment(self, new_texts):
        #Checking if input is a single string, if so, convert to a list
        if isinstance(new_texts, str):
            new_texts = [new_texts]
        
        #Cleaning each text before predicting sentiment
        cleaned_texts = [self.cleaner.clean_text(text) for text in new_texts]
        
        #Transforming the cleaned text data using the loaded TfidfVectorizer
        X_new = self.vectorizer.transform(cleaned_texts)
        
        #Predicting the sentiment using the loaded model
        predictions = self.model.predict(X_new)
        #Geting probability for the positive class
        prediction_proba = self.model.predict_proba(X_new)[:, 1]  
        
        #Returning the predictions and probabilities
        return predictions, prediction_proba

#Usage
if __name__ == "__main__":
    #Path to the saved model and vectorizer
    model_path = 'D:/Big Data Analytics/Term-2/BDM 1034 - Application Design for Big Data 01/Project_sentiment/Development/sentiment_model.pkl'
    vectorizer_path = 'D:/Big Data Analytics/Term-2/BDM 1034 - Application Design for Big Data 01/Project_sentiment/Development/tfidf_vectorizer.pkl'
    
    #Initializing the predictor
    sentiment_predictor = SentimentPredictor(model_path, vectorizer_path)
    
    #Giving text to get the sentiment
    new_texts = [
        "This is an amazing product, I love it!",
        "I hate it here",
        "I love you :)",
        "I hate you",
        "It's amazing",
        "I wanna kill you",
        "I had a decent time, nothing special."
    ]
    
    #Predicting the sentiment
    predictions, probabilities = sentiment_predictor.predict_sentiment(new_texts)
    
    #Displaying the results
    for text, prediction, probability in zip(new_texts, predictions, probabilities):
        sentiment = "Positive" if prediction == 1 else "Negative"
        print(f"Text: {text}\nPredicted Sentiment: {sentiment}, Probability: {probability:.2f}\n")
