import re
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from lime.lime_text import LimeTextExplainer
import numpy as np
import joblib
import matplotlib.pyplot as plt

#Class for text preprocessing
class TextCleaner:
    def __init__(self):
        #Initialize stopwords, lemmatizer, and stemmer
        self.stop_word = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

        #Dictionary containing all emojis with their meanings.
        self.emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
                       ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
                       ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed', 
                       ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
                       '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
                       '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
                       ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

        #Dictionary containing common abbreviations and their meanings.
        self.abbreviations = {
            "$": " dollar ", "€": " euro ", "4ao": "for adults only", "a.m": "before midday", 
            "a3": "anytime anywhere anyplace", "afaik": "as far as I know", "app": "application", 
            "asap": "as soon as possible", "atm": "at the moment", "brb": "be right back", "btw": "by the way",
            "cu": "see you", "faq": "frequently asked questions", "fyi": "for your information", "g9": "genius", 
            "idk": "I do not know", "imho": "in my humble opinion", "imo": "in my opinion", "irl": "in real life",
            "jk": "just kidding", "lol": "laughing out loud", "omg": "oh my god", "omw": "on my way", 
            "ppl": "people", "rofl": "rolling on the floor laughing", "smh": "shake my head", 
            "tbh": "to be honest", "thx": "thank you", "ttyl": "talk to you later", "u": "you", 
            "w/": "with", "w/o": "without", "wtf": "what the fuck", "wtg": "way to go", "wyd": "what you doing"
        }

    #Function to extend contractions
    def expand_contractions(self, text):
        return contractions.fix(text)
    
    #Replacing emojis with their meanings
    def replace_emojis(self, text):
        for emojis, meaning in self.emojis.items():
            text = text.replace(emojis, meaning)
        return text
    
    #Replacing abbreviations with their meanings
    def replace_abbreviations(self, text):
        for abbr, full_form in self.abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full_form, text)
        return text
    
    #function to clean the text
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
        #Removing mentions (e.g., @user)
        text = re.sub(r'@\w+', '', text)
        #Removing special characters and punctuation
        text = re.sub(r'[^a-z\s]', '', text)
        #Tokenization
        tokens = word_tokenize(text)
        #Removing stopwords (commonly used words that don't carry much meaning)
        tokens = [word for word in tokens if word not in self.stop_word]
        #Lemmatization (convert words to base form)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        #Stemming (reducing words to their root form)
        tokens = [self.stemmer.stem(word) for word in tokens]  
        #Joining the tokens back into a single string
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text

#Loading the saved model
classifier = joblib.load('stacked_sentiment_model.pkl')

#Loading the saved vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

#Example text data for model interpretibility
data = [
    'The service was satisfactory, and the product generally met expectations love, though it didn’t exceed them. There were no significant problems, but nothing exceptional stood out either. The delivery was timely, and the item arrived in good condition, which was appreciated. However, the overall experience felt average, lacking any standout features. Communication from customer service was prompt, but there was nothing that made the interaction feel particularly memorable or personalized. In the end, it was a standard purchase with nothing particularly remarkable, but it also wasn’t. Just an ordinary transaction without much to note.'
    ]

#Cleaning the text data using TextCleaner
text_cleaner = TextCleaner()
cleaned_data = [text_cleaner.clean_text(text) for text in data]

#Vectorizing the cleaned text using the loaded vectorizer
X = vectorizer.transform(cleaned_data)

#Target labels for classification (e.g., 0 = Negative, 1 = Positive)
y = np.array([0, 1])

#Creating a LIME Text Explainer
explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

#Defining the prediction function for LIME
def predict_fn(texts):
    #Transforming the input texts into the same format as training data
    text_transformed = vectorizer.transform(texts)
    return classifier.predict_proba(text_transformed)

#Explaining the prediction for a single instance (the first text)
explanation = explainer.explain_instance(cleaned_data[0], predict_fn, num_features=20)

#Showing the explanation in text format
print(explanation.as_list())

#Showing the explanation in a plot format
fig = explanation.as_pyplot_figure()
plt.title('LIME interpretibility for the Text')
plt.show()
