import re
import pandas as pd
import nltk
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer

# Dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

# Dictionary containing common abbreviations and their meanings.
abbreviations = {
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

# Downloading required NLTK resources
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

class data_cleaning:
    def __init__(self, file_path):
        # Initialize with the path to the CSV file
        self.file_path = file_path
        self.df_data = None
        self.stop_word = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()  # For stemming (optional)
        self.lemmatizer = WordNetLemmatizer()  # For lemmatization
        self.tokenizer = WordPunctTokenizer()  # For tokenizing
    
    def load_data(self):
        # Loading data from CSV file
        self.df_data = pd.read_csv(self.file_path, encoding='latin-1', 
                                   names=['target', 'id', 'date', 'flag', 'user', 'text'])
        print("Data loaded successfully.")
        
    def expand_contractions(self, text):
        # Expand contractions in the text using the 'contractions' library
        return contractions.fix(text)
    
    def replace_emojis(self, text):
        # Replace emojis with their meanings
        for emoji, meaning in emojis.items():
            text = text.replace(emoji, meaning)
        return text
    
    def replace_abbreviations(self, text):
        # Replace abbreviations with their meanings
        for abbr, full_form in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full_form, text)
        return text
    
    def clean_text(self, text):
        # Step 1: Normalize text (convert to lowercase)
        text = text.lower()
        
        # Step 2: Expand contractions (e.g., "don't" -> "do not")
        text = self.expand_contractions(text)
        
        # Step 3: Replace emojis with meanings
        text = self.replace_emojis(text)
        
        # Step 4: Replace abbreviations with full forms
        text = self.replace_abbreviations(text)
        
        # Step 5: Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Step 6: Remove mentions (e.g., @username)
        text = re.sub(r'@\w+', '', text)
        
        # Step 7: Remove special characters and punctuation
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Step 8: Tokenization (split text into words)
        tokens = word_tokenize(text)
        
        # Step 9: Remove stopwords (commonly used words that don't carry much meaning)
        tokens = [word for word in tokens if word not in self.stop_word]
        
        # Step 10: Lemmatization (convert words to base form)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Optional Step 11: Stemming (reducing words to their root form)
        tokens = [self.stemmer.stem(word) for word in tokens]  
        
        # Step 12: Join the tokens back into a single string
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def clean_data(self):
        # Applying text cleaning to the dataset
        if self.df_data is not None:
            self.df_data['cleaned_text'] = self.df_data['text'].apply(self.clean_text)
            print("Text data cleaned.")
        else:
            print("Data not loaded yet. Please load the data first.")
    
    def replace_target_values(self):
        # Replacing target values: 4 -> 1 for positive sentiment
        if self.df_data is not None:
            self.df_data['target'] = self.df_data['target'].replace(4, 1)
            print("Target values replaced: 4 -> 1 for positive sentiment.")
        else:
            print("Data not loaded yet. Please load the data first.")
    
    def save_cleaned_data(self, output_file):
        # Saving the cleaned data to a new CSV file
        if self.df_data is not None:
            self.df_data.to_csv(output_file, index=False)
            print(f"Cleaned data saved to {output_file}.")
        else:
            print("No data to save. Please clean the data first.")
    
    def sentiment_distribution(self):
        # Checking sentiment distribution
        if self.df_data is not None:
            print(f"Total Tweets: {len(self.df_data)}")
            print(self.df_data['target'].value_counts())
        else:
            print("Data not loaded yet. Please load the data first.")

# Main execution block
if __name__ == "__main__":
    file_path = 'D:/Big Data Analytics/Term-2/BDM 1034 - Application Design for Big Data 01/Project_sentiment/training_data/train.csv'
    output_file = 'D:/Big Data Analytics/Term-2/BDM 1034 - Application Design for Big Data 01/Project_sentiment/training_data/cleaned_data.csv'
    
    # Calling data_cleaning class
    data = data_cleaning(file_path)
    
    # Loading the dataset
    data.load_data()
    
    # Cleaning the text data
    data.clean_data()
    
    # Replacing target values (4 -> 1)
    data.replace_target_values()
    
    # Checking sentiment distribution
    data.sentiment_distribution()
    
    # Saving the cleaned data to a new file
    data.save_cleaned_data(output_file)
