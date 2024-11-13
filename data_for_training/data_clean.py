import pandas as pd
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import re

#Downloading stopwords for data cleaning
#nltk.download('stopwords')

class data_cleaning:
    def __init__(self, file_path):
        #Initialize with the path to the CSV file
        self.file_path = file_path
        self.df_data = None
        self.stop_word = set(stopwords.words('english'))
        
    def load_data(self):
        #Loading data from CSV file
        self.df_data = pd.read_csv(self.file_path, encoding='latin-1', 
                                   names=['target', 'id', 'date', 'flag', 'user', 'text'])
        print("Data loaded successfully.")
        
    def clean_text(self, text):
        #Clean the text by removing URLs, mentions, special characters, and stopwords
        text = re.sub(r'http\S+', '', text)  #Removes URLs
        text = re.sub(r'@\w+', '', text)     #Removes mentions
        text = re.sub(r'[^A-Za-z\s]', '', text)  #Removes special characters
        text = text.lower()  #Converts to lowercase
        text = ' '.join(word for word in text.split() if word not in self.stop_word)  #Removes stopwords
        return text
    
    def clean_data(self):
        #Applying text cleaning to the dataset
        if self.df_data is not None:
            self.df_data['cleaned_text'] = self.df_data['text'].apply(self.clean_text)
            print("Text data cleaned.")
        else:
            print("Data not loaded yet. Please load the data first.")
    
    def replace_target_values(self):
        #Replacing target values: 4 -> 1 for positive sentiment
        if self.df_data is not None:
            self.df_data['target'] = self.df_data['target'].replace(4, 1)
            print("Target values replaced: 4 -> 1 for positive sentiment.")
        else:
            print("Data not loaded yet. Please load the data first.")
    
    def save_cleaned_data(self, output_file):
        #Saving the cleaned data to a new CSV file
        if self.df_data is not None:
            self.df_data.to_csv(output_file, index=False)
            print(f"Cleaned data saved to {output_file}.")
        else:
            print("No data to save. Please clean the data first.")
    
    def sentiment_distribution(self):
        #Checking sentiment distribution
        if self.df_data is not None:
            print(f"Total Tweets: {len(self.df_data)}")
            print(self.df_data['target'].value_counts())
        else:
            print("Data not loaded yet. Please load the data first.")

#Main execution block
if __name__ == "__main__":
    file_path = 'train.csv'
    output_file = 'cleaned_data.csv'
    
    #Calling data_cleaning class
    data = data_cleaning(file_path)
    
    #Loading the dataset
    data.load_data()
    
    #Cleaning the text data
    data.clean_data()
    
    #Replacing target values (4 -> 1)
    data.replace_target_values()
    
    #Checking sentiment dirtibution
    data.sentiment_distribution()
    
    #Saving the cleaned data to a new file
    data.save_cleaned_data(output_file)
