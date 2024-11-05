# importing required library 
import pandas as pd
from nltk.corpus import stopwords
import nltk

# Downloading stopwords
nltk.download('stopwords')

# Loading your dataset
file_path = 'reddit_data1.csv'
df = pd.read_csv(file_path)

# Defining stop words
stop_words = set(stopwords.words('english'))

# Function to clean stop words
def remove_stopwords(text):
    if isinstance(text, str):  # Check if text is a string to avoid errors with NaN values
        words = text.split()
        cleaned_text = ' '.join(word for word in words if word.lower() not in stop_words)
        return cleaned_text
    return text

# Remove stop words from the 'Comment Body' column
df['cleaned_comment_body'] = df['Comment Body'].apply(remove_stopwords)

# Save the cleaned data to a new CSV
df.to_csv('cleaned_reddit_data.csv', index=False)
print("Stop words removed from 'Comment Body' and saved to 'cleaned_reddit_data.csv'")
