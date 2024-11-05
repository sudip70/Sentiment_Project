# Importing required libraries
import pandas as pd
from nltk.corpus import stopwords
import nltk
import re

# Downloading stopwords if not already downloaded
nltk.download('stopwords')

# Defining stop words and adding custom stopwords if needed
stop_words = set(stopwords.words('english'))
custom_stopwords = {"is", "in"}  # Add any additional stopwords you want to include
stop_words = stop_words.union(custom_stopwords)

# Loading your dataset
file_path = 'cleaned_reddit_data.csv'  # Replace with the path to your actual file
df = pd.read_csv(file_path)

# Function to remove stopwords and clean the text
def clean_and_remove_stopwords(text):
    if isinstance(text, str):  # Ensure the input is a string
        # Convert to lowercase
        text = text.lower()
        # Remove non-alphabetic characters (keeping spaces)
        text = re.sub(r'[^a-z\s]', '', text)
        # Split the text into words
        words = text.split()
        # Filter out stopwords
        cleaned_words = [word for word in words if word not in stop_words]
        # Join cleaned words back into a string
        cleaned_text = ' '.join(cleaned_words)
        return cleaned_text
    return text

# Apply the stopword removal function to the 'Comment Body' column
df['cleaned_comment_body'] = df['Comment Body'].apply(clean_and_remove_stopwords)

# Save the cleaned data to a new CSV file
df.to_csv('final_cleaned_reddit_data.csv', index=False)
print("Stop words removed and text cleaned from 'Comment Body'. Saved to 'final_cleaned_reddit_data.csv'.")
