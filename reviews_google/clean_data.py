import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
# Download NLTK stopwords 
#nltk.download('stopwords')

# function to clean review text
def clean_text(text):
    """
    Cleans the input review text by removing URLs, special characters, stopwords,
    and extra spaces, and converting text to lowercase.
    
    Parameters:
    text (str): The review text.
    
    Returns:
    str: The cleaned review text.
    """
    # If the text is not a string, convert it to an empty string
    # sometimes reviews do have some numerical values in them
    if not isinstance(text, str):
        text = str(text) if text is not None else ''
    
    # Remove URLs 
    text = re.sub(r'http\S+', '', text)  # Removes URLs starting with http
    
    # Remove special characters 
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    
    # Convert the text to lowercase
    text = text.lower()
    
    # Remove extra spaces (including leading/trailing spaces and multiple spaces)
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))  # Get the set of English stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text
