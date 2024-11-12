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

def clean_data(input_file, output_file):
    """
    Cleans the Google reviews data and saves the cleaned dataset to a new file.
    
    Parameters:
    input_file (str): Path to the input CSV file containing the Google reviews.
    output_file (str): Path where the cleaned CSV file will be saved.
    """
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Check if the necessary columns are in the dataframe
    if 'review' not in df.columns:
        print("Error: 'review' column is missing in the dataset.")
        return
    
    # Clean the 'review' column
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    # Checks a few rows of the cleaned data
    print("Sample cleaned reviews:")
    print(df[['review', 'cleaned_review']].head())  # Print first few rows of original and cleaned reviews
    
    # Save the cleaned data to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Cleaned data has been saved to {output_file}")

if __name__ == "__main__":
    # Set the input and output file paths
    input_file = 'C:/Users/lenovo/Documents/projects/review_data_crawler/google.csv'
    output_file = 'C:/Users/lenovo/Documents/projects/review_data_crawler/cleaned_data2.csv'
    
    # data cleaning function
    clean_data(input_file, output_file)
