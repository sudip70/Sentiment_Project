import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the dataset
data = pd.read_csv('cleaned_data.csv')

# Fill any missing values in 'cleaned_text' with an empty string
data['cleaned_text'] = data['cleaned_text'].fillna('')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function for tokenization and lemmatization
def tokenize_and_lemmatize(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Lemmatize each token
    lemmatized_text = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_text

# Apply the function to the 'cleaned_text' column
data['lemmatized_text'] = data['cleaned_text'].apply(tokenize_and_lemmatize)

# Display the results
print(data[['cleaned_text', 'lemmatized_text']].head())
