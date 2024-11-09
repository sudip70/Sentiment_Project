import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK packages (if not already installed)
nltk.download('punkt')
nltk.download('wordnet')

# Load your DataFrame (replace with your actual file path)
df = pd.read_csv("reddit_data1.csv")
column_name = "Comment Body"  # Ensure the column name is correct

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def tokenize_and_lemmatize(text):
    # Ensure the text is a string and handle missing values
    if isinstance(text, str):
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Lemmatize tokens and convert to lowercase
        lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha()]
        return lemmatized_tokens
    else:
        return []  # Return an empty list if the text is not a string

# Apply the tokenization and lemmatization function to the specified column
df['processed_text'] = df[column_name].apply(tokenize_and_lemmatize)

# Optionally, print the DataFrame to check the tokenized and lemmatized text
print(df.head())  # Print the first few rows to check the result

# Save the output to a new CSV file (optional)
df.to_csv("processed_reddit_data.csv", index=False)  # Save the processed data to a CSV file
