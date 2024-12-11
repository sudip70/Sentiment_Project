import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, when
from pyspark.sql.types import StringType

# Dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

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

# Stopwords list (simplified)
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
])

# Spark session setup
spark = SparkSession.builder \
    .appName("TextCleaningPipelineWithoutNLTK") \
    .getOrCreate()

# Helper functions for text cleaning
def replace_emojis(text):
    for emoji, meaning in emojis.items():
        text = text.replace(emoji, meaning)
    return text

def replace_abbreviations(text):
    for abbr, full_form in abbreviations.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full_form, text)
    return text

def clean_text(text):
    if not text:
        return text
    # Convert text to lowercase
    text = text.lower()
    # Replace emojis
    text = replace_emojis(text)
    # Replace abbreviations
    text = replace_abbreviations(text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Registering UDFs
clean_text_udf = udf(clean_text, StringType())

# Loading data from CSV
def load_data(file_path):
    return spark.read.csv(file_path, header=False, inferSchema=True)

# Main function to clean the data
def clean_data(df):
    # Apply text cleaning
    df = df.withColumn('cleaned_text', clean_text_udf(col('_c4')))
    return df

# Replace target values: 4 -> 1
def replace_target_values(df):
    return df.withColumn('_c0', when(col('_c0') == 4, 1).otherwise(col('_c0')))

# Save cleaned data
def save_cleaned_data(df, output_path):
    df.write.csv(output_path, header=True, mode='overwrite')

# Main Execution Block
if __name__ == "__main__":
    # GCS paths
    input_file = "gs://dataproc-staging-us-central1-350620039298-enjpzo1i/Data/training data/training_datasets.csv"
    output_file = "gs://dataproc-staging-us-central1-350620039298-enjpzo1i/cleaned_data/"

    # Load and process data
    df = load_data(input_file)
    df_cleaned = clean_data(df)
    df_cleaned = replace_target_values(df_cleaned)

    # Save the cleaned data
    save_cleaned_data(df_cleaned, output_file)
    print(f"Cleaned data saved to {output_file}.")
