import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import contractions
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#Downloading required NLTK resources
#Commented out as it only needs to be downloaded once
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

#Dictionaries for emojis and abbreviations
emojis = {
    ':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', ':-(': 'sad', ':-<': 'sad', 
    ':P': 'raspberry', ':O': 'surprised', ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed', 
    ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy', '@@': 'eyeroll', 
    ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused', '<(-_-)>': 'robot', 'd[-_-]b': 'dj', 
    ":'-)": 'sadsmile', ';)': 'wink', ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'
}

abbreviations = {
    "$": " dollar ", "€": " euro ", "4ao": "for adults only", "a.m": "before midday", "afaik": "as far as I know", 
    "app": "application", "asap": "as soon as possible", "atm": "at the moment", "brb": "be right back", 
    "btw": "by the way", "cu": "see you", "faq": "frequently asked questions", "fyi": "for your information", 
    "idk": "I do not know", "imho": "in my humble opinion", "lol": "laughing out loud", "omg": "oh my god", 
    "omw": "on my way", "ppl": "people", "smh": "shake my head", "tbh": "to be honest", "ttyl": "talk to you later", 
    "u": "you", "w/": "with", "w/o": "without", "wtf": "what the f***", "wyd": "what you doing"
}

#Adding custom CSS for button styling
st.markdown("""
    <style>
        .predict-button {
            background-color: #4CAF50; 
            color: white; 
            padding: 10px 24px; 
            font-size: 16px; 
            border: none; 
            cursor: pointer; 
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: background-color 0.3s ease;
        }

        .predict-button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

#Data cleaning class to clean the input data
class DataCleaning:
    def __init__(self):
        self.stop_word = set(stopwords.words('english'))
        #Stemming
        self.stemmer = PorterStemmer() 
        #Lemmatization 
        self.lemmatizer = WordNetLemmatizer()  
    
    #Function to expand the contraction
    def expand_contractions(self, text):
        return contractions.fix(text)
    
    #Replacing emojis with their text representation
    def replace_emojis(self, text):
        for emoji, meaning in emojis.items():
            text = text.replace(emoji, meaning)
        return text
    
    #Replacing abbreviations with their full form
    def replace_abbreviations(self, text):
        for abbr, full_form in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full_form, text)
        return text
    
    #Function to clean the text data
    def clean_text(self, text):
        #Lower case
        if isinstance(text, float):
            text = str(text)  # Convert float to string
        text = text.lower()  # Now it's safe to use lower()
        #Expand
        text = self.expand_contractions(text)
        #Emoji
        text = self.replace_emojis(text)
        #Abbreviation
        text = self.replace_abbreviations(text)
        #Links, mentions, special character 
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        #Tokenization
        tokens = word_tokenize(text)
        #Stop words removal
        tokens = [word for word in tokens if word not in self.stop_word]
        #Lemmatization
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        #Stemming
        tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)

#Caching and loading of the model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    #Trained model
    model_path = 'sentiment_model.pkl'
    #Saved Custom Vectorizer
    vectorizer_path = 'tfidf_vectorizer.pkl'
    #Loading the model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

#Sentiment predictor class with caching
@st.cache_resource
class SentimentPredictor:
    #Initilizing both vectorizer and trained model
    def __init__(self, _model, _vectorizer):
        self.model = _model
        self.vectorizer = _vectorizer
        self.cleaner = DataCleaning()
    
    #Function to predict
    def predict_sentiment(self, new_texts):
        if isinstance(new_texts, str):
            new_texts = [new_texts]
        
        cleaned_texts = [self.cleaner.clean_text(text) for text in new_texts]
        X_new = self.vectorizer.transform(cleaned_texts)
        predictions_proba = self.model.predict_proba(X_new)[:, 1]
        return predictions_proba

#Function to display the sample reviews witrh drop down menu
def display_sample_reviews(reviews, num_samples=5):
    st.subheader("Sample Reviews from File")
    st.write(f"Displaying {num_samples} sample reviews from the uploaded file:")
    
    #Using expander for each review to keep the UI clean
    for i, review in enumerate(reviews[:num_samples]):
        with st.expander(f"Review {i + 1}"):
            st.markdown(f"<p style='font-size:16px; color:#333;'>{review}</p>", unsafe_allow_html=True)

#Function to generate and display the WordCloud plot
def display_wordcloud(reviews):
    text = ' '.join(reviews)
    wordcloud = WordCloud(width=300, height=150, background_color='white').generate(text) 
    
    #Displaying wordcloud using matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


#Speedometer for overall sentiment polarity
#Caching the speedometer, for faster execution
@st.cache_data
def display_speedometer(average_probability):
    #Creating the speedometer gauge figure
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=average_probability * 100,
        title={'text': "Overall Sentiment Polarity"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 35], 'color': "red"},      #Negative
                {'range': [35, 65], 'color': "grey"},    #Neutral
                {'range': [65, 100], 'color': "green"}   #Positive
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}
        }
    ))

    #Creating the layout with columns for the gauge and the labels
    #First column for the plot, second for the labels
    col1, col2 = st.columns([4, 1])  

    with col1:
        #Displaying the gauge (speedometer)
        st.plotly_chart(fig)

    with col2:
        #Displaying sentiment labels on the side
        st.write("<h4 style='color: red; font-size: 14px;'>Negative</h4>", unsafe_allow_html=True)
        st.write("<h4 style='color: grey; font-size: 14px;'>Neutral</h4>", unsafe_allow_html=True)
        st.write("<h4 style='color: green; font-size: 14px;'>Positive</h4>", unsafe_allow_html=True)
    return fig

#Bar chart for sentiment distribution
def display_bar_chart(sentiment_counts):
    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Count",
        title="Sentiment Distribution",
        color="Sentiment",
        color_discrete_map={"Positive": "green", "Neutral": "Grey", "Negative": "red"},
        labels={"Sentiment": "Sentiment Type", "Count": "Number of Reviews"}
    )
    return fig

#Streamlit UI
if __name__ == "__main__":
    st.title("Sentiment Analysis with Reviews")

    #Loading model and vectorizer at the start to have better run time
    model, vectorizer = load_model_and_vectorizer()
    sentiment_predictor = SentimentPredictor(model, vectorizer)

    #Uploading CSV file
    uploaded_file = st.file_uploader("Upload a CSV file with reviews", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        #Ensuring the file has a 'text' column
        if 'text' in df.columns:
            reviews = df['text'].tolist()

            display_sample_reviews(reviews, num_samples=5)
            
            #Adding the styled button
            if st.button('Predict Sentiment', key='predict_sentiment', help="Click to predict sentiment", use_container_width=True):
                #Performing sentiment analysis on the uploaded data
                texts = df['text'].tolist()
                probabilities = sentiment_predictor.predict_sentiment(texts)
                
                #Classifying based on probability
                sentiment_labels = []
                for prob in probabilities:
                    if prob < 0.35:
                        sentiment_labels.append("Negative")
                    elif 0.35 <= prob <= 0.65:
                        sentiment_labels.append("Neutral")
                    else:
                        sentiment_labels.append("Positive")
                

                #Creating a DataFrame for sentiment counts
                sentiment_counts = pd.DataFrame({"Sentiment": sentiment_labels})
                sentiment_distribution = sentiment_counts.value_counts().reset_index()
                sentiment_distribution.columns = ["Sentiment", "Count"]

                #Creating two columns for side-by-side display
                #Adjusting column ratios for layout
                col1, col2 = st.columns([1, 3])  
                
                with col1:
                    #Displaying sentiment distribution table
                    st.subheader("Sentiment Distribution")
                    st.write(sentiment_distribution)
                    
                with col2:
                    st.markdown("<div style='height: 10px; width: 20px;'></div>", unsafe_allow_html=True)
                    #Displaying the wordcloud of reviews
                    display_wordcloud(reviews)
                
                
                #Calculating average probability for overall polarity
                average_probability = sum(probabilities) / len(probabilities)
                st.write(f"Average Probability of Positive Sentiment: **{average_probability:.2f}**")
                
                #Displaying two plots side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    #Displaying speedometer for overall polarity
                    fig = display_speedometer(average_probability)
                
                with col2:
                    #Displaying Bar chart for sentiment distribution
                    bar_chart = display_bar_chart(sentiment_distribution)
                    st.plotly_chart(bar_chart)
        else:
            st.warning("The CSV file does not contain a 'text' column.")
