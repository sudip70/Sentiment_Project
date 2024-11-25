import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Title
st.title("Hospital Feedback Sentiment Analysis")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV file
    df = pd.read_csv(uploaded_file)
    
    # Show sample reviews (replace 'text' with the actual column name)
    st.subheader("Sample Reviews")
    st.write(df["text"].head())  # Replace 'text' with the correct column name if necessary
    
    # Function to analyze sentiment
    def analyze_sentiment(text):
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        return polarity
    
    # Analyze sentiment on button click
    if st.button("Get Sentiment"):
        # Apply sentiment analysis
        df['polarity'] = df['text'].apply(analyze_sentiment)  # Replace 'text' with the correct column name
        
        # Display Polarity Score
        polarity_score = df['polarity'].mean()
        st.subheader(f"Average Polarity Score: {polarity_score}")
        
        # Plotting the polarity as a gauge (like your image)
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=polarity_score,
            title={'text': "Polarity Score"},
            gauge={'axis': {'range': [-1, 1]}, 'bar': {'color': "indianred"}}))

        # Sentiment Distribution - Bar chart using Matplotlib
        sentiment_counts = df['polarity'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
        sentiment_distribution = sentiment_counts.value_counts()
        
        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)

        # First column for polarity gauge
        with col1:
            st.plotly_chart(fig)
            

        # Second column for sentiment distribution bar chart
        with col2:
            fig2, ax = plt.subplots()
            ax.bar(sentiment_distribution.index, sentiment_distribution.values, color=['green', 'red', 'gray'])
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Count')
            ax.set_title('Sentiment Distribution')
            st.pyplot(fig2)
