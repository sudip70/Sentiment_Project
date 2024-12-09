import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
    # # Generate a WordCloud
    # generate_wordcloud(df['cleaned_review'])

def generate_wordcloud(cleaned_reviews):
    """
    Generates a WordCloud from the cleaned reviews.
    
    Parameters:
    cleaned_reviews (pd.Series): The cleaned reviews column.
    """
    # Combine all cleaned reviews into a single string
    all_reviews = ' '.join(cleaned_reviews)
    
    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_reviews)
    
    # Display the WordCloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Hide axes
    plt.show()
    
    # Generate a WordCloud
    generate_wordcloud(df['cleaned_review'])    