import tweepy
import pandas as pd
import time
import os
from textblob import TextBlob

# Twitter API v2 credentials (replace with yours from the Developer Portal)
bearer_token = "bearer_token"

# Set up the Tweepy Client for Twitter API v2
client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

# Define queries for healthcare feedback
queries = [
    "#HealthcareExperience", "#PatientExperience", "#HealthcareFeedback", "healthcare review",
    "hospital experience", "doctor feedback", "patient satisfaction", "healthcare quality",
    "treatment outcome", "medical error", "healthcare cost", "mental health services",
    "telemedicine experience", "COVID hospital experience", "#HealthcareSystem"
]

# File path for CSV
csv_file = "healthcare_tweets.csv"

# Create the CSV file with headers if it doesn't exist
if not os.path.isfile(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        headers = ["Tweet_ID", "Date", "User", "Query", "Tweet", "Likes", "Retweets", "URL", "Polarity", "Sentiment"]
        pd.DataFrame(columns=headers).to_csv(file, index=False)

# Load existing data to check for duplicates
if os.path.isfile(csv_file):
    existing_data = pd.read_csv(csv_file)
    existing_ids = set(existing_data['Tweet_ID'])  # Track tweet IDs to avoid duplicates
else:
    existing_ids = set()

# Function to collect tweets with polarity and sentiment
def get_tweets(query, count=100):
    tweets_data = []
    for tweet in tweepy.Paginator(client.search_recent_tweets, query=query, max_results=10, tweet_fields=['created_at', 'public_metrics'], expansions='author_id').flatten(limit=count):
        if tweet.id not in existing_ids:  # Only collect new tweets
            tweet_url = f"https://twitter.com/user/status/{tweet.id}"
            text = tweet.text
            polarity = TextBlob(text).sentiment.polarity  # Calculate polarity
            sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

            tweets_data.append({
                'Tweet_ID': tweet.id,
                'Date': tweet.created_at,
                'User': tweet.author_id,
                'Query': query,
                'Tweet': text,
                'Likes': tweet.public_metrics['like_count'],
                'Retweets': tweet.public_metrics['retweet_count']
            })
            existing_ids.add(tweet.id)  # Add to existing IDs to avoid duplication
        time.sleep(1)  # Avoid hitting rate limits
    return tweets_data

# Collect tweets for all queries and save incrementally
for query in queries:
    print(f"Collecting tweets for query: {query}")
    tweets = get_tweets(query, count=50)  # Adjust the count per query as needed
    
    # Convert to DataFrame
    tweets_df = pd.DataFrame(tweets)
    
    # Append new tweets to CSV
    if not tweets_df.empty:
        tweets_df.to_csv(csv_file, index=False, mode='a', header=False)  # Append without headers after the first write
    
    print(f"Saved {len(tweets)} new tweets for query '{query}' to CSV.")
