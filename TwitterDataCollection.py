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
    # "#HealthcareExperience", "#PatientExperience", "#HealthcareFeedback", "healthcare review",
    # "hospital experience", "doctor feedback",
    "patient satisfaction", "healthcare quality",
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

# # Function to extract keywords from a URL
# def extract_keywords_from_url(url):
#     parsed_url = urlparse(url)
#     keywords = parsed_url.path.split('/') + list(parse_qs(parsed_url.query).keys())
#     keywords = [word.lower() for word in keywords if word]  # Clean and lowercase keywords
#     return keywords

# Keywords that indicate ads, promotions, and job postings
ad_keywords = ["discount", "offer", "buy now", "promo", "link in bio", "sponsored"]
job_keywords = ["hiring", "job opening", "apply now", "position available", "join our team"]

# Function to filter out ad-like, hashtag-only, retweet, and job post tweets
def is_ad_or_unwanted_content(tweet_text):
    # Check if tweet is a retweet
    if tweet_text.startswith("RT @"):
        return True
    # Check for ad or job-related keywords
    if any(keyword in tweet_text.lower() for keyword in ad_keywords + job_keywords):
        return True
    # Exclude tweets with fewer than 5 words (likely less informative)
    if len(tweet_text.split()) < 5:
        return True
    # Exclude tweets that contain mostly hashtags
    if re.fullmatch(r"(#[\w]+(\s+)?)+" , tweet_text.strip()):
        return True
    return False

# Function to collect tweets with polarity and sentiment
def get_tweets(query, count=30):  # Reduced count per query
    tweets_data = []
    for tweet in tweepy.Paginator(client.search_recent_tweets, query=query, max_results=10, tweet_fields=['created_at', 'public_metrics'], expansions='author_id').flatten(limit=count):
        if tweet.id not in existing_ids:  # Only collect new tweets
            tweet_url = f"https://twitter.com/user/status/{tweet.id}"
            text = tweet.text
            # # Filter out unwanted content
            # if is_ad_or_unwanted_content(text):
            #     continue
            polarity = TextBlob(text).sentiment.polarity  # Calculate polarity
            sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

            tweets_data.append({
                'Tweet_ID': tweet.id,
                'Date': tweet.created_at,
                'User': tweet.author_id,
                'Query': query,
                'Tweet': text,
                'Likes': tweet.public_metrics['like_count'],
                'Retweets': tweet.public_metrics['retweet_count'],
                'URL': tweet_url,
                'Polarity': polarity,
                'Sentiment': sentiment
            })
            existing_ids.add(tweet.id)  # Add to existing IDs to avoid duplication
        time.sleep(2)  # Increased delay to avoid hitting rate limits
    return tweets_data

# # Main function to fetch tweets based on keywords extracted from URL
# def fetch_tweets_from_url(url, tweet_count=30):
#     keywords = extract_keywords_from_url(url)
#     print(f"Extracted keywords: {keywords}")
    
# Collect tweets for all queries and save incrementally
# for query in keywords:
for query in queries:
    print(f"Collecting tweets for query: {query}")
    tweets = get_tweets(query, count=30)  # Adjusted the count per query to manage rate limits

    # Convert to DataFrame
    tweets_df = pd.DataFrame(tweets)

    # Append new tweets to CSV
    if not tweets_df.empty:
        tweets_df.to_csv(csv_file, index=False, mode='a', header=False)  # Append without headers after the first write

    print(f"Saved {len(tweets)} new tweets for query '{query}' to CSV.")

# # Example usage
# url = "https://example.com/healthcare/feedback?query=hospital+experience&topic=patient+satisfaction"
# fetch_tweets_from_url(url)
