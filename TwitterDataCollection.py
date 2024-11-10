import pandas as pd
import tweepy

# Twitter API Credentials
api_key = "31MST84iJbbJUZqLwrdggeWJs"
api_secret = "jBNEshs4h1HHAxEMc1iCMYLoWoPomfbVeWhO5VZowws2fjKHzx"
bearer_token = "AAAAAAAAAAAAAAAAAAAAAMw%2BwwEAAAAA%2FdECqxX0cBaUVpZMGvD0LvVX5wg%3DPDI7AWotoK6KKIxiqJoXFQUW4414PZWiCnK1IWS0RFDqdyN8q6"

client = tweepy.Client(bearer_token=bearer_token)

# =====================================================
# The approach below retrieves only the most recent tweets at that specific point in time
# =====================================================
def fetch_twitter_data(query, max_results=100):
    tweets_data = []
    response = client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=["created_at", "text", "author_id"])
    for tweet in response.data:
        tweets_data.append({
            "platform": "Twitter",
            "text": tweet.text,
            "created_at": tweet.created_at
        })
    return pd.DataFrame(tweets_data)

twitter_data = fetch_twitter_data("healthcare OR #HealthcareExperience", max_results=100)