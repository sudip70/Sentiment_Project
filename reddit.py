import praw
import csv
from urllib.parse import urlparse

# Define your Reddit API credentials
reddit = praw.Reddit(client_id='client_id',
                     client_secret='client_secret',
                     user_agent='my_reddit_scraper/0.1 by Project_sentiment')

# Define the Reddit post URL
post_url = 'https://www.reddit.com/r/hospitals/comments/nk4vu3/trying_to_find_a_conservative_leaning_hospital_in/'

# Extract submission ID from the URL
parsed_url = urlparse(post_url)
submission_id = parsed_url.path.split('/')[4]  # Extract 'qmg1n4' from the URL

# Access the Reddit submission (post) using the extracted ID
submission = reddit.submission(id=submission_id)

# Open a CSV file to write data
with open('reddit.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the header row for posts and comments
    writer.writerow(['Post Title', 'Post Score', 'Post URL', 'Post Body', 'Comment Body', 'Comment Score', 'Depth'])

    # Write the post's data to the CSV
    post_body = submission.selftext if submission.selftext else "No body text"
    writer.writerow([submission.title, submission.score, submission.url, post_body, '', '', 0])

    # Print post details to the console
    print(f"Title: {submission.title}")
    print(f"Post Score: {submission.score}")
    print(f"Post URL: {submission.url}")
    print(f"Post Body: {post_body}")
    print("\n-----------------------\n")

    # Load all comments and subcomments
    submission.comments.replace_more(limit=None)  # Load all comments, including nested
    comments = submission.comments.list()

    # Define a recursive function to process comments and subcomments
    def process_comment(comment, depth=1):
        # Write the comment and its score to the CSV
        writer.writerow(['', '', '', '', comment.body, comment.score, depth])

        # Print comment to the console
        print(f"{'  ' * depth}Comment: {comment.body}")
        print(f"{'  ' * depth}Comment Score: {comment.score}")
        print("\n" + "-" * 30 + "\n")

        # Recursively process any replies (subcomments)
        for reply in comment.replies:
            process_comment(reply, depth + 1)

    # Process each top-level comment
    for comment in submission.comments:
        process_comment(comment)

print("Data has been written to reddit.csv")
