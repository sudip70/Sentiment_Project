import praw
import csv
from urllib.parse import urlparse

class RedditScraper:
    def __init__(self, client_id, client_secret, user_agent, post_url):
        #Initializing the RedditScraper with API credentials and the post URL
        self.reddit = praw.Reddit(client_id=client_id,
                                  client_secret=client_secret,
                                  user_agent=user_agent)
        self.post_url = post_url
        self.submission = None
        self.csv_file = 'D:/Big Data Analytics/Term-2/BDM 1034 - Application Design for Big Data 01/Project_sentiment/Reddit Data/reddit.csv'
        
    def extract_submission_id(self):
        #Extracting the submission ID from the Reddit post URL
        parsed_url = urlparse(self.post_url)
        submission_id = parsed_url.path.split('/')[4]
        return submission_id
    
    def load_submission(self):
        #Loading the Reddit post using the submission ID
        submission_id = self.extract_submission_id()
        self.submission = self.reddit.submission(id=submission_id)
        print(f"Loaded submission: {self.submission.title}")
        
    def write_to_csv(self):
        #Writing the post and its comments to a CSV file
        with open(self.csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            #Writing the header row for posts and comments
            writer.writerow(['Post Title', 'Post Score', 'Post URL', 'Post Body', 
                             'Comment Body', 'Comment Score', 'Depth'])

            #Writing the post's data to the CSV
            post_body = self.submission.selftext if self.submission.selftext else "No body text"
            writer.writerow([self.submission.title, self.submission.score, self.submission.url, post_body, '', '', 0])

            #Printing post details to the console
            print(f"Title: {self.submission.title}")
            print(f"Post Score: {self.submission.score}")
            print(f"Post URL: {self.submission.url}")
            print(f"Post Body: {post_body}")
            print("\n-----------------------\n")

            #Loading all comments and subcomments
            self.submission.comments.replace_more(limit=None)
            comments = self.submission.comments.list()

            #Processing each top-level comment
            for comment in comments:
                self.process_comment(comment, writer)

        print(f"Data has been written to {self.csv_file}")

    def process_comment(self, comment, writer, depth=1):
        #Processing each comment recursively, writing the data to the CSV
        #Writing the comment and its score to the CSV
        writer.writerow(['', '', '', '', comment.body, comment.score, depth])

        #Printing comment details to the console
        print(f"{'  ' * depth}Comment: {comment.body}")
        print(f"{'  ' * depth}Comment Score: {comment.score}")
        print("\n" + "-" * 30 + "\n")

        #Recursively process replies (subcomments)
        for reply in comment.replies:
            self.process_comment(reply, writer, depth + 1)

#Main execution block
if __name__ == "__main__":
    #Giving Reddit API credentials and post URL
    client_id = 'client_id'
    client_secret = 'client_secret'
    user_agent = 'my_reddit_scraper/0.1 by Project_sentiment'
    post_url = 'https://www.reddit.com/r/hospitals/comments/nk4vu3/trying_to_find_a_conservative_leaning_hospital_in/'

    #Calling RedditScraper class
    scraper = RedditScraper(client_id=client_id, client_secret=client_secret, 
                            user_agent=user_agent, post_url=post_url)
    
    #Loading the submission
    scraper.load_submission()
    
    #Writing the post and comments to a CSV file
    scraper.write_to_csv()
