import requests
import time
import csv

# Google API key
API_KEY = ''

# Base URL for Google Places API
PLACES_API_URL = "https://maps.googleapis.com/maps/api/place/details/json"

# Function to get place reviews using Google Places API
def get_place_reviews(place_id):
    reviews = []
    next_page_token = None

    while True:
        # Define parameters
        params = {
            'place_id': place_id,  # The Place ID for the location
            'key': API_KEY,        # Your Google API Key
        }

        # Add the next_page_token if available
        if next_page_token:
            params['pagetoken'] = next_page_token

        # Make the API request
        response = requests.get(PLACES_API_URL, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Check if the result contains reviews
            if 'result' in data and 'reviews' in data['result']:
                reviews.extend(data['result']['reviews'])  # Add reviews to the list

            # Check for pagination and if there is a next page token
            next_page_token = data.get('next_page_token')

            # If no next page token is found, we have all the reviews
            if not next_page_token:
                break
            
            # Wait for a few seconds to avoid hitting rate limits (required by Google)
            print("Waiting for next page token...")
            time.sleep(3)  # Wait for a few seconds before fetching the next set of reviews
        else:
            print(f"Error: {response.status_code}")
            break

    # Return all reviews found
    return reviews

# Function to get place ID using Google Places Text Search API
def get_place_id(query):
    search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        'query': query,  # Name of the place you're searching for
        'key': API_KEY,   # Your API Key
    }

    response = requests.get(search_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if 'results' in data and len(data['results']) > 0:
            place_id = data['results'][0]['place_id']
            print(f"Place ID: {place_id}")
            return place_id
        else:
            print("No results found.")
    else:
        print(f"Error: {response.status_code}")

# Function to save reviews into a CSV file
def save_reviews_to_csv(reviews, filename="reviews_health.csv"):
    # Define the header for the CSV file
    header = ['author_name', 'rating', 'text', 'time']
    
    # Open a file in write mode, with newline to prevent extra blank lines in CSV
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()  # Write the header

        # Write each review to the CSV
        for review in reviews:
            writer.writerow({
                'author_name': review.get('author_name', 'No name'),
                'rating': review.get('rating', 'No rating'),
                'text': review.get('text', 'No review text'),
                'time': review.get('time', 'No time')
            })

    print(f"Reviews saved to {filename}")

# Function to print reviews in a readable format (optional)
def print_reviews(reviews):
    if reviews:
        for review in reviews:
            author_name = review.get('author_name', 'No name')
            rating = review.get('rating', 'No rating')
            text = review.get('text', 'No review text')
            time = review.get('time', 'No time')

            # Print review details
            print(f"Author: {author_name}")
            print(f"Rating: {rating}")
            print(f"Review: {text}")
            print(f"Time: {time}")
            print("="*40)
    else:
        print("No reviews found.")

# Main function to get all reviews by place query and save to CSV
def main(query):
    place_id = get_place_id(query)
    if place_id:
        reviews = get_place_reviews(place_id)
        save_reviews_to_csv(reviews)
        print_reviews(reviews)  # Optional, for checking reviews in terminal

# Example: Get reviews for the "Eiffel Tower"
if __name__ == "__main__":
    main("Hospitals in brampton")  # You can replace this with the name of any place
