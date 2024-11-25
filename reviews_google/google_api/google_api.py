import requests
import csv

API_KEY = ''

# Step 1: Search for health-related places
search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
search_params = {
    'query': 'healthcare facilities in ontario',
    'key': API_KEY
}
search_response = requests.get(search_url, params=search_params)
search_results = search_response.json()

# Step 2: Get Place ID of the first result
place_id = search_results['results'][0]['place_id']

# Step 3: Fetch place details including reviews
details_url = "https://maps.googleapis.com/maps/api/place/details/json"
details_params = {
    'place_id': place_id,
    'fields': 'name,reviews',
    'key': ''
}
details_response = requests.get(details_url, params=details_params)
details_result = details_response.json()

# Step 4: Extract reviews
reviews = details_result['result'].get('reviews', [])

# Step 5: Save reviews to CSV
with open('healthcare_reviews3.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Author Name', 'Rating', 'Review Text'])  # Write header row
    
    for review in reviews:
        author_name = review['author_name']
        rating = review['rating']
        review_text = review['text']
        
        # Write review data to the CSV file
        writer.writerow([author_name, rating, review_text])

print("Reviews saved to healthcare_reviews.csv")
