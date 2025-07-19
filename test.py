import requests

# URL of your deployed API
url = "https://twiter-sentiment-analysis.onrender.com/api/predict"

# The tweet to analyze
data = {
    "tweet": "I absolutely love this new feature!"
}

# Send POST request
response = requests.post(url, json=data)

# Check if successful
if response.status_code == 200:
    result = response.json()
    print("Sentiment:", result["sentiment"])
    print("Confidence:", round(result["confidence"], 3))
    print("Cleaned Text:", result["cleaned_text"])
else:
    print("❌ Error:", response.status_code, response.text)
