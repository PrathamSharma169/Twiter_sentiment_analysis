# ============= IMPORTS =============
import uvicorn
import joblib
import re
import nltk
import numpy as np
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from flask import Flask, request, render_template, jsonify, send_from_directory
from fastapi import FastAPI, HTTPException
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

# ============= INITIAL SETUP AND MODEL LOADING =============

# Download NLTK stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Load the pre-trained model and associated components
try:
    model = joblib.load("logistic_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError as e:
    print(f"🔴 Error loading model files: {e}")
    print("Ensure 'logistic_model.pkl', 'vectorizer.pkl', and 'label_encoder.pkl' are in the same directory.")
    # For testing purposes, create dummy objects
    print("⚠️ Running in demo mode without actual model files")
    model = None
    vectorizer = None
    label_encoder = None

# ============= SHARED LOGIC (CLEANING & PREDICTION) =============

# This logic is shared between both Flask and FastAPI
ps = PorterStemmer()
all_stopwords = stopwords.words('english')

def clean_text(text: str) -> str:
    """Pre-processes the input text for the model."""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in all_stopwords]
    return ' '.join(words)

def predict_sentiment(tweet_text: str) -> dict:
    """
    Performs sentiment prediction on the cleaned text.
    Returns a dictionary with sentiment, confidence, and cleaned text.
    """
    if not tweet_text:
        raise ValueError("Input tweet text cannot be empty.")
        
    try:
        cleaned_tweet = clean_text(tweet_text)
        
        # If model files are not available, return demo results
        if model is None or vectorizer is None or label_encoder is None:
            # Simple demo logic based on keywords
            positive_words = ['love', 'great', 'amazing', 'good', 'excellent', 'wonderful', 'fantastic']
            negative_words = ['hate', 'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'frustrating']
            
            text_lower = tweet_text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                sentiment = "positive"
                confidence = 0.75 + (pos_count * 0.05)
            elif neg_count > pos_count:
                sentiment = "negative" 
                confidence = 0.75 + (neg_count * 0.05)
            else:
                sentiment = "neutral"
                confidence = 0.65
                
            return {
                "sentiment": sentiment,
                "confidence": min(confidence, 0.95),
                "cleaned_text": cleaned_tweet
            }
        
        vectorized_tweet = vectorizer.transform([cleaned_tweet])
        
        # Predict sentiment and probabilities
        prediction = model.predict(vectorized_tweet)
        prediction_proba = model.predict_proba(vectorized_tweet)
        
        confidence = np.max(prediction_proba)
        sentiment = label_encoder.inverse_transform(prediction)[0]
        
        return {
            "sentiment": str(sentiment),
            "confidence": float(confidence),
            "cleaned_text": cleaned_tweet
        }
    except Exception as e:
        # Broad exception to catch any error during the prediction pipeline
        raise Exception(f"An error occurred during prediction: {str(e)}")

# ============= FASTAPI APPLICATION (FOR API) =============

# 1. Initialize FastAPI app
fastapi_app = FastAPI(
    title="Tweet Sentiment Analysis API",
    description="An API for analyzing the sentiment of a given text (tweet).",
    version="1.0.0"
)

# 2. Add CORS Middleware to allow requests from any origin (your front end)
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define the request body model for type validation
class TweetRequest(BaseModel):
    tweet: str

# 4. Create the API endpoint
@fastapi_app.post("/api/predict")
async def api_predict(request: TweetRequest):
    """
    FastAPI endpoint to predict sentiment.
    Accepts a JSON object with a 'tweet' key.
    """
    try:
        result = predict_sentiment(request.tweet)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============= FLASK APPLICATION (FOR WEB FRONT END) =============

# 1. Initialize Flask app
flask_app = Flask(__name__)

# 2. Define the main route to serve your index.html
@flask_app.route("/")
def index():
    """Serves the main HTML page."""
    try:
        return render_template("index.html")
    except Exception as e:
        return f"Error: Could not find index.html in templates folder. Make sure you have a 'templates' folder with index.html inside it. Error: {str(e)}", 404

# 3. Serve static files (if needed for CSS, JS, images)
@flask_app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files like CSS, JS, images"""
    return send_from_directory('static', filename)

# 4. Define a fallback predict route for Flask
@flask_app.route("/predict", methods=['POST'])
def flask_predict():
    """
    A fallback route that can be called via form submission.
    Returns JSON, making it compatible with the JS on the front end.
    """
    tweet = request.form.get('tweet')
    if not tweet:
        return jsonify({"error": "No tweet text provided"}), 400
    
    try:
        result = predict_sentiment(tweet)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============= COMBINE & RUN =============

# Mount the Flask app as a WSGI sub-application on the FastAPI server
# This allows FastAPI to handle its routes first, and then pass any other
# requests (like for '/') to Flask.
fastapi_app.mount("/", WSGIMiddleware(flask_app))

# Run the server using uvicorn when the script is executed
if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("📁 Created 'templates' directory")
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
        print("📁 Created 'static' directory")
    
    # Check if index.html exists
    if not os.path.exists('templates/index.html'):
        print("⚠️ Warning: templates/index.html not found!")
        print("Please create a 'templates' folder and put your index.html file inside it.")
    
    print("🚀 Starting server... Navigate to http://127.0.0.1:8000")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)
