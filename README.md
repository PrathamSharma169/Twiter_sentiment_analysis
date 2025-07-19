# Twitter Sentiment Analysis Project

This project implements a sentiment analysis model for tweets using machine learning techniques. It provides a web interface for users to input tweets and receive sentiment predictions.

## Project Structure

```
twitter-sentiment-analysis
├── app.py                # Flask application for user interaction
├── training.py           # Model training and saving
├── testing.py            # Model evaluation on test data
├── requirements.txt      # Project dependencies
├── dataset
│   ├── twitter_training.csv  # Training dataset
│   └── twitter_validation.csv # Validation dataset
├── templates
│   └── index.html         # Frontend HTML template
└── README.md              # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd twitter-sentiment-analysis
   ```

2. **Install dependencies:**
   Create a virtual environment and activate it, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Prepare the dataset:**
   Ensure that the `twitter_training.csv` and `twitter_validation.csv` files are located in the `dataset` folder.

4. **Train the model:**
   Run the `training.py` script to train the sentiment analysis model and save the trained model and label encoder:
   ```
   python training.py
   ```

5. **Run the Flask application:**
   Start the Flask server by running:
   ```
   python app.py
   ```
   Access the web interface at `http://127.0.0.1:5000`.

## Usage Guidelines

- Input a tweet in the provided form on the web interface and submit to receive the predicted sentiment.
- The model has been trained on a dataset of tweets and can classify sentiments into predefined categories.

## Model Evaluation

The `testing.py` script can be used to evaluate the performance of the trained model on the validation dataset. It provides metrics such as accuracy, confusion matrix, and classification report.

## Acknowledgments

This project utilizes various libraries including Flask, TensorFlow, and scikit-learn for building the sentiment analysis model and web application.