import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load saved model, tokenizer, and label encoder
model = load_model("sentiment_lstm_cnn.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Preprocessing function
def clean_text(text):
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 4]
    return " ".join(words)

# Load validation data
df_val = pd.read_csv('dataset/twitter_validation.csv', header=None)
df_val = df_val.dropna()
df_val = df_val[[3, 2]]
df_val.columns = ['tweet_content', 'sentiment']

df_val['cleaned'] = df_val['tweet_content'].apply(clean_text)

# Prepare inputs
X_val = tokenizer.texts_to_sequences(df_val["cleaned"])
X_val = pad_sequences(X_val, maxlen=150)
y_true = label_encoder.transform(df_val["sentiment"])

# Prepare inputs
X_val = tokenizer.texts_to_sequences(df_val["cleaned"])
X_val = pad_sequences(X_val, maxlen=150)

y_true = label_encoder.transform(df_val["sentiment"])

# Predict
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

# Evaluate
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=label_encoder.classes_))
print("\nAccuracy Score:", accuracy_score(y_true, y_pred))
