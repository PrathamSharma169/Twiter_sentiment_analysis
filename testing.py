import pandas as pd
import numpy as np
import re
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Only download once; comment if already available
# nltk.download('stopwords')

# Load trained components
with open('logistic_model.pkl', 'rb') as f:
    model = joblib.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = joblib.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = joblib.load(f)

# Load validation dataset
val_data = pd.read_csv('dataset/test.csv', encoding='ISO-8859-1')
val_data = val_data.dropna()

# Preprocessing function using stemming
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))  # Remove non-alphabetic characters
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# Apply preprocessing
val_data['cleaned_tweet'] = val_data['text'].apply(clean_text)

# Vectorize and encode
X_val = vectorizer.transform(val_data['cleaned_tweet'])
y_true = label_encoder.transform(val_data['sentiment'])

# Predict
y_pred = model.predict(X_val)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=label_encoder.classes_))
print("\nAccuracy Score:", accuracy_score(y_true, y_pred))
