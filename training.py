import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import nltk
nltk.download("stopwords")

# Clean text function
def clean_text(text):
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in all_stopwords]
    return ' '.join(words)

# Load dataset
data = pd.read_csv("dataset\\train.csv", encoding='ISO-8859-1')

# Clean tweets
data['cleaned_tweet'] = data['text'].astype(str).apply(clean_text)

# Encode labels
label_encoder = LabelEncoder()
data['encoded_sentiment'] = label_encoder.fit_transform(data['sentiment'])

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
X = vectorizer.fit_transform(data['cleaned_tweet'])
y = data['encoded_sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
clf = LogisticRegression(max_iter=200, class_weight='balanced')
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model, vectorizer, and label encoder
joblib.dump(clf, 'logistic_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
