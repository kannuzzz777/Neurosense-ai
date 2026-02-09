import json
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os

# Ensure NLTK data is available in the Netlify Function environment
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir='/tmp/')
    nltk.data.path.append('/tmp/')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir='/tmp/')
    nltk.data.path.append('/tmp/')

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Define the preprocess_text function
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    processed_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(processed_tokens)

# --- Load the saved TF-IDF vectorizer and model ---
# Adjust paths as necessary for your deployment environment.
# In a Netlify Function, these files should be alongside predict.py

VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.joblib')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'logistic_regression_model.joblib')

try:
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading models: {e}")
    tfidf_vectorizer = None
    model = None

# Netlify Function handler
def handler(event, context):
    if event['httpMethod'] != 'POST':
        return {
            'statusCode': 405,
            'body': json.dumps({'error': 'Method Not Allowed'})
        }

    try:
        body = json.loads(event['body'])
        symptom_text = body.get('symptom_text', '')

        if not symptom_text:
            return {
                'statusCode': 400,
                'headers': { 'Content-Type': 'application/json' },
                'body': json.dumps({'error': 'No symptom_text provided'})
            }

        if tfidf_vectorizer is None or model is None:
             return {
                'statusCode': 500,
                'headers': { 'Content-Type': 'application/json' },
                'body': json.dumps({'error': 'Models not loaded successfully'})
            }

        # Preprocess and predict
        processed_symptom = preprocess_text(symptom_text)
        symptom_tfidf = tfidf_vectorizer.transform([processed_symptom])
        predicted_disease = model.predict(symptom_tfidf)[0]

        return {
            'statusCode': 200,
            'headers': { 'Content-Type': 'application/json' },
            'body': json.dumps({'predicted_disease': predicted_disease})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': { 'Content-Type': 'application/json' },
            'body': json.dumps({'error': str(e), 'trace': 'Check server logs for more details'})
        }
