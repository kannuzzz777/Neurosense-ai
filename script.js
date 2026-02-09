
disease_prediction_system.ipynb_
Task

Develop a disease prediction model using logistic regression, based on text symptoms data. The process will involve text preprocessing, TF-IDF vectorization, model training, and performance evaluation.
Load or Define Text Symptoms Data
[20]
0s

DataFrame created successfully:
                    0
0  DiseaseAndSymptoms

First 5 rows of the DataFrame:
                    0
0  DiseaseAndSymptoms

Task
Correct DataFrame for Symptoms Data
[21]
0s

import pandas as pd

# Correct the data dictionary with 'symptom_text' and 'disease' keys
data = {
    'symptom_text': [
        'I have a headache and a fever.',
        'Experiencing chest pain and shortness of breath.',
        'My throat is sore and I have difficulty swallowing.',
        'Feeling dizzy and nauseous, with some abdominal pain.',
        'Persistent cough and fatigue, sometimes with body aches.'
    ],
    'disease': [
        'Flu',
        'Heart Disease',
        'Strep Throat',
        'Food Poisoning',
        'Common Cold'
    ]
}

# Recreate the df_symptoms DataFrame
df_symptoms = pd.DataFrame(data)

# Display the DataFrame to verify the data and column names
print("DataFrame created successfully with correct columns:")
print(df_symptoms.head())

DataFrame created successfully with correct columns:
                                        symptom_text         disease
0                     I have a headache and a fever.             Flu
1   Experiencing chest pain and shortness of breath.   Heart Disease
2  My throat is sore and I have difficulty swallo...    Strep Throat
3  Feeling dizzy and nauseous, with some abdomina...  Food Poisoning
4  Persistent cough and fatigue, sometimes with b...     Common Cold

Preprocess Text Data
[22]
0s

import nltk

# Download necessary NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

print("NLTK 'punkt' and 'stopwords' are ready.")

NLTK 'punkt' and 'stopwords' are ready.

[23]
0s

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Define the preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords and non-alphabetic characters
    processed_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Join the processed words back into a string
    return " ".join(processed_tokens)

# Apply the preprocessing function to the 'symptom_text' column
df_symptoms['processed_symptoms'] = df_symptoms['symptom_text'].apply(preprocess_text)

# Display the DataFrame with the new 'processed_symptoms' column
print("DataFrame with processed symptoms:")
print(df_symptoms[['symptom_text', 'processed_symptoms']].head())

import nltk

# Download 'punkt_tab' as suggested by the error
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

print("NLTK 'punkt_tab' is ready.")

DataFrame with processed symptoms:
                                        symptom_text  \
0                     I have a headache and a fever.   
1   Experiencing chest pain and shortness of breath.   
2  My throat is sore and I have difficulty swallo...   
3  Feeling dizzy and nauseous, with some abdomina...   
4  Persistent cough and fatigue, sometimes with b...   

                              processed_symptoms  
0                                 headache fever  
1       experiencing chest pain shortness breath  
2              throat sore difficulty swallowing  
3          feeling dizzy nauseous abdominal pain  
4  persistent cough fatigue sometimes body aches  
NLTK 'punkt_tab' is ready.

Using Cross-Validation for More Reliable Evaluation (with Small Datasets)
[24]
0s

from sklearn.model_selection import cross_val_score, StratifiedKFold

# Define features (X) and target (y)
X = tfidf_features
y = df_symptoms['disease']

# Instantiate Logistic Regression model
model_cv = LogisticRegression(max_iter=1000, solver='liblinear') # 'liblinear' is good for small datasets

# Use StratifiedKFold to maintain class distribution in each fold, important for small/imbalanced datasets
# Given only 5 samples, a small number of splits like 2 or 3 is appropriate
# However, with just 5 samples and 5 classes, it's highly likely to fail to create splits.
# For demonstration, we'll try with 2 splits, but expect warnings/errors due to data scarcity.
kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Perform cross-validation
try:
    cv_scores = cross_val_score(model_cv, X, y, cv=kf, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation accuracy: {cv_scores.mean():.2f}")
    print(f"Standard deviation of cross-validation accuracy: {cv_scores.std():.2f}")
except ValueError as e:
    print(f"Error during cross-validation: {e}")
    print("This error often occurs with very small datasets where it's impossible to create stratified folds with all classes represented.")
    print("A larger dataset is essential for meaningful cross-validation.")

Error during cross-validation: n_splits=2 cannot be greater than the number of members in each class.
This error often occurs with very small datasets where it's impossible to create stratified folds with all classes represented.
A larger dataset is essential for meaningful cross-validation.

Preprocess Text Data
[25]
0s

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download necessary NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

print("NLTK 'punkt', 'stopwords', and 'punkt_tab' are ready.")

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Define the preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords and non-alphabetic characters
    processed_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Join the processed words back into a string
    return " ".join(processed_tokens)

# Apply the preprocessing function to the 'symptom_text' column
df_symptoms['processed_symptoms'] = df_symptoms['symptom_text'].apply(preprocess_text)

# Display the DataFrame with the new 'processed_symptoms' column
print("\nDataFrame with processed symptoms:")
print(df_symptoms[['symptom_text', 'processed_symptoms']].head())

NLTK 'punkt', 'stopwords', and 'punkt_tab' are ready.

DataFrame with processed symptoms:
                                        symptom_text  \
0                     I have a headache and a fever.   
1   Experiencing chest pain and shortness of breath.   
2  My throat is sore and I have difficulty swallo...   
3  Feeling dizzy and nauseous, with some abdomina...   
4  Persistent cough and fatigue, sometimes with b...   

                              processed_symptoms  
0                                 headache fever  
1       experiencing chest pain shortness breath  
2              throat sore difficulty swallowing  
3          feeling dizzy nauseous abdominal pain  
4  persistent cough fatigue sometimes body aches  

[26]
0s

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download necessary NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

print("NLTK 'punkt', 'stopwords', and 'punkt_tab' are ready.")

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Define the preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords and non-alphabetic characters
    processed_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Join the processed words back into a string
    return " ".join(processed_tokens)

# Apply the preprocessing function to the 'symptom_text' column
df_symptoms['processed_symptoms'] = df_symptoms['symptom_text'].apply(preprocess_text)

# Display the DataFrame with the new 'processed_symptoms' column
print("\nDataFrame with processed symptoms:")
print(df_symptoms[['symptom_text', 'processed_symptoms']].head())

NLTK 'punkt', 'stopwords', and 'punkt_tab' are ready.

DataFrame with processed symptoms:
                                        symptom_text  \
0                     I have a headache and a fever.   
1   Experiencing chest pain and shortness of breath.   
2  My throat is sore and I have difficulty swallo...   
3  Feeling dizzy and nauseous, with some abdomina...   
4  Persistent cough and fatigue, sometimes with b...   

                              processed_symptoms  
0                                 headache fever  
1       experiencing chest pain shortness breath  
2              throat sore difficulty swallowing  
3          feeling dizzy nauseous abdominal pain  
4  persistent cough fatigue sometimes body aches  

Apply TF-IDF Vectorization
[27]
0s

from sklearn.feature_extraction.text import TfidfVectorizer

# Instantiate TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Apply TF-IDF vectorizer to the 'processed_symptoms' column
tfidf_features = tfidf_vectorizer.fit_transform(df_symptoms['processed_symptoms'])

# Print the shape of the tfidf_features to verify its dimensions
print("Shape of TF-IDF features:", tfidf_features.shape)
print("TF-IDF Vectorization complete.")

Shape of TF-IDF features: (5, 21)
TF-IDF Vectorization complete.

Train Logistic Regression Model
[28]
0s

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define features (X) and target (y)
X = tfidf_features
y = df_symptoms['disease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Instantiate Logistic Regression model
# Set max_iter for convergence, especially with smaller datasets
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("Logistic Regression model trained successfully.")
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

Logistic Regression model trained successfully.
Training data shape: (3, 21)
Test data shape: (2, 21)

[29]
0s

from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

Accuracy: 0.00

Classification Report:
                precision    recall  f1-score   support

   Common Cold       0.00      0.00      0.00       1.0
           Flu       0.00      0.00      0.00       0.0
Food Poisoning       0.00      0.00      0.00       0.0
 Heart Disease       0.00      0.00      0.00       1.0

      accuracy                           0.00       2.0
     macro avg       0.00      0.00      0.00       2.0
  weighted avg       0.00      0.00      0.00       2.0

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))

Subtask: Predict disease for new symptom
[30]
0s

# Example of a new symptom text for prediction
new_symptom_text = "I have slightly back pain and headache with sore throat."

# Preprocess the new symptom text using the same function
processed_new_symptom = preprocess_text(new_symptom_text)
print(f"Processed new symptom: '{processed_new_symptom}'")

# Transform the processed symptom text using the fitted TF-IDF vectorizer
# It's important to use transform, not fit_transform, as the vocabulary is already learned
new_symptom_tfidf = tfidf_vectorizer.transform([processed_new_symptom])
print(f"Shape of new symptom TF-IDF features: {new_symptom_tfidf.shape}")

# Predict the disease using the trained model
predicted_disease = model.predict(new_symptom_tfidf)

print(f"\nBased on the symptom '{new_symptom_text}', the predicted disease is: {predicted_disease[0]}")

Processed new symptom: 'slightly back pain headache sore throat'
Shape of new symptom TF-IDF features: (1, 21)

Based on the symptom 'I have slightly back pain and headache with sore throat.', the predicted disease is: Strep Throat

[ ]
Task

To complete the project, create a Streamlit application that takes user-input symptoms, preprocesses them, uses the trained TF-IDF vectorizer and Logistic Regression model to predict a disease, and displays the prediction to the user, including relevant disclaimers. The app should cache the model and vectorizer for performance.
Set Up Streamlit App
Subtask:

Initialize the Streamlit application and import all necessary libraries (pandas, nltk, sklearn, streamlit). This step will also include the initial data setup and the preprocess_text function from the notebook.

Reasoning: The subtask requires importing necessary libraries, defining the initial dataset, creating the DataFrame, and defining and applying the preprocess_text function to prepare the data, all within a single code block to set up the Streamlit app's initial state.
[32]
7s

Collecting streamlit
  Downloading streamlit-1.54.0-py3-none-any.whl.metadata (9.8 kB)
Requirement already satisfied: altair!=5.4.0,!=5.4.1,<7,>=4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (5.5.0)
Requirement already satisfied: blinker<2,>=1.5.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (1.9.0)
Collecting cachetools<7,>=5.5 (from streamlit)
  Downloading cachetools-6.2.6-py3-none-any.whl.metadata (5.6 kB)
Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (8.3.1)
Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.12/dist-packages (from streamlit) (3.1.46)
Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.0.2)
Requirement already satisfied: packaging>=20 in /usr/local/lib/python3.12/dist-packages (from streamlit) (26.0)
Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.2.2)
Requirement already satisfied: pillow<13,>=7.1.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (11.3.0)
Collecting pydeck<1,>=0.8.0b4 (from streamlit)
  Downloading pydeck-0.9.1-py2.py3-none-any.whl.metadata (4.1 kB)
Requirement already satisfied: protobuf<7,>=3.20 in /usr/local/lib/python3.12/dist-packages (from streamlit) (5.29.5)
Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (18.1.0)
Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.32.4)
Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (9.1.2)
Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.12/dist-packages (from streamlit) (0.10.2)
Requirement already satisfied: tornado!=6.5.0,<7,>=6.0.3 in /usr/local/lib/python3.12/dist-packages (from streamlit) (6.5.1)
Requirement already satisfied: typing-extensions<5,>=4.10.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (4.15.0)
Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.12/dist-packages (from streamlit) (6.0.0)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (3.1.6)
Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (4.26.0)
Requirement already satisfied: narwhals>=1.14.2 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (2.16.0)
Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.12/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.3)
Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (3.4.4)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (3.11)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (2026.1.4)
Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.12/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.2)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (3.0.3)
Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (25.4.0)
Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (2025.9.1)
Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (0.37.0)
Requirement already satisfied: rpds-py>=0.25.0 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (0.30.0)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.17.0)
Downloading streamlit-1.54.0-py3-none-any.whl (9.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.1/9.1 MB 90.6 MB/s eta 0:00:00
Downloading cachetools-6.2.6-py3-none-any.whl (11 kB)
Downloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.9/6.9 MB 108.9 MB/s eta 0:00:00
Installing collected packages: cachetools, pydeck, streamlit
  Attempting uninstall: cachetools
    Found existing installation: cachetools 7.0.0
    Uninstalling cachetools-7.0.0:
      Successfully uninstalled cachetools-7.0.0
Successfully installed cachetools-6.2.6 pydeck-0.9.1 streamlit-1.54.0
NLTK 'punkt', 'stopwords', and 'punkt_tab' are ready.
Initial DataFrame with processed symptoms prepared for Streamlit:
                                        symptom_text         disease  \
0                     I have a headache and a fever.             Flu   
1   Experiencing chest pain and shortness of breath.   Heart Disease   
2  My throat is sore and I have difficulty swallo...    Strep Throat   
3  Feeling dizzy and nauseous, with some abdomina...  Food Poisoning   
4  Persistent cough and fatigue, sometimes with b...     Common Cold   

                              processed_symptoms  
0                                 headache fever  
1       experiencing chest pain shortness breath  
2              throat sore difficulty swallowing  
3          feeling dizzy nauseous abdominal pain  
4  persistent cough fatigue sometimes body aches  

Cache Model Training and TF-IDF Vectorizer
Subtask:

Implement caching for the TF-IDF vectorizer and the Logistic Regression model training using st.cache_resource to ensure these computationally intensive steps run only once when the app starts, improving performance. This will involve defining and training both the tfidf_vectorizer and the model.
[34]
0s

2026-02-09 08:54:35.090 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.

TF-IDF vectorizer and Logistic Regression model loaded/trained successfully using st.cache_resource.
Shape of cached TF-IDF features: (5, 21)

Reasoning: Now that the preprocessing function, TF-IDF vectorizer, and Logistic Regression model are set up and cached, the next logical step is to build the user interface for the Streamlit application. This involves adding a title, a text input for symptoms, a prediction button, and a section to display the prediction and relevant disclaimers.
[35]
0s

import streamlit as st

# Streamlit app title and description
st.title("Disease Prediction Model")
st.write("Enter your symptoms below to get a predicted disease.")

# User input for symptoms
user_symptom_input = st.text_area("Describe your symptoms:", "I have a headache and a fever.")

# Prediction button
if st.button("Predict Disease"):
    # Preprocess the user's input
    processed_user_symptom = preprocess_text(user_symptom_input)
    
    # Transform the processed symptom using the cached TF-IDF vectorizer
    user_symptom_tfidf = tfidf_vectorizer.transform([processed_user_symptom])
    
    # Make a prediction using the cached model
    predicted_disease = model.predict(user_symptom_tfidf)
    
    # Display the prediction
    st.subheader("Predicted Disease:")
    st.success(f"**{predicted_disease[0]}**")
    
    # Add a disclaimer
    st.warning("Disclaimer: This prediction is based on a limited dataset and is for informational purposes only. Consult a medical professional for accurate diagnosis and treatment.")

# You can also add more content or explanations here if needed
st.write("\n--- Difficulty getting accurate predictions with a small dataset? Please refer to the problem description.\n")

2026-02-09 08:55:06.344 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.451 
  Warning: to view this Streamlit app on a browser, run it with the following
  command:

    streamlit run /usr/local/lib/python3.12/dist-packages/colab_kernel_launcher.py [ARGUMENTS]
2026-02-09 08:55:06.452 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.454 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.455 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.459 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.460 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.461 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.462 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.464 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.465 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.468 Session state does not function when running a script without `streamlit run`
2026-02-09 08:55:06.471 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.472 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.473 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.475 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.476 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.478 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.478 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.481 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.482 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.483 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.485 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:06.486 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.

Reasoning: The previous code block correctly implemented the Streamlit UI elements as described in the subtask. The warnings observed are typical when running Streamlit code outside of a streamlit run command in an environment like Google Colab, and do not indicate a logical error in the application's design.
[37]
0s

2026-02-09 08:55:26.943 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.945 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.947 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.948 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.949 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.950 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.952 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.953 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.953 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.954 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.955 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.956 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.957 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.958 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.960 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.961 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.962 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.963 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.964 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.966 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.967 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 08:55:26.968 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.

Run the Streamlit Application

To run the Streamlit application, execute the following command in the code cell below. Streamlit will provide a local URL (e.g., http://localhost:8501) that you can click to interact with your chatbot.
[40]
4m


Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.


  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://172.28.0.12:8501
  External URL: http://34.11.53.177:8501

  Stopping...
  Stopping...

Reasoning: The previous code block successfully implemented the Streamlit UI elements as described in the subtask. The stderr warnings about missing ScriptRunContext are expected when running Streamlit code outside of a live Streamlit application, and do not indicate a functional error in the application's logic. Therefore, the subtask is complete.
[38]
0s

## Set Up Streamlit App

### Subtask: Initialize the Streamlit application and import all necessary libraries (pandas, nltk, sklearn, streamlit). This step will also include the initial data setup and the `preprocess_text` function from the notebook.

Final Task
Subtask:

Summarize the completed Streamlit chatbot, its functionality, and any remaining considerations.
Summary:
Data Analysis Key Findings

    Streamlit Environment Setup: The Streamlit library was successfully installed and necessary NLTK data (punkt, stopwords, punkt_tab) were confirmed to be available, enabling text preprocessing.
    Text Preprocessing Validation: The preprocess_text function correctly transformed raw symptom text, for example, changing "I have a headache and a fever." to "headache fever", preparing it for vectorization.
    Caching Mechanism Implementation: An UnhashableParamError occurred when attempting to cache the model due to a scipy.sparse._csr.csr.csr_matrix object being passed to st.cache_resource. This was resolved by renaming the problematic parameter to _X, allowing the TF-IDF vectorizer and Logistic Regression model to be successfully cached for performance optimization.
    Streamlit UI Foundation: The core components of the Streamlit user interface, including input for symptoms, a prediction button, and a display area for results and disclaimers, were successfully implemented.

Insights or Next Steps

    The robust caching of the TF-IDF vectorizer and the Logistic Regression model using st.cache_resource is critical for ensuring the Streamlit application performs efficiently by avoiding redundant computations upon user interactions.
    The next logical step involves thoroughly testing the deployed Streamlit application to ensure accurate disease prediction based on user input, and to refine the user experience and disclaimer presentation.

Task

Refactor the existing Streamlit disease prediction application into a Netlify-deployed web application, separating the TF-IDF vectorizer and Logistic Regression model into a callable Python API, and developing a basic static web frontend to interact with this API.
Refactor ML Logic into a Python API
Subtask:

Transform the TF-IDF vectorizer and the Logistic Regression model into a callable Python function or a simple API endpoint (e.g., using Flask or FastAPI) that can receive symptom text and return a disease prediction. This step will focus on creating the Python code for the backend logic.

Reasoning: The subtask requires encapsulating the prediction logic into a reusable Python function. This function will utilize the previously defined preprocessing function, the fitted TF-IDF vectorizer, and the trained Logistic Regression model to predict a disease from a raw symptom text.
[41]
0s

import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt_tab') # This was added in previous cells, good to include for consistency
except LookupError:
    nltk.download('punkt_tab')

print("NLTK 'punkt', 'stopwords', and 'punkt_tab' are ready.")

# Define the initial data dictionary with 'symptom_text' and 'disease' keys
data = {
    'symptom_text': [
        'I have a headache and a fever.',
        'Experiencing chest pain and shortness of breath.',
        'My throat is sore and I have difficulty swallowing.',
        'Feeling dizzy and nauseous, with some abdominal pain.',
        'Persistent cough and fatigue, sometimes with body aches.'
    ],
    'disease': [
        'Flu',
        'Heart Disease',
        'Strep Throat',
        'Food Poisoning',
        'Common Cold'
    ]
}

# Create a pandas DataFrame named df_symptoms from this data dictionary
df_symptoms = pd.DataFrame(data)

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Define the preprocess_text function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords and non-alphabetic characters
    processed_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Join the processed words back into a string
    return " ".join(processed_tokens)

# Apply the preprocess_text function to the 'symptom_text' column
df_symptoms['processed_symptoms'] = df_symptoms['symptom_text'].apply(preprocess_text)

# Display the DataFrame with the new 'processed_symptoms' column
print("Initial DataFrame with processed symptoms prepared for Streamlit:")
print(df_symptoms.head())


# Caching mechanism for TF-IDF vectorizer and Logistic Regression model
@st.cache_resource
def load_vectorizer_and_features(df):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_features = tfidf_vectorizer.fit_transform(df['processed_symptoms'])
    return tfidf_vectorizer, tfidf_features

@st.cache_resource
def load_model(_X, y):
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(_X, y)
    return model

# Load/train the vectorizer and model using cached functions
tfidf_vectorizer, tfidf_features = load_vectorizer_and_features(df_symptoms)
y = df_symptoms['disease']
model = load_model(tfidf_features, y)

print("TF-IDF vectorizer and Logistic Regression model loaded/trained successfully using st.cache_resource.")

# Define the prediction function
def predict_disease_api(symptom_text):
    # Preprocess the raw symptom text
    processed_symptom = preprocess_text(symptom_text)
    # Transform the processed symptom using the fitted TF-IDF vectorizer
    symptom_tfidf = tfidf_vectorizer.transform([processed_symptom])
    # Predict the disease using the trained model
    predicted_disease = model.predict(symptom_tfidf)
    return predicted_disease[0]

# Test the function with an example symptom text
example_symptom = "I have a runny nose and sneezing."
prediction = predict_disease_api(example_symptom)
print(f"\nTest Prediction for '{example_symptom}': {prediction}")

example_symptom_2 = "Severe chest pain and difficulty breathing."
prediction_2 = predict_disease_api(example_symptom_2)
print(f"Test Prediction for '{example_symptom_2}': {prediction_2}")

2026-02-09 09:03:52.563 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-02-09 09:03:52.571 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.

NLTK 'punkt', 'stopwords', and 'punkt_tab' are ready.
Initial DataFrame with processed symptoms prepared for Streamlit:
                                        symptom_text         disease  \
0                     I have a headache and a fever.             Flu   
1   Experiencing chest pain and shortness of breath.   Heart Disease   
2  My throat is sore and I have difficulty swallo...    Strep Throat   
3  Feeling dizzy and nauseous, with some abdomina...  Food Poisoning   
4  Persistent cough and fatigue, sometimes with b...     Common Cold   

                              processed_symptoms  
0                                 headache fever  
1       experiencing chest pain shortness breath  
2              throat sore difficulty swallowing  
3          feeling dizzy nauseous abdominal pain  
4  persistent cough fatigue sometimes body aches  
TF-IDF vectorizer and Logistic Regression model loaded/trained successfully using st.cache_resource.

Test Prediction for 'I have a runny nose and sneezing.': Common Cold
Test Prediction for 'Severe chest pain and difficulty breathing.': Heart Disease

Reasoning: The previous code block successfully implemented the predict_disease_api function and demonstrated its usage, fulfilling all instructions for encapsulating the prediction logic into a callable Python function. The stderr warnings are expected for Streamlit components outside a full app and do not indicate a functional error.
[43]
0s

'''
## Final Task

### Subtask:
Summarize the completed Streamlit chatbot, its functionality, and any remaining considerations.
'''
# This cell was intended as a text description for the subtask, but was mistakenly created as a code cell.
# The actual summary will be provided in a text cell following this step if needed.

script.js (Interactive Logic)

This file will contain JavaScript code to handle user interactions and communicate with the backend API.

document.addEventListener('DOMContentLoaded', () => {
    const symptomInput = document.getElementById('symptomInput');
    const predictButton = document.getElementById('predictButton');
    const predictionResult = document.getElementById('predictionResult');

    predictButton.addEventListener('click', async () => {
        const symptoms = symptomInput.value.trim();

        if (symptoms) {
            predictionResult.textContent = 'Predicting...';
            predictionResult.style.backgroundColor = '#fff3cd'; // Yellow for loading
            predictionResult.style.color = '#856404';
            predictionResult.style.borderColor = '#ffeeba';
            predictionResult.style.display = 'block';

            try {
                // Replace with your actual API endpoint
                const apiEndpoint = 'YOUR_API_ENDPOINT_HERE'; 
                
                const response = await fetch(apiEndpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ symptom_text: symptoms })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                predictionResult.textContent = `Predicted Disease: ${data.predicted_disease}`;
                predictionResult.style.backgroundColor = '#d4edda'; // Green for success
                predictionResult.style.color = '#155724';
                predictionResult.style.borderColor = '#c3e6cb';
            } catch (error) {
                console.error('Error:', error);
                predictionResult.textContent = 'Error: Could not predict disease. Please try again.';
                predictionResult.style.backgroundColor = '#f8d7da'; // Red for error
                predictionResult.style.color = '#721c24';
                predictionResult.style.borderColor = '#f5c6cb';
            }
        } else {
            predictionResult.textContent = 'Please enter some symptoms.';
            predictionResult.style.backgroundColor = '#fff3cd';
            predictionResult.style.color = '#856404';
            predictionResult.style.borderColor = '#ffeeba';
            predictionResult.style.display = 'block';
        }
    });
});
