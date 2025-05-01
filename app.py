import streamlit as st
import pandas as pd
import pickle
import re
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stopwd = stopwords.words('english')

def clean_text(text):

    text = text.lower()  # Lowercasing the text
    text = re.sub('-', ' ', text.lower())   # Replacing `x-x` as `x x`
    text = re.sub(r'http\S+', '', text)  # Removing Links
    text = re.sub(f'[{string.punctuation}]', '', text)  # Remove punctuations
    text = re.sub(r'\s+', ' ', text)  # Removing unnecessary spaces
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Removing single characters

    words = nltk.tokenize.word_tokenize(
        text, language="english", preserve_line=True)
    # Removing the stop words
    text = " ".join([i for i in words if i not in stopwd and len(i) > 2])

    return text.strip()


def load_vectorizer():
    return pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Streamlit app layout
st.title("Spam Email Detector")
st.subheader("Enter an email message to check if it's spam or ham.")
email_input = st.text_area("Email Message Here:")

# Load models
model_files = ["spam_detector_logistic_regression.pkl", "spam_detector_random_forest.pkl",
            "spam_detector_MultiNB.pkl"]
model_options = {model_file.split(
    '.')[0]: model_file for model_file in model_files}
selected_model = st.selectbox("Select Model", list(model_options.keys()))

with open(model_options[selected_model], 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict spam or not
def predict_spam(input_text):
    input_text = clean_text(input_text)
    vectorizer = load_vectorizer()
    input_text = vectorizer.transform([input_text])
    prediction = model.predict(input_text)
    return "Spam" if prediction[0] == 1 else "Not Spam"

    # user_input = st.text_area("Enter the email text:")
if st.button("Predict"):
    result = predict_spam(email_input)
    st.markdown(f"## Prediction: {result}")