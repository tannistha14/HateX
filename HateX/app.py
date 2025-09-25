import streamlit as st
import tensorflow as tf
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np

# ====== NLTK data download and setup ======
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# ====== Text cleaning + NLP preprocessing (from your notebook) ======
def clean_text(text, use_stemming=True, use_lemmatization=False):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    text = text.lower().strip()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    if use_stemming:
        tokens = [stemmer.stem(w) for w in tokens]
    if use_lemmatization:
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# ====== Load the saved model and vectorizer with caching ======
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model("hate_speech_nn_model.h5")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        st.info("Please make sure 'hate_speech_nn_model.h5' and 'tfidf_vectorizer.pkl' are in the same directory.")
        return None, None

model, vectorizer = load_resources()

# ====== Prediction function ======
def predict(text):
    if not model or not vectorizer:
        return None
    
    text_clean = clean_text(text, use_stemming=True, use_lemmatization=False)
    vec = vectorizer.transform([text_clean]).toarray()
    
    # Keras predict returns an array of arrays, so we get the first one
    probs = model.predict(vec)[0]
    
    labels = ["Hate Speech", "Offensive", "Neutral"]
    
    # Return a dictionary of labels and their probabilities
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# ====== Streamlit UI ======
st.set_page_config(page_title="Hate Speech Detector", layout="wide")
st.title("Hate Speech / Offensive Detector")
st.markdown("Enter a text to detect hate speech, offensive content, or neutral.")

user_input = st.text_area("Type a sentence...", height=150, placeholder="Have a good day everyone.")

# Use a button to trigger prediction
if st.button("Predict"):
    if user_input:
        if model and vectorizer:
            with st.spinner("Analyzing..."):
                predictions = predict(user_input)
                if predictions:
                    # Display results in a clear format
                    st.subheader("Prediction Results")
                    predicted_class = max(predictions, key=predictions.get)
                    st.success(f"The text is classified as: **{predicted_class}**")
                    
                    st.markdown("---")
                    st.subheader("Confidence Scores")
                    
                    # Sort and display probabilities using columns for a nice layout
                    cols = st.columns(3)
                    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                    for i, (label, prob) in enumerate(sorted_predictions):
                        with cols[i]:
                            st.metric(label, f"{prob*100:.2f}%")
                            st.progress(prob)
        else:
            st.error("Model or vectorizer not loaded. Please check the files.")
    else:
        st.warning("Please enter some text to analyze.")

# Add example buttons
st.markdown("---")
st.subheader("Try with Examples")
examples = [
    "I hate you!",
    "This is stupid and rude.",
    "Have a good day everyone."
]
col1, col2, col3 = st.columns(3)
with col1:
    if st.button(f'"{examples[0]}"', use_container_width=True):
        st.session_state.user_input = examples[0]
with col2:
    if st.button(f'"{examples[1]}"', use_container_width=True):
        st.session_state.user_input = examples[1]
with col3:
    if st.button(f'"{examples[2]}"', use_container_width=True):
        st.session_state.user_input = examples[2]

# If an example button is clicked, update the text area and run the prediction
if "user_input" in st.session_state and st.session_state.user_input:
    st.text_area("Example Input", value=st.session_state.user_input, height=150)
    if st.button("Analyze Example"):
        with st.spinner("Analyzing..."):
            predictions = predict(st.session_state.user_input)
            if predictions:
                st.subheader("Prediction Results for Example")
                predicted_class = max(predictions, key=predictions.get)
                st.success(f"The example text is classified as: **{predicted_class}**")
                
                st.markdown("---")
                st.subheader("Confidence Scores for Example")
                cols = st.columns(3)
                sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                for i, (label, prob) in enumerate(sorted_predictions):
                    with cols[i]:
                        st.metric(label, f"{prob*100:.2f}%")
                        st.progress(prob)
