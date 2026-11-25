# main.py: IMDB movie review sentiment app with a Simple RNN
import re

import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences

MAX_FEATURES = 10000  # vocabulary size used in training
MAX_LEN = 500  # review length used in training


# Step 1: load the model and the word index only once
@st.cache_resource
def load_assets():
    model = load_model("simple_rnn_imdb.keras")
    word_index = imdb.get_word_index()
    return model, word_index


model, word_index = load_assets()


# Step 2: helper function, text -> padded word ids (same as training)
def preprocess_text(text):
    words = re.findall(r"[a-z0-9']+", text.lower())  # lowercase, drop punctuation
    encoded = [1]  # 1 = start of review
    for word in words:
        index = word_index.get(word)
        if index is None or index + 3 >= MAX_FEATURES:
            encoded.append(2)  # 2 = unknown word
        else:
            encoded.append(index + 3)  # real words start at 4
    return pad_sequences([encoded], maxlen=MAX_LEN)


# Step 3: prediction function
def predict_sentiment(review):
    score = float(model.predict(preprocess_text(review), verbose=0)[0][0])
    sentiment = "Positive" if score > 0.5 else "Negative"
    return sentiment, score


# Step 4: the Streamlit page
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

user_input = st.text_area("Movie review")

if st.button("Classify"):
    if user_input.strip():
        sentiment, score = predict_sentiment(user_input)
        st.write(f"Sentiment: **{sentiment}**")
        st.write(f"Prediction score: {score:.3f}")
        st.progress(score)
    else:
        st.warning("Please type a review first.")
else:
    st.write("Please enter a movie review.")
