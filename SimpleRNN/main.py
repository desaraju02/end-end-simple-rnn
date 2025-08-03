import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


word_index = imdb.get_word_index()
reverse_word_index = { value: key for (key, value) in word_index.items() }

model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper functions
# Function to decode the review text
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])


# Function to preprocess the review text
def preprocess_review(review, maxlen=500):
    words = review.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # 2 is the index for OOV
    padded_review = sequence.pad_sequences([encoded_review], maxlen=maxlen)
    return padded_review


# Step 3: Prediction function
def predict_sentiment(review):
    padded_review = preprocess_review(review)
    prediction = model.predict(padded_review)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


# Design Steamlit app
import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")

# Input text area for user to enter review
user_input = st.text_area("Enter your movie review:")

if st.button("Predict Sentiment"):
    # preprocess_input = preprocess_review(user_input)  
    sentiment, score = predict_sentiment(user_input)
    st.write(f"Sentiment: {sentiment} (Score: {score:.2f})")
