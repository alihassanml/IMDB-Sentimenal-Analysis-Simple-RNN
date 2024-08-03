import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the pre-trained model
my_model = load_model('RNN_model.h5')

# Get the word index
word_index = imdb.get_word_index()
Reverse_word = {value: key for key, value in word_index.items()}
max_len = 500

def decode_review(encoded_review):
    return ' '.join(Reverse_word.get(i - 3, '?') for i in encoded_review)

def preprocess_text(text):
    words = text.lower().split()
    encode_review = [word_index.get(word, 2) + 3 for word in words]
    return sequence.pad_sequences([encode_review], maxlen=max_len)

def predict_sentiment(text):
    encoded_review = preprocess_text(text)
    prediction = my_model.predict(encoded_review)[0][0]
    if prediction > 0.5:
        return 'Positive'
    else:
        return 'Negative'

# Streamlit app
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment")

# Input text box
review_text = st.text_area("")

if st.button("Predict Sentiment"):
    sentiment = predict_sentiment(review_text)
    st.write(f"Sentiment: {sentiment}")

