# IMDB Sentimental Analysis Simple RNN

This repository contains a Streamlit application for predicting the sentiment of movie reviews using a pre-trained Simple RNN model. The model has been trained on the IMDB dataset, and the application allows users to input their own reviews to get sentiment predictions.

## Author

- **Ali Hassan**

## Features

- **Sentiment Analysis**: Predicts whether a given movie review is positive or negative.
- **Review Decoding**: Converts the encoded review back to its original text form for better visualization.

## Prerequisites

Make sure you have the following packages installed:

- `streamlit`
- `tensorflow`
- `numpy`

You can install them using:

```sh
pip install streamlit tensorflow numpy
```

## How to Run the Application

1. **Clone the repository**:

    ```sh
    git clone https://github.com/alihassanml/IMDB-Sentimental-Analysis-Simple-RNN.git
    cd IMDB-Sentimental-Analysis-Simple-RNN
    ```

2. **Download the pre-trained model**:
   
   Make sure to place your `RNN_model.h5` file in the root directory of the cloned repository.

3. **Run the Streamlit application**:

    ```sh
    streamlit run app.py
    ```

4. **Usage**:

    - Enter a movie review in the provided text area.
    - Click on the "Predict Sentiment" button to get the sentiment prediction (Positive or Negative).
    - Optionally, click on the "Decode Review" button to see the encoded review converted back to text.

## Code Overview

### app.py

This is the main application file containing the following key functions and components:

- **Loading the pre-trained model**:
    ```python
    my_model = load_model('RNN_model.h5')
    ```

- **Preprocessing and decoding functions**:
    ```python
    def decode_review(encoded_review):
        return ' '.join(Reverse_word.get(i - 3, '?') for i in encoded_review)

    def preprocess_text(text):
        words = text.lower().split()
        encode_review = [word_index.get(word, 2) + 3 for word in words]
        return sequence.pad_sequences([encode_review], maxlen=max_len)
    ```

- **Predict sentiment function**:
    ```python
    def predict_sentiment(text):
        encoded_review = preprocess_text(text)
        prediction = my_model.predict(encoded_review)[0][0]
        if prediction > 0.5:
            return 'Positive'
        else:
            return 'Negative'
    ```

- **Streamlit app components**:
    ```python
    st.title("Movie Review Sentiment Analysis")
    st.write("Enter a movie review to predict its sentiment:")
    
    review_text = st.text_area("Movie Review", "This movie was fantastic! The acting was great and the plot was thrilling")
    
    if st.button("Predict Sentiment"):
        sentiment = predict_sentiment(review_text)
        st.write(f"Sentiment: {sentiment}")

    if st.button("Decode Review"):
        encoded_review = preprocess_text(review_text)[0]
        decoded_review = decode_review(encoded_review)
        st.write(f"Decoded Review: {decoded_review}")
    ```

## License

This project is licensed under the MIT License.
