import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# NLTK Downloads 

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text Preprocessing Function 

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    words = []
    for i in tokens:
        if i.isalnum():
            words.append(i)

    words = [i for i in words if i not in stopwords.words('english')]
    words = [ps.stem(i) for i in words]

    return " ".join(words)


# Load Saved Model & Vectorizer

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


# Streamlit UI

st.set_page_config(page_title="Spam Classifier", page_icon="📩")
st.title("📩 Email / SMS Spam Classifier")
st.write("Enter a message below to check whether it is **Spam** or **Not Spam**.")

input_sms = st.text_area("✉️ Enter the message")


# Prediction

if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        prediction = model.predict(vector_input)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam Message")