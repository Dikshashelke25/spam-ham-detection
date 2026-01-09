# 📩 Spam & Ham SMS Classification using NLP

An end-to-end Machine Learning & NLP project that classifies SMS or email messages as Spam or Ham (Not Spam).
The project includes EDA, preprocessing, model training, evaluation, and deployment using Streamlit.

## 🚀 Project Overview

Spam messages are unsolicited, promotional, or potentially harmful texts.
This project aims to automatically detect spam messages using Natural Language Processing (NLP) and Machine Learning techniques.

The final output is a deployed web application where users can input a message and instantly get a prediction.

## 🧠 Problem Statement

To build an intelligent system that can:

Analyze the text content of SMS messages

Learn patterns associated with spam messages

Accurately classify messages as Spam or Ham

## 📊 Dataset Description

The dataset contains SMS messages labeled as spam or ham.

Column	Description
Category	spam or ham
Message	Actual text content of the message

Spam → Unwanted or promotional messages

Ham → Legitimate messages

## 🛠️ Tech Stack

Python

Scikit-learn

NLTK

Pandas & NumPy

Streamlit

Pickle

## 🧩 Project Workflow (Sprint-wise)

🟢 Sprint 0: Exploratory Data Analysis (EDA)

Data understanding and cleaning

Class distribution analysis

Message length analysis

Visualization of spam vs ham patterns

🟡 Sprint 1: Data Preprocessing

Text cleaning (lowercasing, tokenization)

Stopword removal

Stemming using Porter Stemmer

Feature engineering

🔵 Sprint 2: Model Development

TF-IDF vectorization

Train-test split

Model training using Multinomial Naive Bayes

Performance evaluation using accuracy, precision, recall, and F1-score

🟣 Sprint 3: NLP Architecture

End-to-end NLP pipeline

Consistent preprocessing for training and inference

🔴 Sprint 4: Model Deployment

Streamlit web application

Real-time message classification

User-friendly interface

## 🏗️ NLP Architecture
Input Message
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Naive Bayes Classifier
      ↓
Spam / Ham Prediction

## 📈 Model Performance

High accuracy on test data

Strong precision for spam detection

Suitable for real-world text classification tasks

## 🖥️ Streamlit Web App

Features:

Text input for SMS/email

Instant spam detection

Clear visual feedback

Run Locally:
streamlit run app.py

## 📁 Project Structure

spam-ham-classifier/
│
├── app.py
├── sms-spam-detection.ipynb
├── spam.csv
├── vectorizer.pkl
├── model.pkl
├── requirements.txt
└── README.md

## 📦 Installation & Setup

Clone the repository

git clone https://github.com/your-username/spam-ham-classifier.git
cd spam-ham-classifier


Install dependencies

pip install -r requirements.txt


Run the app

streamlit run app.py

## ✅ Key Learnings

Text preprocessing techniques in NLP

Feature extraction using TF-IDF

Naive Bayes for text classification

Model deployment with Streamlit

End-to-end ML pipeline development

## 🔮 Future Enhancements

Try advanced models (Logistic Regression, SVM)

Add deep learning models (LSTM, BERT)

Improve UI and UX

Add message confidence score

Deploy on cloud platforms
