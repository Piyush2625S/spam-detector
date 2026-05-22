🛡️ Spam SMS / Email Detector

A machine learning web application that classifies SMS and email messages as spam or legitimate (ham) in real time using Natural Language Processing and supervised learning.

---

## 📌 Project Overview

Every day millions of users receive spam messages that
cause financial fraud and privacy violations. This system
automatically identifies spam messages with 97%+ accuracy
using Natural Language Processing and Machine Learning.

---

## 🎯 Problem Statement

Traditional spam filters fail to detect modern spam patterns.
This project builds an intelligent spam detection system
trained on real SMS data that classifies any message as
Spam or Ham instantly.

---

## ✅ Features

- Detects spam messages instantly
- Shows confidence percentage for every prediction
- Trained on 5,574 real SMS messages
- Simple and clean web interface
- Deployed online — accessible from any device

---

## 🔴 Live Demo

> Start Flask API → Run Streamlit → Open `http://localhost:8501`

bash
# Terminal 1
python app/flask_api.py

# Terminal 2
streamlit run app/app.py


---

## 📌 Project Overview

| Property     | Details                          |
|--------------|----------------------------------|
| Domain       | NLP + Machine Learning           |
| Dataset      | UCI SMS Spam Collection (Kaggle) |
| Records      | 5,169 messages after cleaning    |
| Train/Test   | 80% / 20% split                  |
| Best Model   | Logistic Regression / Naive Bayes|
| Accuracy     | ~98%                             |
| Stack        | Python, Flask, Streamlit         |

---

## 🧠 How It Works


Raw Message
    ↓
Text Cleaning (lowercase, remove URLs, phones, punctuation)
    ↓
Stopword Removal (NLTK)
    ↓
Stemming (PorterStemmer)
    ↓
TF-IDF Vectorization (top 5000 features, 1-2 ngrams)
    ↓
ML Model (Naive Bayes / Logistic Regression)
    ↓
Prediction: SPAM or HAM + Confidence Score


---

## 📁 Folder Structure


spam-detector/
├── data/
│   ├── spam.csv                  ← raw dataset (UCI)
│   └── cleaned_spam.csv          ← after preprocessing
├── model/
│   ├── best_model.pkl            ← saved best model
│   ├── tfidf_vectorizer.pkl      ← saved vectorizer
│   ├── naive_bayes_model.pkl
│   └── logistic_regression_model.pkl
├── src/
│   ├── clean.py                  ← data preprocessing
│   ├── train.py                  ← model training + evaluation
│   └── predict.py                ← prediction logic
├── app/
│   ├── flask_api.py              ← REST API backend
│   └── app.py                    ← Streamlit frontend
├── notebooks/
│   └── EDA.ipynb                 ← exploratory data analysis
├── screenshots/
│   ├── model_comparison.png
│   ├── confusion_naive_bayes.png
│   └── confusion_logistic_regression.png
├── requirements.txt
└── README.md


---

## 👥 Team Roles

| Member   | Responsibility                                      |
|----------|-----------------------------------------------------|
| Member 1 | Data cleaning — `src/clean.py`                      |
| Member 2 | Model training + evaluation — `src/train.py`, `src/predict.py` |
| Member 3 | Flask REST API — `app/flask_api.py`                 |
| Member 4 | Streamlit UI — `app/app.py`                         |
| Member 5 | Documentation, README, report, presentation slides  |

---

## ⚙️ Tech Stack

| Layer        | Technology                          |
|--------------|-------------------------------------|
| Language     | Python 3.10                         |
| ML           | scikit-learn (Naive Bayes, LR)      |
| NLP          | NLTK (stopwords, PorterStemmer)     |
| Vectorizer   | TF-IDF (TfidfVectorizer)            |
| Backend API  | Flask + flask-cors                  |
| Frontend UI  | Streamlit                           |
| Data         | pandas, numpy                       |
| Visualization| matplotlib, seaborn                 |
| Model Saving | pickle                              |
| Dataset      | UCI SMS Spam Collection             |

---

## 🚀 Setup & Installation

### 1. Clone the repository

bash
git clone https://github.com/YOUR_USERNAME/spam-detector.git
cd spam-detector


### 2. Create virtual environment

bash
py -3.10 -m venv venv
venv\Scripts\activate        # Windows CMD


### 3. Install dependencies

bash
pip install -r requirements.txt


### 4. Download dataset

Download `spam.csv` from [Kaggle UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
and place it in the `data/` folder.

### 5. Run the pipeline

bash
# Step 1 — Clean data
python src/clean.py

# Step 2 — Train model
python src/train.py

# Step 3 — Test prediction
python src/predict.py


### 6. Start the application

bash
# Terminal 1 — Flask API
python app/flask_api.py

# Terminal 2 — Streamlit UI
streamlit run app/app.py


Open browser at: `http://localhost:8501`

---

## 📊 Model Results

| Metric    | Naive Bayes | Logistic Regression |
|-----------|-------------|----------------------|
| Accuracy  | 97.87%      | 98.26%               |
| Precision | 98.23%      | 97.77%               |
| Recall    | 89.47%      | 91.58%               |
| F1 Score  | 93.65%      | 94.58%               |

> Logistic Regression selected as best model based on F1 Score.

---

## 🔌 API Reference

### `GET /health`
Returns API and model status.

**Response:**
json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "LogisticRegression"
}


### `POST /predict`
Classifies a single message.

**Request:**
json
{
  "message": "FREE prize! Call now to claim your reward!"
}


**Response:**
json
{
  "label": "spam",
  "confidence": 99.67,
  "cleaned": "free prize call claim reward",
  "status": "success"
}


### `POST /batch`
Classifies multiple messages at once.

**Request:**
json
{
  "messages": ["Free prize now!", "Hey see you tomorrow"]
}


---

## 📦 requirements.txt


pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
nltk==3.8.1
streamlit==1.28.0
flask==3.0.0
flask-cors==4.0.0
matplotlib==3.7.2
seaborn==0.12.2


---

## 📄 License

This project is built for academic purposes.
Dataset credit: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)