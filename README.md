# 🚫 Spam SMS/Email Detector

A Machine Learning based system that detects whether
a message is Spam or Legitimate (Ham) using NLP techniques.

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

## 🛠️ Tech Stack

| Layer           | Technology              |
| --------------- | ----------------------- |
| Language        | Python 3.10             |
| ML Algorithm    | Naive Bayes             |
| Text Processing | TF-IDF Vectorizer       |
| NLP Library     | NLTK                    |
| UI              | Streamlit               |
| Backend         | Flask                   |
| Visualization   | Matplotlib, Seaborn     |
| Dataset         | UCI SMS Spam Collection |
| Deployment      | Streamlit Cloud         |
| Version Control | GitHub                  |

---

## 📁 Folder Structure

spam-detector/
├── data/
│ └── spam.csv
├── model/
│ └── spam_model.pkl
│ └── tfidf_vectorizer.pkl
├── src/
│ ├── clean.py
│ ├── train.py
│ └── predict.py
├── app/
│ └── app.py
├── notebooks/
│ └── EDA.ipynb
├── screenshots/
│ └── demo.png
│ └── confusion_matrix.png
├── requirements.txt
└── README.md

---

## ⚙️ How It Works

User inputs message
↓
Text cleaned (lowercase, remove symbols, stopwords)
↓
TF-IDF converts text to numerical features
↓
Naive Bayes model predicts spam or ham
↓
Result shown with confidence percentage

---

## 📊 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 97.8% |
| F1 Score  | 96.4% |
| Precision | 97.1% |
| Recall    | 95.8% |

---

## 📸 Screenshots

### Home Screen

![Home Screen](screenshots/demo.png)

### Confusion Matrix

![Confusion Matrix](screenshots/confusion_matrix.png)

---

## 🚀 How to Run Locally

### Step 1 — Clone the repo

bash
git clone https://github.com/Piyush2625S/spam-detector.git
cd spam-detector

### Step 2 — Install dependencies

bash
pip install -r requirements.txt

### Step 3 — Train the model

bash
python src/train.py

### Step 4 — Run the app

bash
streamlit run app/app.py

### Step 5 — Open in browser

http://localhost:8501

---

## 🌐 Live Demo

👉 [Click here to try the live app](https://spam-detector.streamlit.app)

---

## 📂 Dataset

- Name: UCI SMS Spam Collection Dataset
- Source: Kaggle
- Size: 5,574 messages
- Labels: Ham (4,827) and Spam (747)
- Link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

---

## 👥 Team

| Member   | Role                           |
| -------- | ------------------------------ |
| Member 1 | Data Cleaning and EDA          |
| Member 2 | Model Training and Evaluation  |
| Member 3 | Flask Backend API              |
| Member 4 | Streamlit UI and Deployment    |
| Member 5 | Documentation and Presentation |

---

## 🔮 Future Improvements

- Add Hindi and Hinglish spam detection
- Implement BERT for better contextual understanding
- Build browser extension for real time detection
- Add email header analysis for deeper spam detection

---

## 📄 License

This project is open source and available under the MIT License.
