# ============================================================
# FILE: src/predict.py
# PURPOSE: Load saved model + vectorizer → predict new messages
# MEMBER: Member 2
# THIS FILE IS USED BY: Member 3 (Flask API) and Member 4 (Streamlit)
# ============================================================


# --- IMPORTS ---

import pickle
# pickle: to LOAD our saved .pkl model and vectorizer files
# We trained and saved them in train.py
# Now we just load and reuse — no retraining needed

import re
# re: Regular Expressions — for cleaning new input messages
# Same cleaning we did in clean.py

import string
# string: for punctuation removal
# string.punctuation = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

import os
# os: to check if model files exist before loading

import nltk
# nltk: Natural Language Toolkit for stopwords and stemming

from nltk.corpus import stopwords
# stopwords: common words to remove ("the", "is", "a", etc.)

from nltk.stem import PorterStemmer
# PorterStemmer: reduces words to root form
# "running" → "run", "winner" → "winner"

# Download required nltk data silently
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


# --- INITIALIZE TOOLS ---

stemmer = PorterStemmer()
# Create stemmer object once — reuse for every prediction

stop_words = set(stopwords.words('english'))
# Load English stopwords into a set for fast lookup


# ============================================================
# FUNCTION 1: clean_input()
# PURPOSE: Clean a raw input message exactly like clean.py did
# WHY: The model was trained on cleaned text
#      So predictions must also be cleaned the same way
#      If we skip cleaning, accuracy drops badly
# ============================================================

def clean_input(text):
    """
    Cleans a single raw message for prediction.
    Must match EXACTLY what clean.py does.
    
    Input : "WINNER!! Claim your FREE prize now! Call 08712300150"
    Output: "winner claim free prize call"
    """

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove phone numbers
    text = re.sub(r'\b\d{7,}\b', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Remove stopwords and apply stemming
    tokens = [
        stemmer.stem(word)
        for word in text.split()
        if word not in stop_words
        if word.isalpha()
    ]

    return ' '.join(tokens)
    # Returns cleaned string ready for vectorizer


# ============================================================
# FUNCTION 2: load_model()
# PURPOSE: Load saved vectorizer and model from model/ folder
# ============================================================

def load_model(model_dir='model'):
    """
    Loads the TF-IDF vectorizer and best model from disk.
    Returns both objects ready for prediction.
    """

    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    model_path      = os.path.join(model_dir, 'best_model.pkl')

    # Check if files exist before trying to load
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"Vectorizer not found at: {vectorizer_path}\n"
            f"Make sure you ran src/train.py first."
        )
        # This gives a clear error instead of a confusing crash

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            f"Make sure you ran src/train.py first."
        )

    # Load vectorizer from pickle file
    with open(vectorizer_path, 'rb') as f:
        # 'rb' = read binary
        vectorizer = pickle.load(f)
    # vectorizer is now the EXACT same object we saved in train.py

    # Load best model from pickle file
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    # model is now ready to make predictions

    return vectorizer, model


# ============================================================
# FUNCTION 3: predict()
# PURPOSE: Main prediction function
#          Takes raw message → returns label + confidence
# THIS IS THE FUNCTION Member 3 and Member 4 WILL IMPORT
# ============================================================

def predict(message, vectorizer, model):
    """
    Takes a raw message string and returns prediction.

    Input:
        message    : raw text string (e.g. "You won a prize!")
        vectorizer : loaded TfidfVectorizer object
        model      : loaded trained model object

    Returns a dictionary:
    {
        'label'      : 'spam' or 'ham',
        'confidence' : 97.32  (percentage, e.g. 97.32%)
        'cleaned'    : 'won prize'  (what model actually saw)
    }
    """

    # STEP 1: Validate input
    if not message or not message.strip():
        # If message is empty or just spaces
        return {
            'label'      : 'unknown',
            'confidence' : 0.0,
            'cleaned'    : ''
        }

    # STEP 2: Clean the input message
    cleaned = clean_input(message)
    # Example: "FREE PRIZE!! Call now" → "free prize call"

    if not cleaned:
        # If message becomes empty after cleaning
        # Example: a message that was only "Hi !!" becomes ""
        return {
            'label'      : 'ham',
            'confidence' : 99.0,
            'cleaned'    : ''
        }
        # Very short messages with no content are almost always ham

    # STEP 3: Vectorize the cleaned message
    message_tfidf = vectorizer.transform([cleaned])
    # We pass a LIST with one item: [cleaned]
    # vectorizer.transform() expects a list, not a single string
    # Result: a sparse matrix with TF-IDF scores

    # STEP 4: Get prediction (0 or 1)
    prediction = model.predict(message_tfidf)[0]
    # .predict() returns an array — [0] gets the first (only) element
    # prediction = 0 means ham
    # prediction = 1 means spam

    # STEP 5: Get confidence score (probability)
    probabilities = model.predict_proba(message_tfidf)[0]
    # predict_proba returns probability for EACH class
    # Example: [0.03, 0.97]
    # Index 0 = probability of ham  = 3%
    # Index 1 = probability of spam = 97%

    confidence = probabilities[prediction] * 100
    # Get the probability of whichever class was predicted
    # Multiply by 100 to convert to percentage
    # Example: 0.9732 → 97.32%

    # STEP 6: Convert numeric label to human-readable string
    label = 'spam' if prediction == 1 else 'ham'

    return {
        'label'      : label,
        'confidence' : round(confidence, 2),
        # round() to 2 decimal places: 97.3241 → 97.32
        'cleaned'    : cleaned
        # We return cleaned text so UI can show what model saw
    }


# ============================================================
# FUNCTION 4: predict_batch()
# PURPOSE: Predict multiple messages at once
#          Useful for testing and the notebook demo
# ============================================================

def predict_batch(messages, vectorizer, model):
    """
    Takes a list of messages → returns list of prediction dicts.
    
    Input : ["Win cash now!", "Hey, are you coming?", "FREE prize"]
    Output: [{'label':'spam',...}, {'label':'ham',...}, ...]
    """

    results = []
    for message in messages:
        result = predict(message, vectorizer, model)
        result['original'] = message
        # Add the original message to result for reference
        results.append(result)

    return results


# ============================================================
# MAIN BLOCK — Test the predictor with sample messages
# ============================================================

if __name__ == '__main__':

    print("=" * 55)
    print("Loading model and vectorizer...")
    print("=" * 55)

    # Load model and vectorizer from model/ folder
    vectorizer, model = load_model()

    print("Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")

    # ── Test messages ──
    # These are real-world style messages to verify prediction works
    test_messages = [
        # Clear spam examples
        "WINNER!! You have been selected to receive a £900 prize! "
        "Call 08712300150 to claim NOW!",

        "FREE entry in 2 a wkly comp to win FA Cup final tkts! "
        "Text FA to 87121",

        "Congratulations! You've won a FREE iPhone. "
        "Click http://claim-prize.com to collect",

        "URGENT: Your account will be suspended. "
        "Verify now at http://fake-bank.com",

        # Clear ham examples
        "Hey, are you coming to the meeting tomorrow at 3pm?",

        "I'll be home late tonight, don't wait up for dinner.",

        "Can you pick up some milk on your way home please?",

        "The project deadline has been moved to next Friday.",

        # Edge cases
        "Hi",
        # Very short message

        "Call me when you get this",
        # Legitimate message with 'call' (spam trigger word)

        "You have won",
        # Borderline — sounds spammy but short
    ]

    print(f"\nTesting {len(test_messages)} messages...\n")
    print("=" * 55)

    # Run batch prediction
    results = predict_batch(test_messages, vectorizer, model)

    # Print results in a clean table format
    print(f"{'#':<4} {'LABEL':<6} {'CONF%':<8} {'MESSAGE'}")
    print("-" * 55)

    for i, result in enumerate(results, 1):
        # Truncate long messages for display
        msg_display = result['original'][:45]
        if len(result['original']) > 45:
            msg_display += '...'

        # Add visual indicator
        indicator = '🔴' if result['label'] == 'spam' else '🟢'

        print(
            f"{i:<4} "
            f"{result['label'].upper():<6} "
            f"{result['confidence']:<8} "
            f"{indicator} {msg_display}"
        )

    print("-" * 55)
    spam_count = sum(1 for r in results if r['label'] == 'spam')
    ham_count  = sum(1 for r in results if r['label'] == 'ham')
    print(f"Spam detected: {spam_count} | Ham detected: {ham_count}")

    print("\n" + "=" * 55)
    print("PREDICT.PY — WORKING CORRECTLY ✅")
    print("Member 3 can now import predict() into Flask API")
    print("Member 4 can now import predict() into Streamlit")
    print("=" * 55)