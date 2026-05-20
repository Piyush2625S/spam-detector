# ============================================================
# FILE: src/train.py
# PURPOSE: Load cleaned data → Train 2 models → Evaluate →
#          Save best model and vectorizer as .pkl files
# MEMBER: Member 2
# ============================================================


# --- IMPORTS ---

import pandas as pd
# pandas: to load and work with our cleaned_spam.csv

import numpy as np
# numpy: for numerical operations (used internally by sklearn)

import os
# os: to create folders for saving model files

import pickle
# pickle: to SAVE our trained model to a file
# Think of it as freezing the trained model so we can reuse it later
# Without pickle, we'd have to retrain every single time

import matplotlib
matplotlib.use('Agg')
# 'Agg' is a non-interactive backend for matplotlib
# This prevents popup windows when saving charts on Windows
# MUST be set BEFORE importing pyplot

import matplotlib.pyplot as plt
# pyplot: to draw charts and graphs

import seaborn as sns
# seaborn: makes prettier charts with less code
# Built on top of matplotlib

from sklearn.model_selection import train_test_split
# train_test_split: splits our data into:
#   - Training set (80%) → model learns from this
#   - Testing set  (20%) → we test on this to check accuracy

from sklearn.feature_extraction.text import TfidfVectorizer
# TfidfVectorizer: converts text into numbers
# ML models cannot understand words — only numbers
# TF-IDF gives higher scores to IMPORTANT rare words
# and lower scores to COMMON unimportant words
# Example: "free" in spam gets a high score
#          "the" gets a near-zero score everywhere

from sklearn.naive_bayes import MultinomialNB
# MultinomialNB: Naive Bayes classifier
# Very fast, works extremely well for text/spam detection
# This is the classic algorithm for spam filters

from sklearn.linear_model import LogisticRegression
# LogisticRegression: another classification algorithm
# Usually slightly more accurate than Naive Bayes
# We train BOTH and pick the better one

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
# These are evaluation tools — they tell us HOW GOOD our model is
# accuracy_score   : overall % correct predictions
# precision_score  : of all predicted spam, how many were actually spam?
# recall_score     : of all actual spam, how many did we catch?
# f1_score         : balance between precision and recall
# confusion_matrix : table showing correct vs wrong predictions
# classification_report: full summary of all above metrics


# ============================================================
# FUNCTION 1: load_data()
# PURPOSE: Load cleaned_spam.csv and prepare X and y
# ============================================================

def load_data(filepath='data/cleaned_spam.csv'):
    """
    Loads the cleaned dataset.
    Returns:
        X = list of cleaned messages (input to model)
        y = list of labels 0/1 (what model should predict)
    """

    print("=" * 55)
    print("STEP 1: Loading cleaned dataset...")
    print("=" * 55)

    df = pd.read_csv(filepath)
    # Read the cleaned CSV that Member 1 produced

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop any rows where cleaned_message is empty after cleaning
    df = df.dropna(subset=['cleaned_message'])
    # Some messages might become empty after removing stopwords
    # Example: a message that was only "Hi" becomes "" after cleaning
    # We remove those to avoid errors

    df['cleaned_message'] = df['cleaned_message'].astype(str)
    # Make absolutely sure every value is a string
    # Sometimes pandas reads empty cells as float NaN
    # astype(str) converts everything to string type

    X = df['cleaned_message'].values
    # X = our INPUT data (the cleaned text messages)
    # .values converts pandas Series to a numpy array
    # Example: ["free entri win cash", "ok see you later", ...]

    y = df['label'].values
    # y = our TARGET data (what we want to predict)
    # Example: [1, 0, 0, 1, 0, ...]
    # 1 = spam, 0 = ham

    print(f"Total samples : {len(X)}")
    print(f"Spam messages : {sum(y == 1)}")
    print(f"Ham  messages : {sum(y == 0)}")

    return X, y


# ============================================================
# FUNCTION 2: vectorize_text()
# PURPOSE: Convert text messages into numerical TF-IDF matrix
# ============================================================

def vectorize_text(X_train, X_test):
    """
    Converts raw text into numbers using TF-IDF.
    
    Why do we pass X_train and X_test separately?
    → We FIT (learn vocabulary) only on training data
    → We TRANSFORM (apply) on both train and test
    → This prevents 'data leakage' — test data must stay unseen
    """

    print("\n" + "=" * 55)
    print("STEP 3: Vectorizing text with TF-IDF...")
    print("=" * 55)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        # Only keep the top 5000 most important words
        # Ignoring rare words that appear only once or twice
        # This keeps the model fast and avoids overfitting

        ngram_range=(1, 2),
        # ngram_range=(1,2) means we consider:
        # Single words: "free", "win", "cash"
        # AND word pairs: "free entry", "win prize", "call now"
        # Word pairs capture more context than single words alone

        min_df=2
        # min_df=2 means: ignore words that appear in fewer than 2 messages
        # Very rare words are usually typos or noise — not useful
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    # fit_transform does TWO things at once:
    # 1. fit    → learns the vocabulary from training data
    # 2. transform → converts training text to numbers
    # Result: a matrix where rows=messages, columns=words, values=TF-IDF scores

    X_test_tfidf = vectorizer.transform(X_test)
    # transform only (NO fit) on test data
    # Uses the SAME vocabulary learned from training data
    # This is critical — test data must be treated as "unseen"

    print(f"Vocabulary size     : {len(vectorizer.vocabulary_)} words")
    print(f"Training matrix shape: {X_train_tfidf.shape}")
    # Shape example: (4135, 5000)
    # Means: 4135 training messages, each represented by 5000 word scores

    print(f"Testing  matrix shape: {X_test_tfidf.shape}")

    return X_train_tfidf, X_test_tfidf, vectorizer
    # We return the vectorizer too — we need to SAVE it with pickle
    # When predicting new messages later, we must use the SAME vectorizer


# ============================================================
# FUNCTION 3: train_and_evaluate()
# PURPOSE: Train both models, print metrics, return results
# ============================================================

def train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test):
    """
    Trains Naive Bayes and Logistic Regression.
    Evaluates both. Returns both trained models + their scores.
    """

    print("\n" + "=" * 55)
    print("STEP 4: Training and Evaluating Models...")
    print("=" * 55)

    # Dictionary to store both models
    # Key = model name, Value = the model object
    models = {
        'Naive Bayes': MultinomialNB(alpha=0.1),
        # alpha=0.1 is the smoothing parameter
        # It handles words in test data that weren't in training data
        # Default is 1.0 — we use 0.1 for slightly better accuracy

        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            # max_iter=1000: allow up to 1000 iterations to find best fit
            # Default is 100 — sometimes not enough for text data
            # Without this you get a ConvergenceWarning

            C=1.0,
            # C controls regularization (prevents overfitting)
            # C=1.0 is the standard balanced value

            solver='lbfgs'
            # lbfgs is a good general-purpose solver for classification
        )
    }

    results = {}
    # Empty dictionary to store evaluation scores for each model
    # We'll fill this as we train each model

    trained_models = {}
    # Stores the actual trained model objects
    # We need these to save with pickle later

    for name, model in models.items():
        # Loop through both models one by one
        # name  = 'Naive Bayes' or 'Logistic Regression'
        # model = the actual model object

        print(f"\n--- Training: {name} ---")

        # TRAIN the model
        model.fit(X_train_tfidf, y_train)
        # .fit() is where actual learning happens
        # The model looks at all training messages + their labels
        # and figures out patterns that separate spam from ham

        # PREDICT on test data
        y_pred = model.predict(X_test_tfidf)
        # Give the trained model unseen test messages
        # It returns predicted labels: [0, 1, 0, 0, 1, ...]

        # CALCULATE metrics
        acc  = accuracy_score(y_test, y_pred)
        # What % of all predictions were correct?

        prec = precision_score(y_test, y_pred)
        # Of messages we CALLED spam, what % were actually spam?
        # High precision = fewer false alarms (ham marked as spam)

        rec  = recall_score(y_test, y_pred)
        # Of actual spam messages, what % did we CATCH?
        # High recall = fewer spam messages slipping through

        f1   = f1_score(y_test, y_pred)
        # Harmonic mean of precision and recall
        # Best single number to compare models

        cm   = confusion_matrix(y_test, y_pred)
        # 2x2 table:
        # [[True Ham,  False Spam],
        #  [False Ham, True Spam ]]

        # Print results
        print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"\n  Full Classification Report:")
        print(classification_report(y_test, y_pred,
              target_names=['Ham (0)', 'Spam (1)']))

        # Store results for comparison later
        results[name] = {
            'accuracy'  : acc,
            'precision' : prec,
            'recall'    : rec,
            'f1'        : f1,
            'confusion' : cm,
            'y_pred'    : y_pred
        }

        trained_models[name] = model
        # Save the trained model object for pickling later

    return results, trained_models


# ============================================================
# FUNCTION 4: save_charts()
# PURPOSE: Generate and save evaluation charts as PNG images
# ============================================================

def save_charts(results, y_test, output_dir='screenshots'):
    """
    Creates 3 charts:
    1. Model comparison bar chart (accuracy, precision, recall, F1)
    2. Confusion matrix for Naive Bayes
    3. Confusion matrix for Logistic Regression
    
    Saves them to screenshots/ folder
    """

    print("\n" + "=" * 55)
    print("STEP 5: Saving evaluation charts...")
    print("=" * 55)

    os.makedirs(output_dir, exist_ok=True)
    # Create screenshots/ folder if it doesn't exist

    # ── CHART 1: Model Comparison Bar Chart ──
    metrics  = ['accuracy', 'precision', 'recall', 'f1']
    # The 4 metrics we want to compare

    nb_scores = [results['Naive Bayes'][m] for m in metrics]
    lr_scores = [results['Logistic Regression'][m] for m in metrics]
    # Get scores for each model

    x = np.arange(len(metrics))
    # x = [0, 1, 2, 3] — positions on x-axis for each metric

    width = 0.35
    # Width of each bar — 0.35 gives space between paired bars

    fig, ax = plt.subplots(figsize=(10, 6))
    # Create a figure (canvas) and axes (the actual plot area)
    # figsize=(10,6) means 10 inches wide, 6 inches tall

    bars1 = ax.bar(x - width/2, nb_scores, width,
                   label='Naive Bayes', color='steelblue')
    bars2 = ax.bar(x + width/2, lr_scores, width,
                   label='Logistic Regression', color='coral')
    # Draw two sets of bars side by side
    # x - width/2 shifts Naive Bayes bars slightly LEFT
    # x + width/2 shifts Logistic Regression bars slightly RIGHT

    # Add value labels on top of each bar
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Naive Bayes vs Logistic Regression')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    ax.set_ylim(0.8, 1.02)
    # y-axis from 0.8 to 1.02 so differences are visible
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    # Automatically adjust spacing so nothing gets cut off

    chart1_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {chart1_path}")

    # ── CHARTS 2 & 3: Confusion Matrices ──
    for name in ['Naive Bayes', 'Logistic Regression']:
        cm = results[name]['confusion']
        # Get the confusion matrix we calculated earlier

        fig, ax = plt.subplots(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            # annot=True: write the numbers inside each cell

            fmt='d',
            # fmt='d': format numbers as integers (not 1.0, just 1000)

            cmap='Blues',
            # Color scheme: darker blue = higher number

            xticklabels=['Predicted Ham', 'Predicted Spam'],
            yticklabels=['Actual Ham', 'Actual Spam'],
            ax=ax
        )

        ax.set_title(f'Confusion Matrix — {name}')
        ax.set_ylabel('Actual Label')
        ax.set_xlabel('Predicted Label')

        plt.tight_layout()

        safe_name = name.lower().replace(' ', '_')
        # Convert "Naive Bayes" → "naive_bayes" for filename

        chart_path = os.path.join(output_dir, f'confusion_{safe_name}.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Chart saved: {chart_path}")


# ============================================================
# FUNCTION 5: save_models()
# PURPOSE: Save best model + vectorizer as .pkl files
# ============================================================

def save_models(trained_models, vectorizer, results,
                model_dir='model'):
    """
    Compares both models by F1 score.
    Saves the BEST model + vectorizer as pickle files.
    Also saves BOTH models individually.
    """

    print("\n" + "=" * 55)
    print("STEP 6: Saving models with pickle...")
    print("=" * 55)

    os.makedirs(model_dir, exist_ok=True)
    # Create model/ folder if it doesn't exist

    # ── Save vectorizer (ALWAYS needed) ──
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        # open file in 'wb' mode = write binary
        # pickle files are binary — not human readable
        pickle.dump(vectorizer, f)
    print(f"  Vectorizer saved : {vectorizer_path}")

    # ── Save both models individually ──
    for name, model in trained_models.items():
        safe_name = name.lower().replace(' ', '_')
        path = os.path.join(model_dir, f'{safe_name}_model.pkl')
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  Model saved      : {path}")

    # ── Find best model by F1 score ──
    best_name = max(results, key=lambda x: results[x]['f1'])
    # max() finds the key with the highest f1 score
    # lambda x: results[x]['f1'] tells max() what to compare

    best_model = trained_models[best_name]

    best_path = os.path.join(model_dir, 'best_model.pkl')
    with open(best_path, 'wb') as f:
        pickle.dump(best_model, f)

    print(f"\n  Best model       : {best_name}")
    print(f"  Best F1 score    : {results[best_name]['f1']:.4f}")
    print(f"  Saved as         : {best_path}")

    return best_name


# ============================================================
# MAIN BLOCK
# ============================================================

if __name__ == '__main__':

    # STEP 1: Load data
    X, y = load_data('data/cleaned_spam.csv')

    # STEP 2: Split into train and test sets
    print("\n" + "=" * 55)
    print("STEP 2: Splitting data into Train/Test sets...")
    print("=" * 55)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        # 20% of data goes to testing, 80% to training

        random_state=42,
        # random_state=42 makes the split reproducible
        # Every time you run this, you get the SAME split
        # Without this, results change every run

        stratify=y
        # stratify=y ensures BOTH train and test sets have
        # the same spam/ham ratio as the full dataset
        # Without this, you might get all spam in test by accident
    )

    print(f"Training samples : {len(X_train)}")
    print(f"Testing  samples : {len(X_test)}")
    print(f"Train spam count : {sum(y_train == 1)}")
    print(f"Test  spam count : {sum(y_test  == 1)}")

    # STEP 3: Vectorize text
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(
        X_train, X_test
    )

    # STEP 4: Train and evaluate models
    results, trained_models = train_and_evaluate(
        X_train_tfidf, X_test_tfidf, y_train, y_test
    )

    # STEP 5: Save charts
    save_charts(results, y_test)

    # STEP 6: Save models
    best_name = save_models(trained_models, vectorizer, results)

    # FINAL SUMMARY
    print("\n" + "=" * 55)
    print("TRAINING COMPLETE — FINAL SUMMARY")
    print("=" * 55)
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy  : {res['accuracy']*100:.2f}%")
        print(f"  Precision : {res['precision']:.4f}")
        print(f"  Recall    : {res['recall']:.4f}")
        print(f"  F1 Score  : {res['f1']:.4f}")

    print(f"\n✅ Best Model     : {best_name}")
    print(f"✅ Files saved to : model/")
    print(f"✅ Charts saved to: screenshots/")
    print("=" * 55)