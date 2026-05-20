# ============================================================
# FILE: src/clean.py
# PURPOSE: Load raw spam.csv, clean the text, save clean data
# MEMBER: Member 1
# ============================================================

# --- IMPORTS ---
# We import the tools (libraries) we need at the top

import pandas as pd
# pandas helps us work with spreadsheet-like data (rows and columns)
# We use it to read spam.csv and save the cleaned data

import nltk
# nltk = Natural Language Toolkit
# It gives us tools to work with human language text

import re
# re = Regular Expressions
# It helps us find and remove patterns in text
# Example: remove all numbers, remove all punctuation

import string
# string is a built-in Python library
# string.punctuation gives us all punctuation marks: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

import os
# os helps us work with files and folders on Windows
# We use it to create folders if they don't exist

from nltk.corpus import stopwords
# stopwords are common English words that don't add meaning
# Examples: "the", "is", "a", "an", "in", "on"
# Removing them helps the model focus on important words

from nltk.stem import PorterStemmer
# PorterStemmer reduces words to their root form
# Examples: "running" → "run", "eating" → "eat", "loved" → "love"
# This helps the model treat similar words as the same word


# --- DOWNLOAD NLTK DATA ---
# These are data files nltk needs to work
# If you already ran the setup command, this just confirms they exist
# silent=True means don't print anything if already downloaded

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


# --- INITIALIZE TOOLS ---

stemmer = PorterStemmer()
# We create one PorterStemmer object and reuse it for every word
# Think of it as turning on the "word reducer" machine once

stop_words = set(stopwords.words('english'))
# We get all English stopwords and store them in a set
# A set is like a list but much faster when checking "is this word in here?"
# Example: stop_words = {"the", "is", "a", "an", "in", ...}


# ============================================================
# FUNCTION 1: clean_text()
# PURPOSE: Takes one raw SMS/email string → returns cleaned string
# ============================================================

def clean_text(text):
    """
    This function takes a single raw message like:
    "FREE entry! Win £1000 CASH! Call NOW 08712300150!!"
    
    And returns a cleaned version like:
    "free entri win cash call"
    """

    # STEP 1: Convert to lowercase
    # "FREE" and "free" should be treated as the same word
    text = text.lower()
    # Example: "FREE Win CASH" → "free win cash"

    # STEP 2: Remove URLs (website links)
    # re.sub means "find this pattern and replace with something"
    # r'http\S+' is the pattern — it means: find anything starting with "http"
    # We replace it with an empty string "" (which deletes it)
    text = re.sub(r'http\S+', '', text)
    # Example: "visit http://win.com now" → "visit  now"

    # STEP 3: Remove email addresses
    # Pattern: find anything@anything.anything
    text = re.sub(r'\S+@\S+', '', text)
    # Example: "email us at win@spam.com" → "email us at "

    # STEP 4: Remove phone numbers
    # Pattern: find sequences of digits (7 or more in a row)
    text = re.sub(r'\b\d{7,}\b', '', text)
    # Example: "call 08712300150 now" → "call  now"

    # STEP 5: Remove punctuation
    # str.maketrans creates a translation table
    # It says: replace every punctuation character with nothing ""
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Example: "hello!! win£££" → "hello win"

    # STEP 6: Remove extra whitespace
    # After all the removals above, we might have double/triple spaces
    # .split() breaks text into individual words (ignores extra spaces)
    # ' '.join() stitches them back together with single spaces
    text = ' '.join(text.split())
    # Example: "hello   win   cash" → "hello win cash"

    # STEP 7: Remove stopwords AND apply stemming at the same time
    # We loop through every word in the text
    # If the word is NOT a stopword → we stem it and keep it
    # If the word IS a stopword → we skip it (don't add to list)
    tokens = [
        stemmer.stem(word)              # reduce word to root form
        for word in text.split()        # loop through each word
        if word not in stop_words       # only keep non-stopwords
        if word.isalpha()               # only keep words (no lone numbers/symbols)
    ]
    # Example: ["free", "entry", "win", "cash", "call", "now"]
    # After stemming: ["free", "entri", "win", "cash", "call", "now"]
    # Note: "entri" is correct — PorterStemmer is aggressive but consistent

    # STEP 8: Join the list back into a single string
    cleaned = ' '.join(tokens)
    # Example: ["free", "entri", "win", "cash"] → "free entri win cash"

    return cleaned
    # We return the fully cleaned text


# ============================================================
# FUNCTION 2: load_and_clean()
# PURPOSE: Load spam.csv → clean all messages → save clean CSV
# ============================================================

def load_and_clean(input_path='data/spam.csv', output_path='data/cleaned_spam.csv'):
    """
    input_path  = where to read the raw data from
    output_path = where to save the cleaned data to
    """

    print("=" * 50)
    print("STEP 1: Loading dataset...")
    print("=" * 50)

    # Read the CSV file into a DataFrame (like an Excel table in Python)
    # encoding='latin-1' is needed because spam.csv has special characters
    # Without this, you get a UnicodeDecodeError
    df = pd.read_csv(input_path, encoding='latin-1')

    print(f"Raw data shape: {df.shape}")
    # .shape gives us (rows, columns)
    # Example output: Raw data shape: (5572, 5)

    print(f"Columns found: {list(df.columns)}")
    # Shows us the column names in the CSV


    print("\nSTEP 2: Fixing column names...")

    # The spam.csv from Kaggle has 5 columns but only 2 matter:
    # Column 'v1' = label (ham or spam)
    # Column 'v2' = actual message text
    # The other 3 columns (Unnamed: 2, 3, 4) are garbage — we drop them

    # Keep only the first two columns
    df = df[['v1', 'v2']]

    # Rename them to something readable
    df.columns = ['label', 'message']
    # Now our DataFrame has two columns: 'label' and 'message'

    print(f"Columns renamed to: {list(df.columns)}")


    print("\nSTEP 3: Checking for missing values...")

    # dropna() removes any row where label or message is empty/null
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"Rows before dropping nulls: {before}")
    print(f"Rows after  dropping nulls: {after}")
    print(f"Null rows removed: {before - after}")


    print("\nSTEP 4: Removing duplicate messages...")

    # drop_duplicates() removes rows where the message is exactly the same
    before = len(df)
    df = df.drop_duplicates(subset='message')
    after = len(df)
    print(f"Rows before removing duplicates: {before}")
    print(f"Rows after  removing duplicates: {after}")
    print(f"Duplicates removed: {before - after}")


    print("\nSTEP 5: Checking class distribution...")

    # value_counts() counts how many ham vs spam messages we have
    print(df['label'].value_counts())
    # Expected output:
    # ham     4516
    # spam     653


    print("\nSTEP 6: Encoding labels...")

    # Machine learning models need numbers, not words
    # We convert: 'ham' → 0  and  'spam' → 1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    # Now the 'label' column has 0s and 1s instead of 'ham' and 'spam'

    print("Labels encoded: ham=0, spam=1")
    print(df['label'].value_counts())


    print("\nSTEP 7: Cleaning all messages (this may take a moment)...")

    # Apply our clean_text() function to EVERY row in the 'message' column
    # .apply() means: run this function on each value one by one
    df['cleaned_message'] = df['message'].apply(clean_text)
    # This creates a NEW column 'cleaned_message' with cleaned text
    # The original 'message' column is kept for reference

    print("All messages cleaned successfully!")


    print("\nSTEP 8: Saving cleaned data...")

    # Create the data/ folder if it somehow doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # exist_ok=True means: don't crash if the folder already exists

    # Save the cleaned DataFrame to a new CSV file
    df.to_csv(output_path, index=False)
    # index=False means: don't write row numbers (0,1,2,3...) into the file

    print(f"Cleaned data saved to: {output_path}")


    print("\n" + "=" * 50)
    print("CLEANING COMPLETE — SUMMARY")
    print("=" * 50)
    print(f"Total messages after cleaning : {len(df)}")
    print(f"Ham  messages (label=0)        : {(df['label'] == 0).sum()}")
    print(f"Spam messages (label=1)        : {(df['label'] == 1).sum()}")
    print(f"Saved to                       : {output_path}")
    print("=" * 50)

    # Show 3 example rows so we can visually verify the cleaning worked
    print("\nSample cleaned rows:")
    print(df[['label', 'message', 'cleaned_message']].head(3).to_string())

    return df
    # We return the cleaned DataFrame
    # Member 2 can import this function and use it directly in train.py


# ============================================================
# MAIN BLOCK
# This runs only when you execute: python src/clean.py
# It does NOT run when another file imports clean.py
# ============================================================

if __name__ == '__main__':
    df = load_and_clean(
        input_path='data/spam.csv',
        output_path='data/cleaned_spam.csv'
    )