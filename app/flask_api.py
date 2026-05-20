# ============================================================
# FILE: app/flask_api.py
# PURPOSE: Flask backend API — receives messages via HTTP
#          requests, runs prediction, returns JSON response
# MEMBER: Member 3
# USED BY: Member 4 (Streamlit app sends requests here)
# ============================================================


# --- IMPORTS ---

from flask import Flask, request, jsonify
# Flask  : the web framework — creates our API server
# request: lets us read data sent TO our API
# jsonify: converts Python dict → JSON response
#          JSON is the standard format for API communication

from flask_cors import CORS
# CORS = Cross-Origin Resource Sharing
# Without this, Streamlit (on port 8501) cannot talk to
# Flask (on port 5000) — browser blocks it for security
# CORS(app) tells Flask: "allow requests from other ports"

import sys
# sys: lets us modify Python's module search path

import os
# os: for file path operations

# Add the src/ folder to Python's path
# Without this, Python cannot find predict.py
# Because predict.py is in src/ but we are running from app/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
# os.path.dirname(__file__) = current file's folder = app/
# '..' goes one level up = spam-detector/
# 'src' goes into src/ folder
# So Python now knows to look in spam-detector/src/ for imports

from predict import predict, load_model
# Import our prediction functions from src/predict.py
# predict()    : predicts a single message
# load_model() : loads vectorizer + model from model/ folder

import logging
# logging: professional way to print messages in a server
# Better than print() for a running API server
# Logs show timestamps, severity levels, etc.


# ============================================================
# APP SETUP
# ============================================================

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    # INFO level: show general info messages
    # Other levels: DEBUG, WARNING, ERROR, CRITICAL

    format='%(asctime)s — %(levelname)s — %(message)s'
    # Format: "2024-01-15 10:30:45 — INFO — Model loaded"
)

logger = logging.getLogger(__name__)
# Create a logger for this file specifically

# Create the Flask application
app = Flask(__name__)
# __name__ tells Flask this is the main module
# Flask uses this to find template/static folders

# Enable CORS for all routes
CORS(app)
# Now any frontend (Streamlit, browser, etc.) can call our API


# ============================================================
# LOAD MODEL AT STARTUP
# We load the model ONCE when the server starts
# NOT inside the route functions
# Loading inside routes = model reloads on EVERY request = slow
# ============================================================

logger.info("Loading model and vectorizer...")

# Build absolute path to model/ folder
# This ensures the path works regardless of where you run the file
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
# os.path.abspath gives full path: C:\Users\...\spam-detector\app
# os.path.dirname gives folder:    C:\Users\...\spam-detector\app

MODEL_DIR  = os.path.join(BASE_DIR, '..', 'model')
# Goes up one level then into model/
# Result: C:\Users\...\spam-detector\model

try:
    vectorizer, model = load_model(model_dir=MODEL_DIR)
    logger.info("Model loaded successfully!")
    logger.info(f"Model type: {type(model).__name__}")
    MODEL_LOADED = True
    # Flag to track if model loaded — used in health check route

except FileNotFoundError as e:
    logger.error(f"Model loading failed: {e}")
    MODEL_LOADED = False
    # Server still starts but returns errors when predicting
    # This is better than crashing — at least we know WHY it failed


# ============================================================
# ROUTE 1: GET /
# PURPOSE: Root route — confirms API is running
# ============================================================

@app.route('/', methods=['GET'])
def home():
    """
    Simple route to confirm the server is alive.
    Visit http://localhost:5000/ in browser to see this.
    """
    return jsonify({
        'status' : 'running',
        'message': 'Spam Detector API is live!',
        'version': '1.0.0',
        'endpoints': {
            'predict' : 'POST /predict',
            'health'  : 'GET  /health',
            'batch'   : 'POST /batch'
        }
    })
    # jsonify converts our Python dict to a JSON response
    # The browser/Streamlit receives this as JSON text


# ============================================================
# ROUTE 2: GET /health
# PURPOSE: Health check — tells us if model is loaded properly
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    """
    Returns server health status.
    Useful for debugging — is the model actually loaded?
    """
    return jsonify({
        'status'      : 'healthy' if MODEL_LOADED else 'unhealthy',
        'model_loaded': MODEL_LOADED,
        'model_type'  : type(model).__name__ if MODEL_LOADED else 'None'
    })


# ============================================================
# ROUTE 3: POST /predict  ← THE MAIN ROUTE
# PURPOSE: Receive a message → return spam/ham prediction
# This is what Member 4's Streamlit app will call
# ============================================================

@app.route('/predict', methods=['POST'])
def predict_route():
    """
    Accepts POST request with JSON body:
    {
        "message": "You won a FREE prize! Call now!"
    }

    Returns JSON response:
    {
        "label"      : "spam",
        "confidence" : 97.32,
        "cleaned"    : "won free prize call",
        "status"     : "success"
    }
    """

    # STEP 1: Check if model is loaded
    if not MODEL_LOADED:
        return jsonify({
            'error' : 'Model not loaded. Run src/train.py first.',
            'status': 'error'
        }), 503
        # 503 = Service Unavailable HTTP status code

    # STEP 2: Get JSON data from the request
    data = request.get_json()
    # request.get_json() reads the JSON body sent by Streamlit
    # Example: {"message": "You won a prize!"}

    # STEP 3: Validate the request
    if not data:
        return jsonify({
            'error' : 'No JSON data received. Send {"message": "your text"}',
            'status': 'error'
        }), 400
        # 400 = Bad Request

    message = data.get('message', '')
    # data.get('message', '') safely gets 'message' key
    # If key doesn't exist, returns '' instead of crashing

    if not message or not message.strip():
        return jsonify({
            'error' : 'Message is empty. Please provide a non-empty message.',
            'status': 'error'
        }), 400

    # STEP 4: Run prediction
    logger.info(f"Predicting message: '{message[:50]}...'")
    # Log first 50 chars of message for debugging

    result = predict(message, vectorizer, model)
    # Call our predict() function from src/predict.py
    # Returns: {'label': 'spam', 'confidence': 97.32, 'cleaned': '...'}

    # STEP 5: Build and return response
    response = {
        'label'      : result['label'],
        # 'spam' or 'ham'

        'confidence' : result['confidence'],
        # e.g. 97.32 (percentage)

        'cleaned'    : result['cleaned'],
        # cleaned version of message (what model saw)

        'status'     : 'success'
    }

    logger.info(f"Result: {result['label']} ({result['confidence']}%)")

    return jsonify(response), 200
    # 200 = OK — successful response


# ============================================================
# ROUTE 4: POST /batch
# PURPOSE: Predict multiple messages at once
#          Useful for testing many messages in one request
# ============================================================

@app.route('/batch', methods=['POST'])
def batch_predict_route():
    """
    Accepts POST request with JSON body:
    {
        "messages": [
            "You won a prize!",
            "Hey, see you tomorrow",
            "FREE entry now!"
        ]
    }

    Returns list of predictions for each message.
    """

    if not MODEL_LOADED:
        return jsonify({
            'error' : 'Model not loaded.',
            'status': 'error'
        }), 503

    data = request.get_json()

    if not data or 'messages' not in data:
        return jsonify({
            'error' : 'Send {"messages": ["msg1", "msg2", ...]}',
            'status': 'error'
        }), 400

    messages = data['messages']
    # Get the list of messages

    if not isinstance(messages, list):
        return jsonify({
            'error' : '"messages" must be a list.',
            'status': 'error'
        }), 400

    if len(messages) > 100:
        return jsonify({
            'error' : 'Maximum 100 messages per batch request.',
            'status': 'error'
        }), 400
        # Prevent abuse — don't allow huge batches

    # Predict each message
    results = []
    for msg in messages:
        result = predict(str(msg), vectorizer, model)
        results.append({
            'original'  : msg,
            'label'     : result['label'],
            'confidence': result['confidence']
        })

    spam_count = sum(1 for r in results if r['label'] == 'spam')
    ham_count  = len(results) - spam_count

    return jsonify({
        'results'    : results,
        'total'      : len(results),
        'spam_count' : spam_count,
        'ham_count'  : ham_count,
        'status'     : 'success'
    }), 200


# ============================================================
# MAIN BLOCK — Start the Flask server
# ============================================================

if __name__ == '__main__':
    logger.info("Starting Spam Detector Flask API...")
    logger.info("API will be available at: http://localhost:5000")
    logger.info("Press CTRL+C to stop the server")

    app.run(
        host='0.0.0.0',
        # 0.0.0.0 means accept connections from any device
        # on the same network — needed for team testing

        port=5000,
        # Port 5000 is Flask's default port
        # Streamlit runs on 8501 — different ports, no conflict

        debug=True
        # debug=True: auto-restarts server when you save changes
        # Shows detailed errors in browser
        # IMPORTANT: set debug=False before real deployment
    )