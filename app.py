"""
app.py
======
Flask backend for the Voice-Enabled Chatbot.

Loads a pre-trained Keras intent classification model and serves
predictions via a REST API.

Routes:
  GET  /      — Serve the chatbot frontend
  POST /chat  — Accept user text, predict intent, return response

Usage:
  Development:  python app.py
  Production:   gunicorn app:app
"""

import json
import os
import pickle
import random

import numpy as np

# Suppress TensorFlow info/warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras


# ─── App Configuration ───────────────────────────────────────────────
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Confidence threshold — below this, return a low-confidence response
CONFIDENCE_THRESHOLD = 0.35

# Low-confidence fallback responses
FALLBACK_RESPONSES = [
    "I'm not fully sure I understood that. Could you try rephrasing your question?",
    "I didn't quite catch that. Could you ask in a different way?",
    "I'm not confident about what you're asking. Please try asking in another way.",
    "Hmm, I'm not sure about that. Could you rephrase or ask something else?",
    "I couldn't understand that clearly. Could you try again with different words?",
]


# ─── Load Model and Artifacts ────────────────────────────────────────
def load_model_artifacts():
    """Load the trained model, vectorizer, label encoder, and intents."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load trained Keras model
    model_path = os.path.join(base_dir, 'chatbot_model.keras')
    model = keras.models.load_model(model_path)
    print(f"[OK] Model loaded from: {model_path}")

    # Load TF-IDF vectorizer
    vec_path = os.path.join(base_dir, 'vectorizer.pkl')
    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"[OK] Vectorizer loaded from: {vec_path}")

    # Load label encoder
    le_path = os.path.join(base_dir, 'label_encoder.pkl')
    with open(le_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"[OK] Label encoder loaded from: {le_path}")

    # Load intents for response retrieval
    intents_path = os.path.join(base_dir, 'intents.json')
    with open(intents_path, 'r', encoding='utf-8') as f:
        intents_data = json.load(f)
    print(f"[OK] Intents loaded from: {intents_path}")

    # Build a lookup: tag -> list of responses
    response_map = {}
    for intent in intents_data['intents']:
        response_map[intent['tag']] = intent['responses']

    return model, vectorizer, label_encoder, response_map


# Load everything at startup
print("\n" + "=" * 50)
print("  Loading chatbot model and artifacts...")
print("=" * 50)
model, vectorizer, label_encoder, response_map = load_model_artifacts()
print("=" * 50)
print("  Chatbot is ready!")
print("=" * 50 + "\n")


# ─── Prediction Function ─────────────────────────────────────────────
def predict_intent(text):
    """
    Predict the intent of user text.

    Returns:
        dict with keys: text, intent, confidence, response
    """
    # Preprocess
    cleaned = text.lower().strip()

    # Transform using TF-IDF vectorizer
    features = vectorizer.transform([cleaned]).toarray()

    # Predict with the neural network
    prediction = model.predict(features, verbose=0)[0]

    # Get the top predicted class
    predicted_index = np.argmax(prediction)
    confidence = float(prediction[predicted_index])
    predicted_tag = label_encoder.inverse_transform([predicted_index])[0]

    # Check confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        response_text = random.choice(FALLBACK_RESPONSES)
        predicted_tag = "unknown"
    else:
        # Select a random response for the predicted intent
        responses = response_map.get(predicted_tag, FALLBACK_RESPONSES)
        response_text = random.choice(responses)

    return {
        'text': text,
        'intent': predicted_tag,
        'confidence': round(confidence, 4),
        'response': response_text
    }


# ─── Routes ───────────────────────────────────────────────────────────
@app.route('/')
def index():
    """Serve the chatbot frontend."""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    # Validate content type
    if not request.is_json:
        return jsonify({
            'error': 'Request must be JSON. Send Content-Type: application/json'
        }), 400

    data = request.get_json(silent=True)

    # Validate request body
    if data is None:
        return jsonify({
            'error': 'Invalid JSON body.'
        }), 400

    message = data.get('message', '').strip()

    # Validate message
    if not message:
        return jsonify({
            'error': 'Message cannot be empty. Provide {"message": "your text"}'
        }), 400

    if len(message) > 500:
        return jsonify({
            'error': 'Message too long. Maximum 500 characters.'
        }), 400

    # Predict intent and get response
    try:
        result = predict_intent(message)
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'An internal error occurred while processing your message.'
        }), 500


# ─── Health Check ─────────────────────────────────────────────────────
@app.route('/health')
def health():
    """Simple health check endpoint."""
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


# ─── Main ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
