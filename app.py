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
import subprocess
import tempfile

import numpy as np

# Suppress TensorFlow info/warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
from google import genai
from google.genai import types
import whisper
import imageio_ffmpeg


# ─── App Configuration ───────────────────────────────────────────────
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Confidence threshold — below this, return a low-confidence response
CONFIDENCE_THRESHOLD = 0.35
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')
GEMINI_SYSTEM_INSTRUCTION = (
    'You are VoiceBot, a concise and helpful AI assistant. '
    'Answer the user directly in plain text. Keep responses under 150 words.'
)

# Low-confidence fallback responses
FALLBACK_RESPONSES = [
    "I'm not fully sure I understood that. Could you try rephrasing your question?",
    "I didn't quite catch that. Could you ask in a different way?",
    "I'm not confident about what you're asking. Please try asking in another way.",
    "Hmm, I'm not sure about that. Could you rephrase or ask something else?",
    "I couldn't understand that clearly. Could you try again with different words?",
]


def load_recording(audio_path):
    """Decode browser audio to Whisper's required 16 kHz mono float32 format.

    MediaRecorder produces WebM/Opus in most browsers. Librosa cannot
    reliably open that format on Windows without a separately installed
    FFmpeg. imageio-ffmpeg ships a compatible FFmpeg binary with the Python
    dependency, so this works on a clean Windows installation as well.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe, '-nostdin', '-loglevel', 'error', '-i', audio_path,
        '-f', 's16le', '-ac', '1', '-ar', '16000', '-'
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, timeout=60
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError('Audio decoding timed out. Please use a shorter recording.') from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode('utf-8', errors='replace').strip()
        raise ValueError(f'Unable to decode the audio recording: {detail or "unsupported audio format"}') from exc

    return np.frombuffer(completed.stdout, np.int16).astype(np.float32) / 32768.0


# ─── Load Model and Artifacts ────────────────────────────────────────
def load_model_artifacts():
    """Load the trained model, vectorizer, label encoder, intents, and Whisper STT model."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load trained Keras model
    model_path = os.path.join(base_dir, 'chatbot_model.keras')
    model = keras.models.load_model(model_path)
    print(f"[OK] Keras Intent Model loaded from: {model_path}")

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

    # Load local OpenAI Whisper model for Speech Recognition
    print("[INFO] Loading local OpenAI Whisper STT model (tiny.en)...")
    whisper_model = whisper.load_model('tiny.en')
    print("[OK] Local Whisper STT model loaded successfully.")

    # Build a lookup: tag -> list of responses
    response_map = {}
    for intent in intents_data['intents']:
        response_map[intent['tag']] = intent['responses']

    return model, vectorizer, label_encoder, response_map, whisper_model


# Load everything at startup
print("\n" + "=" * 50)
print("  Loading chatbot model and artifacts...")
print("=" * 50)
model, vectorizer, label_encoder, response_map, whisper_model = load_model_artifacts()
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


def generate_response(text):
    """Use Gemini when configured, otherwise retain the local model response."""
    local_result = predict_intent(text)
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        local_result['response_source'] = 'local_intent_model'
        return local_result

    try:
        client = genai.Client(api_key=api_key)
        gemini_result = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=GEMINI_SYSTEM_INSTRUCTION,
                temperature=0.4,
                max_output_tokens=250,
            ),
        )
        reply = (gemini_result.text or '').strip()
        if reply:
            local_result['response'] = reply
            local_result['response_source'] = 'gemini'
            local_result['gemini_model'] = GEMINI_MODEL
        else:
            local_result['response_source'] = 'local_intent_model'
    except Exception:
        # Do not expose provider errors or credentials to the browser.
        app.logger.exception('Gemini request failed; using the local response.')
        local_result['response_source'] = 'local_intent_model'

    return local_result


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
        result = generate_response(message)
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'An internal error occurred while processing your message.'
        }), 500


@app.route('/voice-chat', methods=['POST'])
def voice_chat():
    """Handle voice requests by processing audio locally using Whisper STT."""
    if 'audio' not in request.files and not request.data:
        return jsonify({'error': 'No audio file provided in request.'}), 400

    temp_path = None
    try:
        # Save uploaded audio file to a temporary location
        suffix = '.wav'
        if 'audio' in request.files:
            file = request.files['audio']
            filename = file.filename or 'speech.wav'
            ext = os.path.splitext(filename)[1]
            if ext:
                suffix = ext
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                file.save(tmp.name)
                temp_path = tmp.name
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(request.data)
                temp_path = tmp.name

        # Decode WebM/WAV/etc. into the format Whisper expects.
        audio_data = load_recording(temp_path)

        # Remove temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            temp_path = None

        if len(audio_data) == 0:
            return jsonify({'error': 'Audio recording was empty. Please try speaking again.'}), 400

        # Transcribe locally using OpenAI Whisper
        transcription_result = whisper_model.transcribe(audio_data, fp16=False)
        transcribed_text = transcription_result.get('text', '').strip()

        if not transcribed_text:
            return jsonify({
                'text': '',
                'intent': 'unknown',
                'confidence': 0.0,
                'response': "I couldn't hear any clear speech. Please try speaking into your microphone again."
            })

        # Predict intent using Keras model
        result = generate_response(transcribed_text)
        result['stt_model'] = 'OpenAI Whisper (tiny.en)'
        return jsonify(result)

    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        app.logger.error(f"Voice processing error: {str(e)}")
        return jsonify({'error': f'Voice processing failed: {str(e)}'}), 500


# ─── Health Check ─────────────────────────────────────────────────────
@app.route('/health')
def health():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'ok',
        'intent_model_loaded': model is not None,
        'whisper_stt_loaded': whisper_model is not None,
        'gemini_enabled': bool(os.environ.get('GEMINI_API_KEY'))
    })


# ─── Main ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
