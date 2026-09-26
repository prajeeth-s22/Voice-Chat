"""Flask backend for the Gemini-powered VoiceBot."""

import json
import os

from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB

GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-3.8-flash')
SYSTEM_INSTRUCTION = (
    'You are VoiceBot, a concise and helpful AI assistant. '
    'Answer the user directly in plain text and keep replies under 150 words.'
)
VOICE_RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'transcript': {'type': 'string'},
        'response': {'type': 'string'},
    },
    'required': ['transcript', 'response'],
}


def gemini_client():
    """Create a client without ever exposing its API key to the browser."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY is not configured on the server.')
    return genai.Client(api_key=api_key)


def text_response(message):
    client = gemini_client()
    result = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.4,
            max_output_tokens=250,
        ),
    )
    reply = (result.text or '').strip()
    if not reply:
        raise ValueError('Gemini returned an empty response.')
    return {
        'text': message,
        'intent': 'gemini',
        'confidence': 1.0,
        'response': reply,
        'response_source': 'gemini',
        'gemini_model': GEMINI_MODEL,
    }


def voice_response(audio_bytes, mime_type):
    client = gemini_client()
    result = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            (
                'Transcribe the speech in this audio and answer the speaker as '
                'VoiceBot. If there is no intelligible speech, return an empty '
                'transcript and ask the speaker to try again.'
            ),
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type='application/json',
            response_schema=VOICE_RESPONSE_SCHEMA,
            temperature=0.4,
            max_output_tokens=300,
        ),
    )
    payload = result.parsed
    if payload is None:
        payload = json.loads(result.text or '{}')
    if not isinstance(payload, dict):
        payload = payload.model_dump()

    transcript = str(payload.get('transcript', '')).strip()
    reply = str(payload.get('response', '')).strip()
    if not reply:
        reply = "I couldn't hear any clear speech. Please try recording again."

    return {
        'text': transcript,
        'intent': 'gemini',
        'confidence': 1.0 if transcript else 0.0,
        'response': reply,
        'response_source': 'gemini',
        'gemini_model': GEMINI_MODEL,
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON.'}), 400
    data = request.get_json(silent=True) or {}
    message = str(data.get('message', '')).strip()
    if not message:
        return jsonify({'error': 'Message cannot be empty.'}), 400
    if len(message) > 4_000:
        return jsonify({'error': 'Message too long. Maximum 4000 characters.'}), 400
    try:
        return jsonify(text_response(message))
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 503
    except Exception as exc:
        app.logger.exception('Gemini text request failed.')
        status = getattr(exc, 'code', None) or getattr(exc, 'status_code', None)
        detail = f' (provider status {status})' if status else ''
        return jsonify({'error': f'Gemini could not process the message{detail}. Please try again.'}), 502


@app.route('/voice-chat', methods=['POST'])
def voice_chat():
    audio_file = request.files.get('audio')
    if audio_file is not None:
        audio_bytes = audio_file.read()
        mime_type = audio_file.mimetype or 'audio/webm'
    else:
        audio_bytes = request.get_data()
        mime_type = request.mimetype or 'audio/webm'

    if not audio_bytes:
        return jsonify({'error': 'No audio file provided in request.'}), 400

    try:
        return jsonify(voice_response(audio_bytes, mime_type))
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 503
    except Exception as exc:
        app.logger.exception('Gemini voice request failed.')
        status = getattr(exc, 'code', None) or getattr(exc, 'status_code', None)
        detail = f' (provider status {status})' if status else ''
        return jsonify({'error': f'Gemini could not process the recording{detail}. Please try again.'}), 502


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'gemini_enabled': bool(os.environ.get('GEMINI_API_KEY')),
        'gemini_model': GEMINI_MODEL,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
