# VoiceBot — Gemini AI

A Flask voice chatbot that records audio in the browser and sends it securely
to the Gemini API for transcription and a generated reply. No local speech or
machine-learning model is bundled with the application.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
$env:GEMINI_API_KEY = "your-api-key"
python app.py
```

Open `http://127.0.0.1:5000` and permit microphone access.

## Environment variables

| Name | Required | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | Yes | Gemini API key, configured only on the server or host. |
| `GEMINI_MODEL` | No | Gemini model name; defaults to `gemini-2.5-flash`. |

Never place the API key in frontend JavaScript, commit it, or add it to
`.env.example`. Configure it as a secret in Vercel or your hosting provider.

## Deploy to Vercel

1. Import this GitHub repository into Vercel.
2. Add `GEMINI_API_KEY` as a **Secret** in Project Settings → Environment Variables.
3. Deploy. The application uses the Python 3.12 runtime defined in `.python-version`.

The `/health` endpoint reports whether Gemini is configured without exposing
the key.
