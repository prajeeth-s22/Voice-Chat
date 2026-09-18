# 🎙️ Voice-Enabled Chatbot using Speech Recognition & Deep Learning

A fully functional online chatbot that accepts **voice input** from a user, converts speech to text using the Web Speech API, classifies the user's intent using a **Deep Learning** neural network, and generates an appropriate response — all deployed as a publicly accessible web application.

---

## 📋 Table of Contents

- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Deep Learning Model](#-deep-learning-model)
- [Installation](#-installation)
- [Training the Model](#-training-the-model)
- [Running Locally](#-running-locally)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [Example Input/Output](#-example-inputoutput)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Live Demo](#-live-demo)

---

## ✨ Features

- **Voice Input**: Click-to-speak microphone using the browser's Speech Recognition API
- **Text Input Fallback**: Type messages when microphone is unavailable
- **Deep Learning Intent Classification**: TF-IDF + Neural Network predicts user intent with confidence scores
- **Low-Confidence Handling**: Gracefully handles unknown or ambiguous queries (threshold: 0.35)
- **Text-to-Speech**: Optional audio playback of chatbot responses using the Web Speech API
- **Real-time Pipeline Display**: Shows recognized speech, predicted intent, confidence score, and response
- **Responsive Design**: Mobile-friendly dark futuristic AI dashboard theme
- **Production Deployment**: Deployed via Gunicorn on Render with HTTPS

---

## 🛠 Technology Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3, JavaScript (Vanilla) |
| **Speech Recognition** | Web Speech API (`webkitSpeechRecognition`) |
| **Text-to-Speech** | Web Speech API (`speechSynthesis`) |
| **Backend** | Python, Flask |
| **Deep Learning** | TensorFlow / Keras |
| **NLP Preprocessing** | scikit-learn (TF-IDF Vectorizer) |
| **Deployment** | Gunicorn, Render |

---

## 🏗 Architecture

```
User speaks into microphone
        ↓
Browser Microphone (getUserMedia)
        ↓
Web Speech API (SpeechRecognition)
        ↓
Recognized Text
        ↓
POST /chat → Flask Backend
        ↓
TF-IDF Vectorization (scikit-learn)
        ↓
Neural Network Prediction (Keras)
        ↓
Predicted Intent + Confidence Score
        ↓
Response Selection (from intents.json)
        ↓
JSON Response → Frontend Display
        ↓
Optional Text-to-Speech (speechSynthesis)
```

---

## 📊 Dataset

The chatbot uses a custom intent dataset (`intents.json`) with **13 intent categories**:

| Intent | Description | Example Pattern |
|--------|-------------|-----------------|
| `greeting` | Greetings | "Hello", "Hi there" |
| `goodbye` | Farewells | "Bye", "See you later" |
| `thanks` | Gratitude | "Thank you", "Thanks" |
| `name` | Bot's identity | "What is your name?" |
| `capabilities` | Bot's abilities | "What can you do?" |
| `artificial_intelligence` | AI concepts | "What is AI?" |
| `machine_learning` | ML concepts | "Explain machine learning" |
| `deep_learning` | DL concepts | "What is deep learning?" |
| `speech_recognition` | Speech tech | "How does voice recognition work?" |
| `chatbot` | Chatbot concepts | "What is a chatbot?" |
| `deployment` | Deployment topics | "How to deploy a web app?" |
| `project` | About this project | "Tell me about this project" |
| `help` | Help requests | "I need help" |

Each intent contains 10–15 diverse training patterns and 3–5 response variations.

---

## 🧠 Deep Learning Model

### Architecture

```
Input Layer (TF-IDF features)
        ↓
Dense Layer (128 neurons, ReLU activation)
        ↓
Dropout Layer (rate = 0.35)
        ↓
Dense Layer (64 neurons, ReLU activation)
        ↓
Dropout Layer (rate = 0.25)
        ↓
Output Layer (13 neurons, Softmax activation)
```

### Training Configuration

- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: Categorical Cross-Entropy
- **Metrics**: Accuracy
- **Epochs**: 200
- **Batch Size**: 16
- **Train/Val Split**: 80% / 20% (stratified)
- **Feature Extraction**: TF-IDF (max 1000 features, unigrams + bigrams)

### Saved Artifacts

| File | Description |
|------|-------------|
| `chatbot_model.keras` | Trained Keras model |
| `vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `label_encoder.pkl` | Fitted label encoder |
| `training_history.png` | Training accuracy/loss plot |

---

## 📥 Installation

### Prerequisites

- Python 3.10 or 3.11
- pip

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd voice-chatbot

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

```bash
python train_model.py
```

This will:
1. Load `intents.json`
2. Create TF-IDF features from training patterns
3. Encode intent labels
4. Build and train the neural network
5. Save model files (`chatbot_model.keras`, `vectorizer.pkl`, `label_encoder.pkl`)
6. Generate training plot (`training_history.png`)

Expected output:
```
============================================================
  Voice-Enabled Chatbot — Model Training
============================================================

[1/7] Loading intents dataset...
[2/7] Preparing training data...
[3/7] Creating TF-IDF features...
[4/7] Encoding labels...
[5/7] Splitting data (80% train / 20% validation)...
[6/7] Building and training neural network...
[7/7] Saving model and artifacts...

============================================================
  Training complete! Model is ready for deployment.
============================================================
```

---

## 🚀 Running Locally

```bash
# Make sure model files exist (run train_model.py first)
python app.py
```

Open your browser and visit: `http://localhost:5000`

---

## 🌐 Deployment

### Deploy to Render

1. Push code to a GitHub repository
2. Go to [render.com](https://render.com) and create a new **Web Service**
3. Connect your GitHub repository
4. Configure:
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
5. Deploy

> **Note**: The free tier may take 30–60 seconds to cold-start after inactivity.

---

## 📁 Project Structure

```
voice-chatbot/
│
├── app.py                  # Flask backend application
├── train_model.py          # Model training script
├── intents.json            # Intent dataset
├── requirements.txt        # Python dependencies
├── Procfile                # Deployment configuration
├── README.md               # This file
│
├── chatbot_model.keras     # Trained Keras model (generated)
├── vectorizer.pkl          # TF-IDF vectorizer (generated)
├── label_encoder.pkl       # Label encoder (generated)
├── training_history.png    # Training plot (generated)
│
├── templates/
│   └── index.html          # Chatbot frontend
│
└── static/
    ├── style.css           # Dark futuristic theme
    └── script.js           # Speech recognition & chat logic
```

---

## 💬 Example Input/Output

| User Input | Intent | Confidence | Response |
|-----------|--------|------------|----------|
| "Hello" | greeting | ~95% | "Hello! How can I assist you today?" |
| "What is deep learning?" | deep_learning | ~92% | Educational response about deep learning |
| "How do I deploy this?" | deployment | ~88% | Response about deployment process |
| "Thank you" | thanks | ~97% | "You're welcome! Happy to help!" |
| "asdfghjkl" | unknown | <35% | "I'm not sure I understood that..." |

---

## ⚠️ Limitations

- **Browser Dependency**: Speech Recognition requires Chrome/Edge (best support for Web Speech API)
- **Internet Required**: Web Speech API sends audio to cloud servers for processing
- **Dataset Size**: Limited to 13 intent categories; accuracy depends on training data quality
- **Language**: Currently English (en-US) only
- **Cold Start**: Free-tier deployment may have initial loading delay
- **No Context Memory**: Each query is treated independently (no conversation history)

---

## 🔮 Future Enhancements

- Add conversation history and context-aware responses
- Support multiple languages (Hindi, Tamil, Telugu, etc.)
- Expand the intent dataset with more categories
- Implement LSTM/Transformer-based models for better text understanding
- Add user authentication and personalized responses
- Implement conversation logging and analytics dashboard
- Add more sophisticated NLP preprocessing (stemming, lemmatization)
- Support file upload and document-based Q&A

---

## 🌍 Live Demo

**Live URL**: `<DEPLOYMENT_URL_PLACEHOLDER>`

**Repository**: `<GITHUB_REPO_PLACEHOLDER>`

---

## 📄 License

This project is developed for academic purposes as part of a university project submission.

---

## 👤 Author

Developed as part of the course project: **"Voice-Enabled Chatbot using Speech Recognition and Deep Learning"**
