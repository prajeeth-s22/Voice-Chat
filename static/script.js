document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const micBtn = document.getElementById('mic-btn');
    const statusText = document.getElementById('status-text');
    const recognizedSpeechEl = document.getElementById('recognized-speech');
    const textInput = document.getElementById('text-input');
    const sendBtn = document.getElementById('send-btn');
    const chatbotResponseEl = document.getElementById('chatbot-response');
    const ttsBtn = document.getElementById('tts-btn');
    const intentBadge = document.getElementById('intent-badge');
    const confidenceBadge = document.getElementById('confidence-badge');

    // Speech Recognition Setup
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;
    let isListening = false;

    if (!SpeechRecognition) {
        statusText.textContent = 'Speech Recognition API not supported in this browser.';
        statusText.style.color = 'red';
        micBtn.disabled = true;
    } else {
        recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.continuous = false;

        recognition.onstart = () => {
            isListening = true;
            statusText.textContent = 'Listening...';
            micBtn.classList.add('listening');
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            recognizedSpeechEl.textContent = transcript;
            recognizedSpeechEl.classList.remove('placeholder');
            sendMessage(transcript);
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error', event.error);
            isListening = false;
            micBtn.classList.remove('listening');
            
            let errorMsg = 'Error during recognition.';
            if (event.error === 'not-allowed') {
                errorMsg = 'Microphone permission denied.';
            } else if (event.error === 'no-speech') {
                errorMsg = 'No speech detected.';
            } else if (event.error === 'network') {
                errorMsg = 'Network error during recognition.';
            }
            
            statusText.textContent = errorMsg;
            setTimeout(() => {
                if (!isListening) statusText.textContent = 'Ready';
            }, 3000);
        };

        recognition.onend = () => {
            isListening = false;
            micBtn.classList.remove('listening');
            if (statusText.textContent === 'Listening...') {
                statusText.textContent = 'Processing...';
            }
        };

        micBtn.addEventListener('click', () => {
            if (isListening) {
                recognition.stop();
            } else {
                try {
                    recognition.start();
                } catch (e) {
                    console.error("Could not start recognition:", e);
                }
            }
        });
    }

    // Chat Function
    async function sendMessage(text) {
        if (!text || text.trim() === '') return;
        
        statusText.textContent = 'Processing...';
        recognizedSpeechEl.textContent = text;
        recognizedSpeechEl.classList.remove('placeholder');
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: text })
            });

            if (!response.ok) {
                throw new Error(`Server returned ${response.status}`);
            }

            const data = await response.json();
            
            // Update UI
            chatbotResponseEl.textContent = data.response || "No response provided.";
            chatbotResponseEl.classList.remove('placeholder');
            
            intentBadge.textContent = data.intent || "Unknown";
            
            const confidence = data.confidence ? (data.confidence * 100).toFixed(1) : 0;
            confidenceBadge.textContent = `${confidence}%`;
            
            statusText.textContent = 'Response Generated';
            ttsBtn.disabled = false;
            
            // Optionally auto-play TTS by un-commenting next line:
            // speak(data.response);
            
        } catch (error) {
            console.error('Chat error:', error);
            chatbotResponseEl.textContent = `Error: ${error.message}. Is the Flask backend running?`;
            chatbotResponseEl.classList.remove('placeholder');
            statusText.textContent = 'Error occurred';
        }
    }

    // Text Input Handlers
    const handleTextInput = () => {
        const text = textInput.value;
        if (text.trim() !== '') {
            sendMessage(text);
            textInput.value = '';
        }
    };

    sendBtn.addEventListener('click', handleTextInput);
    
    textInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleTextInput();
        }
    });

    // Text-to-Speech
    let lastSpokenText = '';
    
    function speak(text) {
        if (!('speechSynthesis' in window)) {
            console.warn('Text-to-Speech not supported');
            return;
        }
        
        window.speechSynthesis.cancel(); // Cancel any ongoing speech
        
        const utterance = new SpeechSynthesisUtterance(text);
        
        // Try to find an English voice
        const voices = window.speechSynthesis.getVoices();
        const englishVoice = voices.find(voice => voice.lang.startsWith('en-'));
        if (englishVoice) {
            utterance.voice = englishVoice;
        }
        
        window.speechSynthesis.speak(utterance);
    }

    ttsBtn.addEventListener('click', () => {
        const text = chatbotResponseEl.textContent;
        if (text && !chatbotResponseEl.classList.contains('placeholder')) {
            speak(text);
        }
    });
    
    // Ensure voices are loaded before we might need them
    if ('speechSynthesis' in window) {
        window.speechSynthesis.getVoices();
    }
});
