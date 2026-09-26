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

    // Local Voice Recording (MediaRecorder) Setup
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    // Check microphone support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        statusText.textContent = 'Microphone API not supported in this browser.';
        statusText.style.color = 'red';
        micBtn.disabled = true;
    } else {
        micBtn.addEventListener('click', async () => {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        });
    }

    async function startRecording() {
        try {
            audioChunks = [];
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                // Stop all microphone audio tracks
                stream.getTracks().forEach(track => track.stop());
                
                const mimeType = mediaRecorder.mimeType || 'audio/webm';
                const audioBlob = new Blob(audioChunks, { type: mimeType });
                
                if (audioBlob.size < 100) {
                    statusText.textContent = 'Recording too short.';
                    return;
                }
                
                await sendAudioToBackend(audioBlob);
            };

            mediaRecorder.start();
            isRecording = true;
            statusText.textContent = 'Recording... Click mic to stop';
            statusText.style.color = '#00ffcc';
            micBtn.classList.add('listening');

        } catch (err) {
            console.error('Error accessing microphone:', err);
            statusText.textContent = 'Microphone access denied or unreadable.';
            statusText.style.color = '#ff4d4d';
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        isRecording = false;
        micBtn.classList.remove('listening');
        statusText.textContent = 'Sending recording to Gemini...';
        statusText.style.color = '#e0e0e0';
    }

    async function sendAudioToBackend(audioBlob) {
        recognizedSpeechEl.textContent = 'Processing audio with Gemini...';
        recognizedSpeechEl.classList.add('placeholder');
        chatbotResponseEl.textContent = 'Thinking...';
        chatbotResponseEl.classList.add('placeholder');

        const formData = new FormData();
        const extension = audioBlob.type.includes('webm') ? 'webm' : 'wav';
        formData.append('audio', audioBlob, `speech.${extension}`);

        try {
            const response = await fetch('/voice-chat', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `Server error ${response.status}`);
            }

            const data = await response.json();
            
            // Display recognized text
            recognizedSpeechEl.textContent = data.text || "(No speech detected)";
            recognizedSpeechEl.classList.remove('placeholder');

            // Display chatbot response
            chatbotResponseEl.textContent = data.response || "No response provided.";
            chatbotResponseEl.classList.remove('placeholder');
            
            intentBadge.textContent = data.intent || "Unknown";
            
            const confidence = data.confidence !== undefined ? (data.confidence * 100).toFixed(1) : 0;
            confidenceBadge.textContent = `${confidence}%`;
            
            statusText.textContent = 'Response Generated (Gemini AI)';
            statusText.style.color = '#00ffcc';
            ttsBtn.disabled = false;
            
        } catch (error) {
            console.error('Voice chat error:', error);
            chatbotResponseEl.textContent = `Error: ${error.message}`;
            chatbotResponseEl.classList.remove('placeholder');
            statusText.textContent = 'Processing Error';
            statusText.style.color = '#ff4d4d';
        }
    }

    // Text Input Function (Fallback)
    async function sendMessage(text) {
        if (!text || text.trim() === '') return;
        
        statusText.textContent = 'Processing...';
        statusText.style.color = '#e0e0e0';
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
            
            const confidence = data.confidence !== undefined ? (data.confidence * 100).toFixed(1) : 0;
            confidenceBadge.textContent = `${confidence}%`;
            
            statusText.textContent = 'Response Generated (Gemini AI)';
            statusText.style.color = '#00ffcc';
            ttsBtn.disabled = false;
            
        } catch (error) {
            console.error('Chat error:', error);
            chatbotResponseEl.textContent = `Error: ${error.message}. Is the Flask backend running?`;
            chatbotResponseEl.classList.remove('placeholder');
            statusText.textContent = 'Error occurred';
            statusText.style.color = '#ff4d4d';
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
