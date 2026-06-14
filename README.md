AI Voice Assistant for Journaling

# AI Voice Assistant for Journaling

An AI-powered voice journaling application built with Streamlit and OpenAI APIs. Users can record their thoughts through a microphone, automatically transcribe speech to text, receive thoughtful AI responses, and listen to responses through text-to-speech.

---

## Features

* 🎤 Record voice notes directly from your microphone
* 📝 Automatic speech-to-text transcription using OpenAI Whisper
* 🤖 AI-powered journaling conversations using GPT
* 🔊 Text-to-speech responses using OpenAI TTS
* 💬 Persistent chat history during the session
* 🌐 Simple and intuitive Streamlit web interface

---

## Tech Stack

* Python
* Streamlit
* OpenAI API
* Whisper Speech-to-Text
* OpenAI Text-to-Speech
* NumPy
* SoundDevice

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-voice-journaling.git
cd ai-voice-journaling
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run voice_assistant.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## How It Works

1. Click **Record Your Thoughts**
2. Speak into your microphone for 5 seconds
3. Audio is recorded and saved as a temporary WAV file
4. OpenAI Whisper transcribes the speech into text
5. GPT generates a thoughtful journaling response
6. OpenAI TTS converts the response into audio
7. The response is displayed and played back to the user

---

## Project Structure

```text
.
├── voice_assistant.py
├── requirements.txt
├── .env
└── README.md
```

---

## Requirements

```text
streamlit==1.32.0
openai==1.12.0
python-dotenv==1.0.1
sounddevice==0.4.6
numpy==1.26.4
```

---

## Future Improvements

* Real-time voice conversations
* Adjustable recording duration
* Journal entry storage and search
* User authentication
* Mood tracking and analytics
* Multiple voice options
* Conversation memory across sessions

---

## Disclaimer

This application is intended for personal journaling and reflection purposes. It should not be considered a substitute for professional mental health or medical advice.
