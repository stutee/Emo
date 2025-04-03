import streamlit as st
import os
import sounddevice as sd
import numpy as np
import wave
import tempfile
import openai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def record_audio(duration=5, sample_rate=44100):
    """Record audio from microphone"""
    recording = sd.rec(int(duration * sample_rate), 
                      samplerate=sample_rate, 
                      channels=1, 
                      dtype=np.float32)
    sd.wait()
    return recording

def save_audio(recording, sample_rate=44100):
    """Save recorded audio to a temporary WAV file"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((recording * 32767).astype(np.int16).tobytes())
        return temp_file.name

def transcribe_audio(audio_file):
    """Transcribe audio using OpenAI Whisper"""
    with open(audio_file, 'rb') as file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=file
        )
    return transcript.text

def get_ai_response(text):
    """Get AI response using GPT"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant for journaling. Provide thoughtful, empathetic responses."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

def text_to_speech(text):
    """Convert text to speech using OpenAI TTS"""
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    return response

def main():
    st.title("AI Voice Assistant for Journaling")
    st.write("Record your thoughts and get AI responses!")

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Record button
    if st.button("Record Your Thoughts"):
        with st.spinner("Recording... Speak now!"):
            # Record audio
            recording = record_audio(duration=5)
            
            # Save audio to temporary file
            audio_file = save_audio(recording)
            
            # Transcribe audio
            transcribed_text = transcribe_audio(audio_file)
            
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": transcribed_text})
            with st.chat_message("user"):
                st.write(transcribed_text)
            
            # Get AI response
            ai_response = get_ai_response(transcribed_text)
            
            # Add AI response to chat
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.write(ai_response)
            
            # Convert AI response to speech
            speech = text_to_speech(ai_response)
            
            # Save speech to temporary file and play it
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(speech.content)
                st.audio(temp_file.name)

if __name__ == "__main__":
    main() 