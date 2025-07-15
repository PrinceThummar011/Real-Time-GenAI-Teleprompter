import streamlit as st
import io
import os
import openai
from groq import Groq
from typing import Optional
from datetime import datetime

class TranscriptionService:
    def __init__(self, service_type="groq"):
        self.service_type = service_type
        self.openai_client = None
        self.groq_client = None
        
        if service_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
        elif service_type == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.groq_client = Groq(api_key=api_key)
    
    def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio data to text"""
        try:
            if self.service_type == "groq" and self.groq_client:
                # Create a temporary file-like object
                audio_file = io.BytesIO(audio_data)
                audio_file.name = "audio.wav"
                
                # Groq uses Whisper models for transcription
                transcription = self.groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3",  # Groq's Whisper model
                    response_format="text"
                )
                return transcription.strip()
            elif self.service_type == "openai" and self.openai_client:
                # Keep OpenAI as fallback option
                audio_file = io.BytesIO(audio_data)
                audio_file.name = "audio.wav"
                
                response = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                return response.strip()
            else:
                # Fallback: Mock transcription for demo
                return f"[Mock transcription at {datetime.now().strftime('%H:%M:%S')}]"
        except Exception as e:
            st.error(f"Transcription error: {e}")
            return None