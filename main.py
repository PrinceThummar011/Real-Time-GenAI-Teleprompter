import streamlit as st
import asyncio
import json
import time
from datetime import datetime, timedelta
import threading
import queue
import io
import wave
import sounddevice as sd
import soundfile as sf
import openai  
from groq import Groq 
from typing import List, Dict, Optional
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# Configuration
CHUNK_SIZE = 1024
FORMAT = 'int16'
CHANNELS = 1
RATE = 16000
RECORD_SECONDS_CHUNK = 2  # Send audio chunks every 2 seconds
LLM_UPDATE_INTERVAL = 3   # Update LLM suggestions every 3 seconds

class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.frames = []

    def start_recording(self):
        """Start recording audio from microphone"""
        try:
            self.is_recording = True
            threading.Thread(target=self._record_loop, daemon=True).start()
            return True
        except Exception as e:
            st.error(f"Error starting audio recording: {e}")
            return False

    def _record_loop(self):
        """Internal loop to record audio"""
        try:
            with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype=FORMAT, blocksize=CHUNK_SIZE) as stream:
                while self.is_recording:
                    data, _ = stream.read(CHUNK_SIZE)
                    self.audio_queue.put(data.tobytes())
        except Exception as e:
            st.error(f"Audio recording error: {e}")

    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False

    def get_audio_chunk(self, duration_seconds=RECORD_SECONDS_CHUNK):
        """Get audio chunk of specified duration"""
        frames = []
        frames_needed = int(RATE / CHUNK_SIZE * duration_seconds)

        for _ in range(frames_needed):
            try:
                frame = self.audio_queue.get(timeout=0.1)
                frames.append(frame)
            except queue.Empty:
                break

        if frames:
            wav_buffer = io.BytesIO()
            with sf.SoundFile(wav_buffer, mode='x', samplerate=RATE, channels=CHANNELS, format='WAV', subtype='PCM_16') as file:
                for frame in frames:
                    file.buffer_write(frame, dtype=FORMAT)
            wav_buffer.seek(0)
            return wav_buffer.getvalue()
        return None

    def cleanup(self):
        self.stop_recording()

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

class LLMAssistant:
    def __init__(self, model_type="groq"):
        self.model_type = model_type
        self.openai_client = None
        self.groq_client = None
        self.conversation_context = []
        
        if model_type == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.groq_client = Groq(api_key=api_key)
        elif model_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
        
        self.system_prompt = """You are an AI sales assistant helping a sales representative during a live call. 
        
        Your role is to provide SHORT, actionable suggestions based on the conversation transcript. 
        
        Guidelines:
        - Keep suggestions to 1-2 sentences maximum
        - Focus on sales techniques, objection handling, and relationship building
        - Provide specific, actionable advice
        - Use these categories:
          💡 Tip - General sales advice
          ⚠️ Reminder - Important things not to forget
          ❗ Alert - Urgent actions or red flags
          🎯 Close - Closing opportunities
        
        Only respond with the suggestion, starting with the appropriate emoji category.
        If no specific advice is needed, respond with "No suggestions at this time."
        """
    
    def get_suggestions(self, transcript_chunk: str) -> List[str]:
        """Get AI suggestions based on transcript"""
        try:
            if not transcript_chunk.strip():
                return []
                
            if self.model_type == "groq" and self.groq_client:
                response = self.groq_client.chat.completions.create(
                    model="llama3-8b-8192",  # Groq's fast Llama model
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Recent conversation: {transcript_chunk}"}
                    ],
                    max_tokens=100,
                    temperature=0.7
                )
                
                suggestion = response.choices[0].message.content.strip()
                if suggestion and suggestion != "No suggestions at this time.":
                    return [suggestion]
                return []
            elif self.model_type == "openai" and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Recent conversation: {transcript_chunk}"}
                    ],
                    max_tokens=100,
                    temperature=0.7
                )
                
                suggestion = response.choices[0].message.content.strip()
                if suggestion and suggestion != "No suggestions at this time.":
                    return [suggestion]
                return []
            else:
                # Mock suggestions for demo
                mock_suggestions = [
                    "💡 Tip: Ask about their current challenges",
                    "⚠️ Reminder: Mention the ROI benefits",
                    "❗ Alert: Customer mentioned budget concerns",
                    "🎯 Close: Good time to ask for next steps"
                ]
                import random
                return [random.choice(mock_suggestions)]
                
        except Exception as e:
            st.error(f"LLM error: {e}")
            return []

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'transcript' not in st.session_state:
        st.session_state.transcript = []
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = []
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = None
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = None
    if 'transcription_service' not in st.session_state:
        st.session_state.transcription_service = TranscriptionService("groq")  # Use Groq by default
    if 'llm_assistant' not in st.session_state:
        st.session_state.llm_assistant = LLMAssistant("groq")  # Use Groq by default
    if 'last_llm_update' not in st.session_state:
        st.session_state.last_llm_update = 0

def start_session():
    """Start recording session"""
    st.session_state.audio_recorder = AudioRecorder()
    if st.session_state.audio_recorder.start_recording():
        st.session_state.is_recording = True
        st.session_state.session_start_time = datetime.now()
        st.session_state.transcript = []
        st.session_state.suggestions = []
        st.success("🎙️ Recording started!")
    else:
        st.error("Failed to start recording")

def stop_session():
    """Stop recording session"""
    if st.session_state.audio_recorder:
        st.session_state.audio_recorder.stop_recording()
        st.session_state.audio_recorder.cleanup()
    st.session_state.is_recording = False
    st.session_state.audio_recorder = None
    st.success("🛑 Recording stopped!")

def process_audio_chunk():
    """Process audio chunk for transcription and suggestions"""
    if not st.session_state.is_recording or not st.session_state.audio_recorder:
        return
    
    # Get audio chunk
    audio_data = st.session_state.audio_recorder.get_audio_chunk()
    if not audio_data:
        return
    
    # Transcribe audio
    transcript_text = st.session_state.transcription_service.transcribe_audio(audio_data)
    if transcript_text and transcript_text.strip():
        timestamp = datetime.now().strftime("%H:%M:%S")
        transcript_entry = {
            "timestamp": timestamp,
            "text": transcript_text
        }
        st.session_state.transcript.append(transcript_entry)
    
    # Update LLM suggestions periodically
    current_time = time.time()
    if current_time - st.session_state.last_llm_update > LLM_UPDATE_INTERVAL:
        if st.session_state.transcript:
            # Get recent transcript for context
            recent_transcript = " ".join([entry["text"] for entry in st.session_state.transcript[-5:]])
            suggestions = st.session_state.llm_assistant.get_suggestions(recent_transcript)
            
            for suggestion in suggestions:
                suggestion_entry = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "text": suggestion
                }
                st.session_state.suggestions.append(suggestion_entry)
            
            # Keep only recent suggestions
            st.session_state.suggestions = st.session_state.suggestions[-10:]
        
        st.session_state.last_llm_update = current_time

def export_session_data():
    """Export session data as JSON"""
    if not st.session_state.transcript and not st.session_state.suggestions:
        st.warning("No data to export")
        return
    
    session_data = {
        "session_info": {
            "start_time": st.session_state.session_start_time.isoformat() if st.session_state.session_start_time else None,
            "export_time": datetime.now().isoformat()
        },
        "transcript": st.session_state.transcript,
        "suggestions": st.session_state.suggestions
    }
    
    json_data = json.dumps(session_data, indent=2)
    
    st.download_button(
        label="📥 Download Session Data",
        data=json_data,
        file_name=f"sales_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def main():
    st.set_page_config(
        page_title="GenAI Sales Teleprompter",
        page_icon="🎙️",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Header
    st.title("🎙️ Real-Time GenAI Sales Teleprompter (Powered by Groq)")
    st.markdown("*Your AI wingman for sales calls - Lightning fast with Groq!*")
    
    # API Configuration Section
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Service selection
        transcription_service = st.selectbox(
            "Transcription Service",
            ["groq", "openai"],
            index=0,
            help="Choose your transcription service"
        )
        
        llm_service = st.selectbox(
            "LLM Service",
            ["groq", "openai"],
            index=0,
            help="Choose your LLM service for suggestions"
        )
        
        if st.button("🔄 Update Services"):
            st.session_state.transcription_service = TranscriptionService(transcription_service)
            st.session_state.llm_assistant = LLMAssistant(llm_service)
            st.success("Services updated!")
        
        st.markdown("---")
        st.markdown("**API Keys Required:**")
        st.markdown("- `GROQ_API_KEY` (Primary)")
        st.markdown("- `OPENAI_API_KEY` (Optional)")
    
    # Controls
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    
    with col1:
        if not st.session_state.is_recording:
            if st.button("🎙️ Start Session", type="primary"):
                start_session()
        else:
            if st.button("🛑 Stop Session", type="secondary"):
                stop_session()
    
    with col2:
        mic_status = "🟢 ON" if st.session_state.is_recording else "🔴 OFF"
        st.markdown(f"**Mic Status:** {mic_status}")
    
    with col3:
        if st.session_state.session_start_time:
            elapsed = datetime.now() - st.session_state.session_start_time
            st.markdown(f"**Session Time:** {str(elapsed).split('.')[0]}")
        else:
            st.markdown("**Session Time:** --:--:--")
    
    with col4:
        if st.session_state.transcript or st.session_state.suggestions:
            export_session_data()
    
    # Process audio if recording
    if st.session_state.is_recording:
        process_audio_chunk()
    
    # Main content area
    col_transcript, col_suggestions = st.columns([3, 2])
    
    with col_transcript:
        st.subheader("📝 Live Transcript")
        
        transcript_container = st.container()
        with transcript_container:
            if st.session_state.transcript:
                # Show recent transcript entries
                for entry in st.session_state.transcript[-20:]:  # Show last 20 entries
                    st.markdown(f"**{entry['timestamp']}** - {entry['text']}")
            else:
                st.markdown("*Transcript will appear here when recording starts...*")
    
    with col_suggestions:
        st.subheader("💡 AI Suggestions")
        
        suggestions_container = st.container()
        with suggestions_container:
            if st.session_state.suggestions:
                # Show recent suggestions
                for suggestion in st.session_state.suggestions[-5:]:  # Show last 5 suggestions
                    st.markdown(f"**{suggestion['timestamp']}**")
                    st.markdown(f"{suggestion['text']}")
                    st.markdown("---")
            else:
                st.markdown("*AI suggestions will appear here during the call...*")
    
    # Instructions
    with st.expander("ℹ️ Setup Instructions"):
        st.markdown("""
        ### Requirements:
        1. **Groq API Key**: Set your `GROQ_API_KEY` environment variable
           - Get your free API key from: https://console.groq.com/
        2. **OpenAI API Key** (Optional): Set `OPENAI_API_KEY` for fallback
        3. **Microphone Access**: Allow browser to access your microphone
        4. **Python Dependencies**: Install required packages:
           ```bash
           pip install streamlit groq openai pyaudio python-dotenv
           ```
        
        ### Environment Variables (.env file):
        ```
        GROQ_API_KEY=your_groq_api_key_here
        OPENAI_API_KEY=your_openai_api_key_here  # Optional
        ```
        
        ### How to Use:
        1. Set up your API keys in the .env file
        2. Click "Start Session" to begin recording
        3. Speak naturally - the app will transcribe in real-time using Groq's Whisper
        4. AI suggestions will appear based on the conversation using Groq's Llama models
        5. Click "Stop Session" when done
        6. Export your session data if needed
        
        ### Groq Models Used:
        - **Transcription**: whisper-large-v3 (Ultra-fast speech-to-text)
        - **LLM Suggestions**: llama3-8b-8192 (Lightning-fast chat completions)
        
        ### Benefits of Groq:
        - ⚡ Ultra-fast inference speeds
        - 💰 Cost-effective API pricing
        - 🎯 High-quality model outputs
        - 🚀 Optimized for real-time applications
        """)
    
    # Auto-refresh for real-time updates
    if st.session_state.is_recording:
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()