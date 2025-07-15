import streamlit as st
import json
import time
from datetime import datetime
from audio_recorder import AudioRecorder
from transcription import TranscriptionService
from llm_assistant import LLMAssistant

# Configuration
LLM_UPDATE_INTERVAL = 3   # Update LLM suggestions every 3 seconds

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
        st.success("��️ Recording started!")
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