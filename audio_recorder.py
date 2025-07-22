import streamlit as st
import io
import wave
from datetime import datetime
from audio_recorder_streamlit import audio_recorder

# Try to import pyaudio for local development
try:
    import pyaudio
    import queue
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    st.info("Running in cloud mode - using file upload instead of live recording")

# Configuration
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
CHANNELS = 1
RATE = 16000
RECORD_SECONDS_CHUNK = 2

class AudioRecorder:
    def __init__(self):
        self.is_cloud_mode = not PYAUDIO_AVAILABLE
        if not self.is_cloud_mode:
            self.audio = pyaudio.PyAudio()
            self.stream = None
            self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recorded_audio = None
        
    def start_recording(self):
        """Start recording audio - cloud or local mode"""
        if self.is_cloud_mode:
            return self._start_cloud_recording()
        else:
            return self._start_local_recording()
    
    def _start_cloud_recording(self):
        """Cloud mode: Use streamlit audio recorder"""
        st.info("🎤 Click the record button below to start recording")
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
        )
        
        if audio_bytes:
            self.recorded_audio = audio_bytes
            self.is_recording = True
            return True
        return False
    
    def _start_local_recording(self):
        """Local mode: Use PyAudio"""
        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._audio_callback
            )
            self.is_recording = True
            self.stream.start_stream()
            return True
        except Exception as e:
            st.error(f"Error starting audio recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        if not self.is_cloud_mode and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream (local mode only)"""
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def get_audio_chunk(self, duration_seconds=RECORD_SECONDS_CHUNK):
        """Get audio chunk - cloud or local mode"""
        if self.is_cloud_mode:
            return self._get_cloud_audio_chunk()
        else:
            return self._get_local_audio_chunk(duration_seconds)
    
    def _get_cloud_audio_chunk(self):
        """Get audio chunk in cloud mode"""
        if self.recorded_audio:
            # Return the recorded audio data
            audio_data = self.recorded_audio
            self.recorded_audio = None  # Clear after use
            return audio_data
        return None
    
    def _get_local_audio_chunk(self, duration_seconds):
        """Get audio chunk in local mode"""
        frames = []
        frames_needed = int(RATE / CHUNK_SIZE * duration_seconds)
        
        for _ in range(frames_needed):
            try:
                frame = self.audio_queue.get(timeout=0.1)
                frames.append(frame)
            except queue.Empty:
                break
        
        if frames:
            # Convert to WAV format
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(self.audio.get_sample_size(FORMAT))
                wav_file.setframerate(RATE)
                wav_file.writeframes(b''.join(frames))
            
            wav_buffer.seek(0)
            return wav_buffer.getvalue()
        return None
    
    def cleanup(self):
        """Cleanup audio resources"""
        self.stop_recording()
        if not self.is_cloud_mode:
            self.audio.terminate()