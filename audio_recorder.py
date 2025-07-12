import streamlit as st
import queue
import io
import wave
import pyaudio
from datetime import datetime

# Configuration
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS_CHUNK = 2  # Send audio chunks every 2 seconds

class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
    def start_recording(self):
        """Start recording audio from microphone"""
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
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
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
        self.audio.terminate()