Here’s a polished `README.md` file for your **Real-Time GenAI Teleprompter** project:

---

# 🎤 Real-Time GenAI Teleprompter

A real-time AI assistant for sales conversations.

This lightweight Streamlit app listens to your live voice input via microphone, transcribes the conversation in real time using a speech-to-text (STT) engine, and sends chunks of text to an LLM (like GPT-4o or LLaMA 3) to generate helpful, proactive sales prompts — just like an AI wingman.

---

## 🧠 Concept

> Think of it as a **real-time whisper assistant**.

* 🔊 **Live Audio Input** via your microphone
* ✍️ **Real-Time Transcription** using Whisper, Deepgram, or any STT
* 💡 **AI Suggestions** every few seconds from an LLM (GPT-4o / LLaMA)
* 📄 **Session Logs** with export option for transcript and AI tips

---

## 🔧 Features

### 1. 🎙️ Audio Input

* Microphone access via Streamlit widget or browser
* Visual indicator: mic **ON/OFF**
* **Start** / **Stop** session buttons

### 2. 📝 Real-Time Transcription

* Continuous transcription as you speak
* Scrollable, live-updating transcript
* Timestamps included
* No speaker separation — all audio is captured

### 3. 🤖 LLM Integration

* Every \~2–3 seconds, the transcript is chunked and sent to an LLM
* Short tips/suggestions are returned in real time
* Example system prompt:

  > *"You are an assistant helping a sales agent in a live call..."*

#### Supported Models:

* **GPT-4o** via OpenAI (preferred)
* **LLaMA 3** via Ollama, Together AI, or your choice

### 4. 🪄 Suggestions Panel

* Live display of LLM-generated tips
* Shows top 1–2 suggestions at a time
* Tagged as:

  * 💡 Tip
  * ⚠️ Reminder
  * ❗ Alert

### 5. 📦 Session Handling

* Start/Stop session control
* Session timer display
* On session end:

  * Option to export full transcript + AI suggestions
  * Export format: `.json` or `.txt`

---

## 🚀 Tech Stack

| Component      | Tool/Library                                      |
| -------------- | ------------------------------------------------- |
| **Frontend**   | Streamlit                                         |
| **Backend**    | Python (no server required, but optional FastAPI) |
| **STT Engine** | Deepgram (Streaming) or Whisper                   |
| **LLM**        | GPT-4o (OpenAI) / LLaMA 3 (Ollama / Together AI)  |
| **Export**     | Built-in file download (JSON/TXT)                 |

---

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/real-time-genai-teleprompter.git
cd real-time-genai-teleprompter
```

### 2. Install Dependencies

```bash
pip install streamlit openai groq pyaudio python-dotenv
```

(Replace `groq` with your STT/LLM provider library as needed.)

### 3. Set Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here  # Optional
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 📤 Output Example

At the end of a session, you can export:

* Full **Transcript**
* All **AI Suggestions**
* Format: `.json` or `.txt`

---

## ⚠️ Notes

* No login/admin required — just run locally and go
* Ideal for prototyping real-time voice assistants or sales agents
* Can be extended with advanced STT models or multi-user support

---


