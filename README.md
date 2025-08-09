# 🎙 Real-Time Speech-to-Text & Response AI

A **low-latency voice AI assistant** that listens to your voice, transcribes it in real-time, generates a response using a lightweight LLM, and speaks it back to you — all within seconds.

This project is designed to **mimic real-time reasoning-oriented voice AI systems** with optimizations for speed and scalability.

---

## ✨ Features
- 🎤 **Real-Time Speech-to-Text (STT)** using Google Speech Recognition API (can swap with Wav2Vec2).
- 🤖 **Text Generation** using DistilGPT-2 from HuggingFace Transformers.
- 🔊 **Text-to-Speech (TTS)** using Google TTS (gTTS).
- ⚡ Optimized for **low latency** with minimal dependencies.
- 🛠 Modular design to easily swap STT, LLM, and TTS engines.

---

## 📦 Installation
```bash
git clone https://github.com/Gokul12120/Real-Time-Voice-ai.git
cd real_time_voice_ai
pip install -r requirements.txt
```

---

## 🚀 Usage

### **Microphone Mode**
```bash
python app.py --mode mic
```

### **Audio File Mode**
```bash
python app.py --mode file --file static/sample.wav
```

---

## 🎥 Demo
Here’s the real-time workflow in action (replace with your GIF/video):

![Demo GIF](demo.gif)

*(To record: use OBS or ScreenToGif and capture terminal + audio playback)*

---

## 🛠 Tech Stack
- **Python 3.10+**
- HuggingFace Transformers (DistilGPT-2)
- SpeechRecognition + gTTS
- PyTorch
- SoundDevice

---

## 📈 Optimizations Implemented
- Modular architecture for easy swapping of components.
- Caching of models for faster repeated loads.
- Adjustable parameters for generation speed.

---

## 🚀 Future Improvements
- Integrate **Wav2Vec2** or **Whisper** for higher STT accuracy.
- Implement **streaming inference** for sub-second responses.
- Quantize models using `bitsandbytes` for better performance on CPUs.
- Deploy as a **FastAPI** microservice with WebSocket streaming.

---

## 📜 License
MIT License — free to use, modify, and distribute.

---

💡 **Pro Tip:** This project is designed to showcase **hands-on ML deployment skills** — perfect for research, prototyping, or demonstrating low-latency AI systems.
