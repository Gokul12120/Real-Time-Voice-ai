#!/usr/bin/env python3
# Real-Time Speech-to-Text & Response AI (Prototype)
# Supports: microphone input (streaming chunks) or audio file input.
# Pipeline: Audio -> STT (wav2vec2 or Google SR) -> LLM response (distilgpt2) -> TTS
#
# Usage:
#    python app.py --mode mic
#    python app.py --mode file --file_path ./static/sample.wav

import argparse
import os
import queue
import time
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

from gtts import gTTS

# Transformers imports kept guarded because user may opt to run without them
try:
    from transformers import pipeline, AutoProcessor, AutoModelForCTC, AutoModelForCausalLM
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

CHUNK_SECONDS = 2.0  # audio chunk length for streaming (seconds)
SAMPLE_RATE = 16000


def _play_audio(filepath):
    """Play audio cross-platform using simple commands (user can replace with better player)."""
    if os.name == "nt":
        os.system(f"start {filepath}")
    else:
        # try mpg123 (common on Linux/Mac brew)
        os.system(f"mpg123 {filepath} >/dev/null 2>&1 || afplay {filepath} >/dev/null 2>&1")


class MicStreamer:
    def __init__(self, device=None, rate=SAMPLE_RATE, chunk_seconds=CHUNK_SECONDS):
        if sd is None:
            raise RuntimeError("sounddevice is not available. Please install dependencies.")
        self.rate = rate
        self.chunk_samples = int(chunk_seconds * rate)
        self.q = queue.Queue()
        self.device = device

    def _callback(self, indata, frames, time_info, status):
        # Called in a separate thread by sounddevice
        self.q.put(indata.copy())

    def generator(self):
        with sd.InputStream(samplerate=self.rate, channels=1, callback=self._callback, device=self.device):
            buffer = np.zeros((0, 1), dtype=np.float32)
            print("Microphone stream started. Speak into the mic...")
            while True:
                chunk = self.q.get()
                buffer = np.concatenate([buffer, chunk], axis=0)
                if buffer.shape[0] >= self.chunk_samples:
                    to_yield = buffer[: self.chunk_samples]
                    buffer = buffer[self.chunk_samples :]
                    yield (to_yield.flatten(), self.rate)


def transcribe_chunk_wav2vec(chunk_audio, rate, model_name="facebook/wav2vec2-base-960h"):
    """Transcribe raw audio chunk using HuggingFace Wav2Vec2 (CPU/GPU depending on environment)."""
    if not HF_AVAILABLE:
        raise RuntimeError("HuggingFace transformers not available. Install transformers and torch.")
    from transformers import AutoProcessor, AutoModelForCTC
    import torch
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name)
    input_values = processor(chunk_audio, sampling_rate=rate, return_tensors="pt", padding=True).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    pred_ids = logits.argmax(dim=-1)
    transcription = processor.decode(pred_ids[0])
    return transcription.lower()


def transcribe_with_google_sr(chunk_audio, rate):
    """Fallback using SpeechRecognition's Google Web Speech API (requires internet)."""
    if sr is None:
        raise RuntimeError("speech_recognition package not available.")
    recognizer = sr.Recognizer()
    # Save chunk to temp WAV
    from scipy.io.wavfile import write
    tmp_path = "tmp_chunk.wav"
    write(tmp_path, rate, (chunk_audio * 32767).astype("int16"))
    with sr.AudioFile(tmp_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
    except Exception as e:
        text = ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return text.lower()


class SimpleLLM:
    def __init__(self, model_name="distilgpt2", quantize=False):
        self.model_name = model_name
        self.quantize = quantize
        self._load()

    def _load(self):
        if not HF_AVAILABLE:
            print("Transformers not available. LLM operations will use a fallback rule-based responder.")
            self.generator = None
            return
        try:
            # Example: load a small causal LM pipeline. For production, replace with optimized runtime.
            if self.quantize:
                # Bitsandbytes quantization would be applied here if available
                print("Quantization requested - ensure bitsandbytes + accelerate are installed and configured.")
            self.generator = pipeline("text-generation", model=self.model_name, max_length=64, device_map="auto")
        except Exception as e:
            print("Error loading LLM pipeline:", e)
            self.generator = None

    def generate(self, prompt):
        if self.generator is None:
            # simple fallback
            return "I heard you. (fallback response — enable transformers to get LLM output)"
        out = self.generator(prompt, max_length=64, do_sample=True, num_return_sequences=1)
        return out[0]["generated_text"]


def tts_and_play(text, filename="response.mp3", lang="en"):
    tts = gTTS(text)
    tts.save(filename)
    _play_audio(filename)


def run_streaming_pipeline(use_wav2vec=False, use_quant=False):
    mic = MicStreamer()
    llm = SimpleLLM(quantize=use_quant)
    for chunk, rate in mic.generator():
        # Convert numpy float32 [-1,1] to required format
        audio = chunk.astype(np.float32)
        print("Chunk received — running STT...")
        # Try wav2vec if requested and available
        transcription = ""
        if use_wav2vec and HF_AVAILABLE:
            try:
                transcription = transcribe_chunk_wav2vec(audio, rate)
            except Exception as e:
                print("wav2vec error:", e)
        if not transcription:
            # fallback to Google SR
            transcription = transcribe_with_google_sr(audio, rate)
        if transcription.strip() == "":
            print("[no speech recognized]")
            continue
        print("You:", transcription)
        response = llm.generate(transcription)
        print("AI:", response)
        tts_and_play(response)


def run_file_mode(file_path, use_wav2vec=False, use_quant=False):
    # Load audio file, chunk it, then run same pipeline
    import soundfile as sf
    data, rate = sf.read(file_path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    # chunk
    samples_per_chunk = int(CHUNK_SECONDS * rate)
    llm = SimpleLLM(quantize=use_quant)
    for start in range(0, len(data), samples_per_chunk):
        chunk = data[start : start + samples_per_chunk]
        if len(chunk) < 100:  # ignore tiny chunks
            continue
        print("Chunk -> STT...")
        transcription = ""
        if use_wav2vec and HF_AVAILABLE:
            try:
                transcription = transcribe_chunk_wav2vec(chunk, rate)
            except Exception as e:
                print("wav2vec error:", e)
        if not transcription:
            transcription = transcribe_with_google_sr(chunk, rate)
        if transcription.strip() == "":
            continue
        print("You:", transcription)
        response = llm.generate(transcription)
        print("AI:", response)
        tts_and_play(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mic", "file"], default="mic")
    parser.add_argument("--file_path", type=str, default=None)
    parser.add_argument("--wav2vec", action="store_true", help="Use wav2vec2 HF model for STT (requires transformers).")
    parser.add_argument("--quant", action="store_true", help="Request quantized LLM (requires bitsandbytes + accelerate).")
    args = parser.parse_args()

    if args.mode == "mic":
        run_streaming_pipeline(use_wav2vec=args.wav2vec, use_quant=args.quant)
    else:
        if not args.file_path:
            raise ValueError("file mode requires --file_path")
        run_file_mode(args.file_path, use_wav2vec=args.wav2vec, use_quant=args.quant)


if __name__ == "__main__":
    main()
