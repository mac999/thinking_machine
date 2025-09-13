# title: Korean Speech-to-Text and Text-to-Speech with faster-whisper and Kokoro
# author: taewook kang (laputa99999@gmail.com)
# date: 2025-08-16
# install: pip install faster-whisper kokoro-onnx sounddevice soundfile
#          Requires kokoro-v1.0.int8.onnx and voices-v1.0.bin in the same folder
#          Uses faster-whisper for STT and Kokoro for TTS
import os, ssl, time, numpy as np, sounddevice as sd, soundfile as sf
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
os.environ["CURL_CA_BUNDLE"] = ""

ssl._create_default_https_context = ssl._create_unverified_context

# Settings 
REC_SECONDS = 10               # 10 second recording
SR = 16000                     # Suitable SR for STT
INPUT_WAV = "input.wav"
VOICE = "af_heart"             # Default voice (see voices list). Can synthesize Korean.
KOKORO_ONNX = "kokoro-v1.0.int8.onnx"
KOKORO_VOICES = "voices-v1.0.bin"

# 1) Record 10 seconds from microphone and save WAV 
print("Recording for 10 seconds. Please speak. (recording started)")
sd.default.samplerate = SR
sd.default.channels = 1
audio = sd.rec(int(REC_SECONDS * SR), dtype="float32")
sd.wait()
sf.write(INPUT_WAV, audio, SR)
print(f"Saved recording: {INPUT_WAV}")

# 2) STT: Convert Korean speech to text with faster-whisper 
#    small model balances speed/accuracy; good starting point
#    runs fine on CPU with int8 inference
model = WhisperModel("small", device="cpu", compute_type="int8", local_files_only=False)
segments, info = model.transcribe(
    INPUT_WAV,
    vad_filter=True,   # Remove silence segments
    language="ko"      # Force Korean (auto-detect if omitted)
)

user_text = "".join(seg.text for seg in segments).strip()
print("Transcribed text:")
print(user_text if user_text else "(empty text)")

# 3) TTS: Synthesize Korean text to speech with Kokoro and play 
kokoro = Kokoro(KOKORO_ONNX, KOKORO_VOICES)
samples, sr_out = kokoro.create(user_text or "No recorded speech.",
                                voice=VOICE, speed=1.0, lang="ko") # # lang="ko" for Korean synthesis; adjust speed for rate

# Save file and play
OUT_WAV = "output_ko.wav"
sf.write(OUT_WAV, samples, sr_out)
print(f"Saved synthesized audio: {OUT_WAV}")
sd.play(samples, sr_out)
sd.wait()
print("Done")