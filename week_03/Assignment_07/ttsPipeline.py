from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from huggingface_hub import snapshot_download
# ────────────────────────────────
# 1️⃣ Load model and tokenizer
# ────────────────────────────────
model = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
print("✅ Model and tokenizer loaded")
# ────────────────────────────────
# 2️⃣ Input text
# ────────────────────────────────
text = "Xin chào anh em đến với bài tập của khoá AI Application Engineer"

# ────────────────────────────────
# 3️⃣ Generate waveform
# ────────────────────────────────
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    output = model(**inputs).waveform
print("✅ Inputs tokenized")
# ────────────────────────────────
# 4️⃣ Save to WAV
# ────────────────────────────────
waveform = output.squeeze().cpu().numpy().astype(np.float32)
sf.write("output.wav", waveform, model.config.sampling_rate)
print("✅ Waveform generated")
# ────────────────────────────────
# 5️⃣ Convert to MP3 - Optional
# ────────────────────────────────
sound = AudioSegment.from_wav("output.wav")
sound.export("output.mp3", format="mp3")

print("✅ Success! MP3 file saved as: output.mp3")