from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
from faster_whisper import WhisperModel 

# Import your medical NLP pipeline
from Input_handling import medical_pipeline

app = FastAPI(
    title="Medical Symptom Analyzer API",
    description="Accepts text or voice, converts voice to text, then analyzes symptoms.",
    version="1.0"
)

# Allow all origins (React, mobile apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------
# NEW: Load Model Efficiently (Uses ~200MB RAM)
# --------------------------------------------------------
print("Loading Faster-Whisper model...")
# 'cpu' + 'int8' is the secret combo for Free Tier
model = WhisperModel("tiny", device="cpu", compute_type="int8") 

# --------------------------------------------------------
# TEXT INPUT API
# --------------------------------------------------------
@app.post("/analyze-text/")
async def analyze_text(text: str = Form(...)):
    result = medical_pipeline(text)
    return {
        "input_type": "text",
        "text_received": text,
        "analysis": result
    }

# --------------------------------------------------------
# AUDIO INPUT API
# --------------------------------------------------------
@app.post("/analyze-audio/")
async def analyze_audio(file: UploadFile = File(...)):
    # 1. Save uploaded audio temporarily
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 2. Transcribe using Faster-Whisper
    print("Transcribing audio...")
    segments, info = model.transcribe(file_path, beam_size=5)
    
    # Join the segments into one string
    text = " ".join([segment.text for segment in segments])

    # 3. Analyze symptoms
    result = medical_pipeline(text)

    # 4. Cleanup temp file
    if os.path.exists(file_path):
        os.remove(file_path)

    return {
        "input_type": "audio",
        "transcribed_text": text,
        "analysis": result
    }

@app.get("/")
def home():
    return {"message": "Medical Analyzer API Running! Visit /docs for Swagger UI."}
