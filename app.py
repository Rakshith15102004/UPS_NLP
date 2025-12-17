from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import whisper
import os

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

# Load Whisper model once at startup
print("Loading Whisper model... Please wait 5–10 seconds.")
model = whisper.load_model("tiny")  # Best for Indian languages

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
    # Save uploaded audio
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Whisper transcription
    print("Transcribing audio...")
    output = model.transcribe(file_path, fp16=False)
    text = output["text"]

    # Process text
    result = medical_pipeline(text)

    # Cleanup
    os.remove(file_path)

    return {
        "input_type": "audio",
        "transcribed_text": text,
        "analysis": result
    }

@app.get("/")
def home():
    return {"message": "Medical Analyzer API Running! Visit /docs for Swagger UI."}

