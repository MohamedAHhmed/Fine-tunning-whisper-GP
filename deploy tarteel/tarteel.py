from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
import soundfile as sf
import io
import os
import numpy as np
from dotenv import load_dotenv
#uvicorn tarteel:app --host 0.0.0.0 --port 8000 --reload
# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Allow frontend requests (adjust allowed origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to your frontend's URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get Hugging Face token and model name from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Read the uploaded audio file
        audio_bytes = await file.read()
        
        # Load audio using soundfile
        audio_file = io.BytesIO(audio_bytes)
        audio, original_sr = sf.read(audio_file)
        
        # If audio has multiple channels, convert it to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Validate that audio is not empty
        if audio.size == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
        
        # Define the target sample rate and minimum duration (in seconds)
        target_sr = 16000
        min_duration_sec = 1.0  # Require at least 1 second of audio
        
        # Resample if needed
        if original_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
        
        # Check if the resampled audio is long enough
        if len(audio) < target_sr * min_duration_sec:
            raise HTTPException(
                status_code=400,
                detail="Audio file is too short. Please provide at least 1 second of audio."
            )
        
        # Process the audio with Whisper's processor to get input features (log-mel spectrogram)
        input_features = processor(audio, sampling_rate=target_sr, return_tensors="pt").input_features

        # Generate transcription using the Whisper model
        generated_ids = model.generate(input_features)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return {"transcription": transcription}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
