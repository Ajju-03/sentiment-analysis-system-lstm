import warnings
# FutureWarning from the tf2onnx library
warnings.filterwarnings("ignore", category=FutureWarning, module="keras.src.export.tf2onnx_lib")
import os
# Set the environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np
import re

app = FastAPI(
    title="Sentiment Prediction API",
    description="FastAPI-based ML inference service"
)

# Load model and tokenizer
try:
    model = keras.models.load_model("sentiment_analysis2.keras")
    tokenizer = joblib.load("tokenizer.pkl")
    print("Model and tokenizer loaded successfully")
except Exception as e:
    print(f"Failed to load model/tokenizer: {e}")
    model = None
    tokenizer = None

# Review
class ReviewRequest(BaseModel):
    review: str
# Sentiment & Probability
class PredictionResponse(BaseModel):
    sentiment: str
    probability: float
    
# text preprocessing 
def preprocess_text(text: str) -> np.ndarray:
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    MAX_LEN = 200
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    return padded

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest):
    # model and tokenizer
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model or vectorizer not loaded")
    # if review not submitted
    if not request.review.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty")

    # handle text & 
    seq_padded = preprocess_text(request.review)
    prediction = model.predict(seq_padded)
    prob_positive = float(prediction[0][0]) 
    
    sentiment = "Positive" if prob_positive >= 0.5 else "Negative"

    return {
        "sentiment": sentiment,
        "probability": round(prob_positive, 2)
    }

