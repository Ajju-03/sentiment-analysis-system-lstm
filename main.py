from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = FastAPI(
    title="Sentiment Prediction API",
    description="FastAPI-based ML inference service"
)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


try:
    model = joblib.load("logistic_regression_model.pk1")
    vectorizer = joblib.load("tfidf_vectorizer.pk1")
    print("Model and vectorizer loaded successfully")
except Exception as e:
    print(f"Failed to load model/vectorizer: {e}")
    model = None
    vectorizer = None


class ReviewRequest(BaseModel):
    review: str


class PredictionResponse(BaseModel):
    sentiment: str


# ---------------------------
# Text preprocessing
# ---------------------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = text.split()
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(cleaned_tokens)


# ---------------------------
# Prediction endpoint
# ---------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest):

    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model or vectorizer not loaded")

    if not request.review.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty")

    cleaned_text = preprocess_text(request.review)

    try:
        text_vector = vectorizer.transform([cleaned_text])
        proba = model.predict_proba(text_vector)[0]
        prob_positive = float(proba[1])
        sentiment = "Positive" if prob_positive >= 0.35 else "Negative"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return {
        "sentiment": sentiment,
        "probability": round(prob_positive, 2)
    }
