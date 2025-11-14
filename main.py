from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import json
import os

# ---------- Загрузка модели ----------
MODEL_DIR = "goemotions-bert-multilabel/final_model"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

with open(os.path.join(MODEL_DIR, "label_names.json"), "r") as f:
    label_names = json.load(f)


def predict_emotions(text: str, threshold: float = 0.4):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    indices = np.where(probs >= threshold)[0]
    results = [
        {"label": label_names[i], "score": float(probs[i])}
        for i in indices
    ]

    # сортировка по вероятности
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# ---------- Схемы JSON ----------
class PredictRequest(BaseModel):
    text: str
    threshold: float = 0.4

class Emotion(BaseModel):
    label: str
    score: float

class PredictResponse(BaseModel):
    text: str
    emotions: List[Emotion]

# ---------- FastAPI ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    emotions = predict_emotions(req.text, req.threshold)
    return PredictResponse(text=req.text, emotions=emotions)

@app.get("/health")
def health():
    return {"status": "ok"}
