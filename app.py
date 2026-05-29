from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import os
import uvicorn

from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification
from peft import PeftModel

app = FastAPI(
    title       = "Financial Sentiment Analysis API",
    description = "Sentiment prediction using DistilRoBERTa + LoRA",
    version     = "1.0.0"
)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import os
import uvicorn

from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification
from peft import PeftModel

# ──────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────
app = FastAPI(
    title       = "Financial Sentiment Analysis API",
    description = "Sentiment prediction using DistilRoBERTa + LoRA",
    version     = "1.0.0"
)

# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────
MODEL_NAME  = 'distilroberta-base'
MAX_SEQ_LEN = 64
NUM_CLASSES = 3
SAVE_DIR    = r'C:\Users\DSU\OneDrive - Dakota State University\Desktop\pytorch\saved_models'

LABEL_MAP = {
    0: 'Negative',
    1: 'Neutral',
    2: 'Positive'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ──────────────────────────────────────────
# Load Model & Tokenizer at Startup
# ──────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

print("Loading base model...")
base_model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels = NUM_CLASSES
)

print("Loading LoRA adapter weights...")
model = PeftModel.from_pretrained(base_model, SAVE_DIR)
model = model.to(device)
model.eval()

print("Model loaded successfully on:", device)

# ──────────────────────────────────────────
# Request / Response Schemas
# ──────────────────────────────────────────
class SingleRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "The company reported record profits this quarter."
            }
        }

class BatchRequest(BaseModel):
    texts: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "The company reported record profits this quarter.",
                    "Stock declined sharply after earnings miss.",
                    "Board approved the quarterly dividend as expected."
                ]
            }
        }

class PredictionResponse(BaseModel):
    text        : str
    sentiment   : str
    confidence  : float
    probabilities: dict

class BatchResponse(BaseModel):
    results     : List[PredictionResponse]
    total       : int

# ──────────────────────────────────────────
# Helper — Predict One Sentence
# ──────────────────────────────────────────
def predict_single(text: str) -> PredictionResponse:
    encoding = tokenizer(
        str(text),
        max_length     = MAX_SEQ_LEN,
        padding        = 'max_length',
        truncation     = True,
        return_tensors = 'pt'
    )

    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask
        )
        logits = outputs.logits
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()

    return PredictionResponse(
        text         = text,
        sentiment    = LABEL_MAP[pred],
        confidence   = round(probs[pred].item(), 4),
        probabilities = {
            'Negative': round(probs[0].item(), 4),
            'Neutral' : round(probs[1].item(), 4),
            'Positive': round(probs[2].item(), 4)
        }
    )

# ──────────────────────────────────────────
# Routes
# ──────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message" : "Financial Sentiment Analysis API",
        "model"   : "DistilRoBERTa + LoRA",
        "device"  : str(device),
        "endpoints": {
            "health"    : "/health",
            "predict"   : "/predict",
            "batch"     : "/predict/batch",
            "docs"      : "/docs"
        }
    }


@app.get("/health")
def health():
    return {
        "status"  : "healthy",
        "model"   : "DistilRoBERTa + LoRA",
        "device"  : str(device),
        "labels"  : list(LABEL_MAP.values())
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: SingleRequest):
    """
    Predict sentiment for a single text sentence.
    Returns: sentiment label, confidence score, and all class probabilities.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        result = predict_single(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    """
    Predict sentiment for a list of sentences.
    Returns: list of predictions with sentiment, confidence, and probabilities.
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty.")

    if len(request.texts) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 texts per batch.")

    try:
        results = [predict_single(text) for text in request.texts]
        return BatchResponse(results=results, total=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────
# Run
# ──────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)