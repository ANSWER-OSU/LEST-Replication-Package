import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "textattack/roberta-base-CoLA"
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

print(f"loading CoLA grammar classifier... (device={DEVICE})")
_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval().to(DEVICE)

# CoLA convention: label 1 = linguistically acceptable, label 0 = not.
_ACCEPTABLE_LABEL = 1


def _acceptability(text):
    enc = _tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = _model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return probs[_ACCEPTABLE_LABEL].item()


def grammar_score(mutated, threshold=0.5):
    acceptability = _acceptability(mutated)
    return {
        "grammatical":   acceptability >= threshold,
        "acceptability": acceptability,
    }
