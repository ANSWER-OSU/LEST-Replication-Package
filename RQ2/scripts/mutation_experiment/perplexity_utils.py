import os
from functools import lru_cache

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

print(f"loading gpt2 for perplexity... (device={DEVICE})")
_tok = GPT2TokenizerFast.from_pretrained("gpt2")
_model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)


@lru_cache(maxsize=None)
def _perplexity(text):
    enc = _tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = _model(**enc, labels=enc["input_ids"])
    return torch.exp(out.loss).item()


def perplexity_score(original, mutated, threshold=2.0):
    o = _perplexity(original)
    m = _perplexity(mutated)
    ratio = m / o if o > 0 else float("inf")
    return {
        "grammatical":   ratio <= threshold,
        "ppl_ratio":     ratio,
        "mutated_ppl":   m,
        "original_ppl":  o,
    }
