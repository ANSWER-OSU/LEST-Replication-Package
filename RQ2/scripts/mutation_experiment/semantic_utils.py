import os
from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer, util

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)


@lru_cache(maxsize=None)
def _embed(text):
    return model.encode(text, convert_to_tensor=True)


def semantic_preserved(
    original,
    mutated,
    threshold=0.80
):

    emb1 = _embed(original)
    emb2 = _embed(mutated)

    similarity = util.cos_sim(
        emb1, emb2
    ).item()

    preserved = similarity >= threshold

    return {
        "preserved":  preserved,
        "similarity": similarity
    }
