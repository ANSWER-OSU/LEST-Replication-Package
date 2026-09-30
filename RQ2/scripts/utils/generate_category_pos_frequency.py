import json
from pathlib import Path
from collections import Counter

import pandas as pd
import spacy

CATEGORY_IDS = [7]  # Change this

VALID_POS = {"NOUN", "VERB", "ADJ"}

nlp = spacy.load("en_core_web_sm")

for category_id in CATEGORY_IDS:

    input_path = Path(
        f"RQ1/outputs/processed/category{category_id}_sorrybench.json"
    )

    output_path = Path(
        f"RQ2/outputs/category{category_id}_pos_frequency.csv"
    )

    print(f"\nProcessing category {category_id}...")

    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    counter = Counter()

    for item in dataset:

        doc = nlp(item["prompt"])

        for token in doc:

            if token.pos_ not in VALID_POS:
                continue

            if token.is_stop:
                continue

            if not token.is_alpha:
                continue

            lemma = token.lemma_.lower().strip()

            if len(lemma) < 2:
                continue

            counter[(lemma, token.pos_)] += 1

    rows = [
        {
            "word": lemma,
            "pos": pos,
            "frequency": freq,
        }
        for (lemma, pos), freq in counter.items()
    ]

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by="frequency",
        ascending=False,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved to {output_path}")
    print(df.head(10))
