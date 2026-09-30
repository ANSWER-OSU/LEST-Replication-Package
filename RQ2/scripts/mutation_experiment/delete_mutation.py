import re
import random
import spacy

nlp = spacy.load("en_core_web_sm")


def delete_mutation(prompt, max_deletions=1):
    """Delete up to max_deletions adjectives/adverbs. Returns str or None."""
    if not prompt or not isinstance(prompt, str):
        return None

    doc = nlp(prompt)
    eligible = [
        token for token in doc
        if token.pos_ in ("ADJ", "ADV") and token.dep_ != "acomp"
    ]
    if not eligible:
        return None

    n = random.randint(1, min(max_deletions, len(eligible)))
    drop = set(random.sample(eligible, n))
    drop_idxs = {token.i for token in drop}

    kept = "".join(
        token.text_with_ws for token in doc if token.i not in drop_idxs
    )
    mutated = kept.strip()
    mutated = re.sub(r"\s+([.,;:!?'])", r"\1", mutated)
    mutated = re.sub(r"\s{2,}", " ", mutated)
    mutated = re.sub(r"([.,;:!?])\s*\1+", r"\1", mutated)
    mutated = re.sub(r"^[,;:]\s*", "", mutated).strip()
    mutated = mutated[:1].upper() + mutated[1:] if mutated else mutated

    if not mutated or mutated == prompt:
        return None
    return mutated
