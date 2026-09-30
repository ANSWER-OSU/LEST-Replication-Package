import re


def _tokenize(text):
    """Lowercase and split into word tokens, stripping punctuation."""
    return re.findall(r"[a-z0-9']+", text.lower())


def lexical_difference(original, mutated):
    
    orig_set = set(_tokenize(original))
    mut_set  = set(_tokenize(mutated))

    shared    = orig_set & mut_set
    new_words = mut_set - orig_set
    dropped   = orig_set - mut_set
    union     = orig_set | mut_set

    jaccard = len(shared) / len(union) if union else 1.0

    return {
        "jaccard_similarity": round(jaccard, 4),
        "lexical_difference": round(1 - jaccard, 4),
        "num_new_words":      len(new_words),
        "num_dropped_words":  len(dropped),
        "new_words":          sorted(new_words),
        "dropped_words":      sorted(dropped),
    }
