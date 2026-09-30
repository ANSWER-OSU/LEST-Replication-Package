import os
import random
import re
from dataclasses import dataclass

import numpy as np
import spacy
from lemminflect import getInflection, getLemma, getAllInflections
from wordfreq import zipf_frequency


# Path to the counter-fitted vectors. None = auto-find next to this file.
VECTORS_PATH = None

# spaCy model. Only does tagging/lemmas here, so en_core_web_sm is fine.
SPACY_MODEL = "en_core_web_sm"

# Distinct mutations mutate_prompt_batch returns per prompt.
NUM_MUTATIONS = 10

# Max content words swapped in a single mutation.
MAX_PERTURB = 3

# Neighbors pulled per word before filtering.
MAX_CANDIDATES = 12

# Rank-temperature for neighbor sampling. ~0 greedy, larger explores deeper.
TEMP = 1.0

# Junk-neighbor frequency floor (Zipf: ~7="the", ~4=common, <2.5=junk). 0 off.
MIN_ZIPF = 2.4

# Cosine floor on neighbors. Counter-fitted needs none. 0 off.
MIN_SIM = 0.0


_PENN_TO_UPOS = {}
for _t in ("NN", "NNS", "NNP", "NNPS"):
    _PENN_TO_UPOS[_t] = "NOUN"
for _t in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
    _PENN_TO_UPOS[_t] = "VERB"
for _t in ("JJ", "JJR", "JJS"):
    _PENN_TO_UPOS[_t] = "ADJ"
for _t in ("RB", "RBR", "RBS"):
    _PENN_TO_UPOS[_t] = "ADV"

CONTENT_UPOS = {"NOUN", "VERB", "ADJ", "ADV"}

# Flipping these changes meaning instead of preserving it.
NEGATION = {"not", "n't", "no", "never", "none", "neither", "nor",
            "without", "cannot", "nothing", "nowhere", "nobody"}

# Content-tagged but rarely have a clean synonym; cheaper to skip.
SKIP_LEMMAS = {"be", "have", "do", "use", "get", "make", "go", "thing", "way",
               "very", "really", "also", "more", "most", "such"}

_WORD_RE = re.compile(r"^[a-zA-Z][a-zA-Z\-]*$")


def coarse(tag):
    return _PENN_TO_UPOS.get(tag, "")


def match_case(original, new):
    if original.isupper() and len(original) > 1:
        return new.upper()
    if original[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


# a/an exceptions where the article disagrees with the spelling (sound vs letter)
_AN_EXCEPTIONS = {"hour", "honest", "honor", "honour", "heir"}
_A_EXCEPTIONS = {"university", "unicorn", "european", "one", "user",
                 "unit", "useful", "unique", "ufo"}


def _article_for(word):
    w = word.lower().strip(".,!?;:\"'()")
    if not w:
        return "a"
    if w in _AN_EXCEPTIONS:
        return "an"
    if w in _A_EXCEPTIONS:
        return "a"
    return "an" if w[0] in "aeiou" else "a"


def fix_articles(text):
    """Repair a/an agreement broken by a swap ('a elaborate' -> 'an elaborate')."""
    def repl(m):
        art, nxt = m.group(1), m.group(2)
        correct = _article_for(nxt)
        if art[0].isupper():
            correct = correct.capitalize()
        return f"{correct} {nxt}"
    return re.sub(r"\b([Aa]n?)\s+([A-Za-z]+)", repl, text)


def real_word(w, min_zipf):
    #Ensures that the generated word is real
    if min_zipf <= 0:
        return True
    return zipf_frequency(w.lower(), "en") >= min_zipf


def attested_pos(word):
    try:
        infl = getAllInflections(word.lower())
    except Exception:
        return set()
    pos = set()
    for tag in infl:
        if tag.startswith("VB"):
            pos.add("VERB")
        elif tag.startswith("NN"):
            pos.add("NOUN")
        elif tag.startswith("JJ"):
            pos.add("ADJ")
        elif tag.startswith("RB"):
            pos.add("ADV")
    return pos


def _looks_like_typo(a, b):
    if a == b:
        return False
    if a in b or b in a:
        return True
    if len(a) == len(b) >= 5 and sum(x != y for x, y in zip(a, b)) == 1:
        return True
    return False


@dataclass
class Tok:
    text: str
    start: int
    end: int
    tag: str
    lemma: str
    protected: bool


def _in_hyphen_compound(text, start, end):
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return before == "-" or after == "-"


def _vectors_path():
    if VECTORS_PATH:
        return VECTORS_PATH
    here = os.path.dirname(os.path.abspath(__file__))
    name = "counter-fitted-vectors.txt"
    for cand in (os.path.join(here, name), name):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"{name} not found. Run: python get_counterfitted.py")


def _load_vectors(path):
    words, rows, seen = [], [], set()
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            if len(parts) < 5:
                continue
            w = parts[0].lower()
            if w in seen or not _WORD_RE.match(w):
                continue
            seen.add(w)
            words.append(w)
            rows.append(np.asarray(parts[1:], dtype=np.float32))
    matrix = np.vstack(rows)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = (matrix / norms).astype(np.float32)
    return words, {w: i for i, w in enumerate(words)}, matrix


_NLP = spacy.load(SPACY_MODEL)
_VEC_WORDS, _VEC_INDEX, _VEC_MATRIX = _load_vectors(_vectors_path())


def _parse(text):
    toks = []
    for t in _NLP(text):
        start, end = t.idx, t.idx + len(t.text)
        protected = (
            bool(t.ent_type_) or t.like_num or t.is_currency
            or t.lemma_.lower() in NEGATION or t.text.lower() in NEGATION
            or t.lemma_.lower() in SKIP_LEMMAS
            or coarse(t.tag_) not in CONTENT_UPOS
            or t.pos_ == "PRON"
            or _in_hyphen_compound(text, start, end)
        )
        toks.append(Tok(t.text, start, end, t.tag_, t.lemma_.lower(), protected))
    return toks


def _neighbors(word, k):
    i = _VEC_INDEX.get(word.lower())
    if i is None:
        return []
    sims = _VEC_MATRIX @ _VEC_MATRIX[i]
    n = min(k + 8, len(_VEC_WORDS) - 1)
    top = np.argpartition(-sims, n)[:n + 1]
    top = top[np.argsort(-sims[top])]
    out = []
    for j in top:
        if j == i:
            continue
        out.append((_VEC_WORDS[j], float(sims[j])))
        if len(out) >= k:
            break
    return out


def _candidates(tok):
    """Cleaned, re-inflected, junk- and wrong-PoS-filtered replacements."""
    out = []
    orig = tok.text.lower()
    upos = coarse(tok.tag)
    for nb, sim in _neighbors(tok.lemma, MAX_CANDIDATES):
        if sim < MIN_SIM:
            continue
        if "_" in nb or " " in nb or "-" in nb:
            continue
        if not _WORD_RE.match(nb) or nb == orig or nb == tok.lemma:
            continue
        lem = getLemma(nb, upos=upos) if upos else ()
        lemma = lem[0] if lem else nb
        if upos and upos not in attested_pos(lemma):
            continue
        if _looks_like_typo(lemma, orig):
            continue
        if not real_word(lemma, MIN_ZIPF):
            continue
        infl = getInflection(lemma, tag=tok.tag)
        surface_l = infl[0] if infl else lemma
        if _looks_like_typo(surface_l, orig):
            continue
        if not real_word(surface_l, MIN_ZIPF):
            continue
        surface = match_case(tok.text, surface_l)
        if surface.lower() == orig:
            continue
        if surface not in out:
            out.append(surface)
    return out


def _rank_sample(candidates):
    # candidates arrive best-first; weight by exp(-rank/temp)
    ranks = np.arange(len(candidates))
    w = np.exp(-ranks / max(TEMP, 1e-3))
    w /= w.sum()
    return random.choices(candidates, weights=w, k=1)[0]


def _splice(text, toks, repl):
    pieces, prev = [], 0
    for i, t in enumerate(toks):
        if i in repl:
            pieces.append(text[prev:t.start])
            pieces.append(repl[i])
            prev = t.end
    pieces.append(text[prev:])
    return "".join(pieces)


def _retag_ok(toks, repl, mutated):
    by_start = {t.idx: coarse(t.tag_) for t in _NLP(mutated)}
    offset = 0
    for i, t in enumerate(toks):
        if i in repl:
            got = by_start.get(t.start + offset)
            if got is not None and got != coarse(t.tag):
                return False
            offset += len(repl[i]) - (t.end - t.start)
    return True


def _mutate(text, n, max_perturb):
    toks = _parse(text)
    usable = [(i, c) for i, t in enumerate(toks)
              if not t.protected and (c := _candidates(t))]
    if not usable:
        return []
    cap = min(max_perturb, len(usable))
    results, seen = [], {text.strip()}
    tries, max_tries = 0, n * 25
    while len(results) < n and tries < max_tries:
        tries += 1
        budget = random.randint(1, cap)
        chosen = random.sample(range(len(usable)), budget)
        repl = {usable[idx][0]: _rank_sample(usable[idx][1]) for idx in chosen}
        mutated = fix_articles(_splice(text, toks, repl))
        key = mutated.strip()
        if key in seen or not _retag_ok(toks, repl, mutated):
            continue
        seen.add(key)
        results.append(mutated)
    return results


def mutate_prompt(prompt, max_changes=MAX_PERTURB):
    out = _mutate(prompt, n=1, max_perturb=max_changes)
    return out[0] if out else None


# N distinct, deduped mutations in one call (rank-temperature + dedup).
def mutate_prompt_batch(prompt, n=NUM_MUTATIONS):
    return _mutate(prompt, n=n, max_perturb=MAX_PERTURB)
