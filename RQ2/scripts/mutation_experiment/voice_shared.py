import re
import spacy
from functools import lru_cache

_NLP = spacy.load("en_core_web_sm")


@lru_cache(maxsize=1024)
def nlp(text):
    return _NLP(text)


def split_sentences(prompt):
    return re.split(r'(?<!\b[A-Z]\.)(?<!e\.g\.)(?<=[.!?])\s+', prompt.strip())


_FIRST_PERSON_OPENERS = {
    "i", "i'm", "i've", "i'd", "i'll"
}

_THIRD_PERSON_SUBJECTS = {"he", "she", "they", "someone", "one"}

_FIRST_SECOND_PERSON_TOKENS = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself"
}

_FIRST_PERSON_SWAP_TOKENS = re.compile(r"\b(me|my|myself)\b", re.I)
_THEY_FAMILY_PRONOUNS = re.compile(
    r"\b(they|them|their|theirs|themselves|themself)\b", re.I
)


def _is_explicit_third_person(prompt):
    doc = nlp(prompt.strip())

    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return False

    subj = next(
        (c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")),
        None
    )
    if subj is None:
        return False

    subj_text = subj.text.lower()
    is_third_subject = subj_text in _THIRD_PERSON_SUBJECTS
    if not is_third_subject and subj_text == "person":
        is_third_subject = any(
            c.dep_ == "det" and c.text.lower() == "a"
            for c in subj.children
        )

    if not is_third_subject:
        return False

    return not any(
        token.i != subj.i and token.text.lower() in _FIRST_SECOND_PERSON_TOKENS
        for token in doc
    )


def has_root_object_me(prompt):
    doc = nlp(prompt.strip())
    return any(
        token.text.lower() == "me"
        and token.dep_ in ("dative", "dobj")
        and token.head.dep_ == "ROOT"
        for token in doc
    )


def has_relcl_first_person(prompt):
    doc = nlp(prompt.strip())
    return any(
        token.text.lower() in ("i", "me")
        and token.dep_ in ("nsubj", "nsubjpass")
        and token.head.dep_ == "relcl"
        for token in doc
    )


def first_to_third_pronoun_collision(sentence, prompt):
    if not _FIRST_PERSON_SWAP_TOKENS.search(sentence):
        return False
    return bool(_THEY_FAMILY_PRONOUNS.search(prompt))


def detect_prompt_type(prompt):
    p = prompt.strip().lower()
    first_word = p.split()[0] if p.split() else ""

    if first_word in _FIRST_PERSON_OPENERS:
        if re.search(
            r"\b(tell me|give me|show me|help me|say me|ask me|"
            r"let me know|inspire me|suggest|recommend)\b",
            p
        ):
            return "mixed"
        return "first_person"

    if _is_explicit_third_person(prompt):
        return "third_person"

    return "other"


def find_request_sentence(sentences):
    for i, sentence in enumerate(sentences):
        if detect_prompt_type(sentence) != "first_person":
            return i
    return len(sentences) - 1


def rebuild_prompt(before, mutated_sentence, after):
    mutated_sentence = mutated_sentence.strip()
    if mutated_sentence and mutated_sentence[-1] not in ".!?":
        mutated_sentence += "."

    mutated = " ".join(before + [mutated_sentence] + after)
    return re.sub(r"\s+", " ", mutated).strip()


def rebuild_prompt_from_sentences(sentences):
    pieces = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence[-1] not in ".!?":
            sentence += "."
        pieces.append(sentence)

    mutated = " ".join(pieces)
    return re.sub(r"\s+", " ", mutated).strip()
