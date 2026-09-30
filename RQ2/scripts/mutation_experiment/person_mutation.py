import random
from lemminflect import getInflection

from voice_shared import (
    nlp,
    detect_prompt_type,
    has_root_object_me,
    has_relcl_first_person,
    first_to_third_pronoun_collision,
    split_sentences,
    find_request_sentence,
    rebuild_prompt,
    rebuild_prompt_from_sentences,
)


THIRD_PERSON_SUBJECTS = ["one", "a person"]

CONTRACTION_EXPANSIONS = {
    "'m": "is",
    "'ve": "has",
    "'d": "would",
    "'ll": "will",
}

_HE_FAMILY = {
    "subjects": {"he"}, "possessive": {"his"},
    "object": {"him"}, "reflexive": {"himself"},
}
_SHE_FAMILY = {
    "subjects": {"she"}, "possessive": {"her"},
    "object": {"her"}, "reflexive": {"herself"},
}
_THEY_FAMILY = {
    "subjects": {"they", "someone", "one", "a person"}, "possessive": {"their"},
    "object": {"them"}, "reflexive": {"themselves"},
}
_FAMILIES = [_HE_FAMILY, _SHE_FAMILY, _THEY_FAMILY]


def _family_for_subject(subj_text):
    for family in _FAMILIES:
        if subj_text in family["subjects"]:
            return family
    return None


def find_verb_complex(subject_token):
    if subject_token is None:
        return set()

    main_verb = subject_token.head
    complex_idxs = {main_verb.i}
    for child in main_verb.children:
        if child.dep_ in ("aux", "auxpass"):
            complex_idxs.add(child.i)
        elif child.dep_ == "conj" and child.pos_ in ("VERB", "AUX"):
            complex_idxs.add(child.i)
            for grandchild in child.children:
                if grandchild.dep_ in ("aux", "auxpass"):
                    complex_idxs.add(grandchild.i)
    return complex_idxs


def explicit_first_to_third(sentence, subject_word):
    doc = nlp(sentence)

    i_token = next(
        (t for t in doc if t.text.lower() == "i" and t.dep_ in ("nsubj", "nsubjpass")),
        None
    )
    verb_complex = find_verb_complex(i_token)

    pieces = []
    i = 0
    n = len(doc)

    while i < n:
        token = doc[i]
        low = token.text.lower()

        if low == "i" and token.dep_ in ("nsubj", "nsubjpass"):
            replacement = subject_word
            if token.i == 0:
                replacement = replacement[0].upper() + replacement[1:]

            next_token = doc[i + 1] if i + 1 < n else None
            if next_token is not None and next_token.text.lower() in CONTRACTION_EXPANSIONS:
                pieces.append(replacement + " ")
                expanded = CONTRACTION_EXPANSIONS[next_token.text.lower()]
                pieces.append(expanded + next_token.whitespace_)
                i += 2
                continue

            pieces.append(replacement + token.whitespace_)
            i += 1
            continue

        if low == "my":
            pieces.append("their" + token.whitespace_)
            i += 1
            continue

        if low == "myself":
            pieces.append("themselves" + token.whitespace_)
            i += 1
            continue

        if low == "me":
            pieces.append("them" + token.whitespace_)
            i += 1
            continue

        if token.tag_ == "VBP" and token.i in verb_complex:
            infl = getInflection(token.lemma_, tag="VBZ")
            replacement = infl[0] if infl else token.text
            pieces.append(replacement + token.whitespace_)
            i += 1
            continue

        pieces.append(token.text + token.whitespace_)
        i += 1

    return "".join(pieces).strip()


def explicit_third_to_first(sentence):
    doc = nlp(sentence)

    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return None

    subj = next(
        (c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")),
        None
    )
    if subj is None:
        return None

    subj_text = subj.text.lower()
    a_person_det = None
    if subj_text == "person":
        a_person_det = next(
            (c for c in subj.children if c.dep_ == "det" and c.text.lower() == "a"),
            None
        )
        if a_person_det is not None:
            subj_text = "a person"

    family = _family_for_subject(subj_text)
    if family is None:
        return None

    verb_complex = find_verb_complex(subj)

    pieces = []
    i = 0
    n = len(doc)

    while i < n:
        token = doc[i]
        low = token.text.lower()

        if a_person_det is not None and token.i == a_person_det.i:
            pieces.append("I ")
            i += 1
            continue
        if a_person_det is not None and token.i == subj.i:
            i += 1  # already emitted "I" at the determiner above
            continue

        if low in family["subjects"] and token.dep_ in ("nsubj", "nsubjpass"):
            next_token = doc[i + 1] if i + 1 < n else None
            key = next_token.text.lower() if next_token is not None else None
            if key == "'s":
                lemma = next_token.lemma_.lower()
                pieces.append(("I'm" if lemma == "be" else "I've") + next_token.whitespace_)
                i += 2
                continue
            if key == "'re":
                pieces.append("I'm" + next_token.whitespace_)
                i += 2
                continue
            if key in ("'ll", "'d"):
                pieces.append("I" + key + next_token.whitespace_)
                i += 2
                continue

            pieces.append("I" + token.whitespace_)
            i += 1
            continue

        if low in family["possessive"] or low in family["object"]:
            if low == "her":
                replacement = "my" if token.dep_ == "poss" else "me"
            elif low in family["possessive"]:
                replacement = "my"
            else:
                replacement = "me"
            pieces.append(replacement + token.whitespace_)
            i += 1
            continue

        if low in family["reflexive"]:
            pieces.append("myself" + token.whitespace_)
            i += 1
            continue

        if token.i in verb_complex:
            lemma = token.lemma_.lower()
            if lemma == "be" and token.tag_ in ("VBZ", "VBP"):
                pieces.append("am" + token.whitespace_)
                i += 1
                continue
            if token.tag_ == "VBZ":
                infl = getInflection(lemma, tag="VBP")
                replacement = infl[0] if infl else token.text
                pieces.append(replacement + token.whitespace_)
                i += 1
                continue

        pieces.append(token.text + token.whitespace_)
        i += 1

    return "".join(pieces).strip()


def _first_to_third_label(sentence, prompt):
    label = None
    prompt_type = detect_prompt_type(sentence)
    if prompt_type == "first_person":
        label = "first_person"
    elif has_root_object_me(sentence):
        label = "first_person_object"
    elif has_relcl_first_person(sentence):
        label = "first_person_relcl"

    if label is None:
        return None
    if first_to_third_pronoun_collision(sentence, prompt):
        return None
    return label


def person_first_to_third(prompt, num_variants=5):
    prompt = prompt.strip()
    if len(prompt) < 3:
        return []

    sentences = split_sentences(prompt)
    labels = [_first_to_third_label(sentence, prompt) for sentence in sentences]
    eligible_idxs = [i for i, label in enumerate(labels) if label is not None]
    if not eligible_idxs:
        return []

    detected_label = "+".join(sorted(set(labels[i] for i in eligible_idxs)))

    candidates = []
    for subject_word in THIRD_PERSON_SUBJECTS:
        mutated_sentences = list(sentences)
        for i in eligible_idxs:
            mutated_sentences[i] = explicit_first_to_third(sentences[i], subject_word)
        candidates.append(mutated_sentences)

    sampled = random.sample(candidates, min(num_variants, len(candidates)))

    results = []
    seen = set()

    for idx, mutated_sentences in enumerate(sampled):
        mutated = rebuild_prompt_from_sentences(mutated_sentences)

        if mutated.lower() == prompt.lower():
            continue
        if mutated in seen:
            continue

        seen.add(mutated)
        results.append({
            "mutation_id":          idx + 1,
            "mutation_type":        "person_first_to_third",
            "prompt_type_detected": detected_label,
            "target_person":        "third",
            "original":             prompt,
            "mutated_prompt":       mutated
        })

    return results


def person_third_to_first(prompt, num_variants=5):
    prompt = prompt.strip()
    if len(prompt) < 3:
        return []

    sentences = split_sentences(prompt)
    req_idx = find_request_sentence(sentences)
    request_sentence = sentences[req_idx]
    before = sentences[:req_idx]
    after = sentences[req_idx + 1:]

    prompt_type = detect_prompt_type(request_sentence)
    if prompt_type != "third_person":
        return []

    mutated_sentence = explicit_third_to_first(request_sentence)
    if not mutated_sentence:
        return []

    mutated = rebuild_prompt(before, mutated_sentence, after)
    if mutated.lower() == prompt.lower():
        return []

    return [{
        "mutation_id":          1,
        "mutation_type":        "person_third_to_first",
        "prompt_type_detected": prompt_type,
        "target_person":        "first",
        "original":             prompt,
        "mutated_prompt":       mutated
    }]


def person_first_to_third_applicable(prompt):
    prompt = prompt.strip()
    if len(prompt) < 3:
        return False
    sentences = split_sentences(prompt)
    return any(
        _first_to_third_label(sentence, prompt) is not None for sentence in sentences
    )


def person_third_to_first_applicable(prompt):
    prompt = prompt.strip()
    if len(prompt) < 3:
        return False
    sentences = split_sentences(prompt)
    req_idx = find_request_sentence(sentences)
    return detect_prompt_type(sentences[req_idx]) == "third_person"
