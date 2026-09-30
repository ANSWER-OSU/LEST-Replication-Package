from lemminflect import getInflection

from voice_shared import (
    nlp,
    split_sentences,
    find_request_sentence,
    rebuild_prompt,
)


_SUBJ_TO_OBJ_PRONOUN = {
    "i": "me", "we": "us", "he": "him", "she": "her",
    "they": "them", "you": "you", "it": "it",
}
_OBJ_TO_SUBJ_PRONOUN = {v: k for k, v in _SUBJ_TO_OBJ_PRONOUN.items()}

_TIGHT_DEPS = ("prt", "acomp", "oprd")

_OBJECT_MODIFIER_DEPS = ("advcl", "prep")


def find_passive_target(doc):
    for token in doc:
        if token.dep_ != "ROOT" or token.pos_ != "VERB":
            continue

        if any(
            child.dep_ == "conj" and child.pos_ == "VERB"
            for child in token.children
        ):
            return None

        if any(child.dep_ == "neg" for child in token.children):
            return None

        if any(child.dep_ == "aux" for child in token.children):
            return None

        if any(child.dep_ == "dative" for child in token.children):
            return None

        for child in token.children:
            if child.dep_ in ("dobj", "obj") and child.pos_ in ("NOUN", "PROPN"):
                return token, child

    return None


def find_active_target(doc):
    for token in doc:
        if token.dep_ != "ROOT" or token.pos_ != "VERB":
            continue

        has_auxpass = any(c.dep_ == "auxpass" for c in token.children)
        nsubjpass = next(
            (c for c in token.children if c.dep_ == "nsubjpass"), None
        )
        if not has_auxpass or nsubjpass is None:
            return None

        if any(
            child.dep_ == "conj" and child.pos_ == "VERB"
            for child in token.children
        ):
            return None

        if any(child.dep_ == "aux" for child in token.children):
            return None

        if any(child.dep_ == "neg" for child in token.children):
            return None

        agent_token = None
        for child in token.children:
            if child.text.lower() == "by" and child.dep_ in ("agent", "prep"):
                agent_token = next(
                    (g for g in child.children if g.dep_ == "pobj"), None
                )
                break

        if agent_token is None:
            return None

        return token, nsubjpass, agent_token

    return None


def _phrase(subtree):
    return "".join(t.text_with_ws for t in subtree).strip()


def _is_plural_np(head_token):
    if head_token.tag_ in ("NNS", "NNPS"):
        return True
    if head_token.text.lower() in ("they", "we", "them", "us"):
        return True
    if any(c.dep_ == "conj" for c in head_token.children):
        return True
    return False


def _participle(lemma):
    infl = getInflection(lemma, tag="VBN")
    return infl[0] if infl else lemma


def _capitalize(text):
    return text[0].upper() + text[1:] if text else text


def _join_pieces(pieces):
    out = ""
    for piece in pieces:
        if not piece:
            continue
        if not out:
            out = piece
        elif piece.startswith(","):
            out += piece
        else:
            out += " " + piece
    return out


def _split_remaining_children(root, excluded_idxs):
    tight, free_before, free_after = [], [], []
    comma_before = {
        c.i + 1 for c in root.children
        if c.dep_ == "punct" and c.text == ","
    }

    for child in sorted(root.children, key=lambda t: t.i):
        if child.i in excluded_idxs or child.dep_ == "punct":
            continue
        text = _phrase(sorted(child.subtree, key=lambda t: t.i))
        is_stranded_prep = (
            child.dep_ == "prep"
            and not any(c.dep_ == "pobj" for c in child.children)
        )
        if child.dep_ in _TIGHT_DEPS or is_stranded_prep:
            tight.append(text)
        elif child.i < root.i:
            free_before.append(text)
        else:
            if child.i in comma_before:
                text = ", " + text
            free_after.append(text)

    return tight, free_before, free_after


def _extended_object_subtree(root, obj_token):
    idxs = {t.i for t in obj_token.subtree}
    changed = True
    while changed:
        changed = False
        edge = max(idxs) + 1
        for child in root.children:
            if child.dep_ not in _OBJECT_MODIFIER_DEPS:
                continue
            child_idxs = {t.i for t in child.subtree}
            if child_idxs and min(child_idxs) == edge:
                idxs |= child_idxs
                changed = True
    return idxs


def explicit_active_to_passive(sentence):
    doc = nlp(sentence)
    target = find_passive_target(doc)
    if target is None:
        return None
    root, dobj = target

    subj = next((c for c in root.children if c.dep_ == "nsubj"), None)

    object_idxs = _extended_object_subtree(root, dobj)
    new_subject_phrase = _phrase(sorted(
        (t for t in doc if t.i in object_idxs), key=lambda t: t.i
    ))
    plural_subject = _is_plural_np(dobj)

    # INSERT auxpass(X0,X2): X2 carries the original tense
    is_past = root.tag_ == "VBD"
    be_form = ("was" if not plural_subject else "were") if is_past else \
              ("is" if not plural_subject else "are")

    # NODE OP: X0 -> past participle
    participle = _participle(root.lemma_)

    agent_phrase = None
    if subj is not None:
        subj_tokens = sorted(subj.subtree, key=lambda t: t.i)
        subj_text = _phrase(subj_tokens)
        low = subj_text.lower()
        was_sentence_initial = subj_tokens[0].i == 0
        subj_text = _SUBJ_TO_OBJ_PRONOUN.get(
            low, low if was_sentence_initial else subj_text
        )
        agent_phrase = f"by {subj_text}"

    excluded = {root.i} | object_idxs
    if subj is not None:
        excluded |= {t.i for t in subj.subtree}
    tight, free_before, free_after = _split_remaining_children(root, excluded)

    pieces = (
        free_before
        + [new_subject_phrase, be_form, participle]
        + tight
        + ([agent_phrase] if agent_phrase else [])
        + free_after
    )
    return _capitalize(_join_pieces(pieces))


def explicit_passive_to_active(sentence):
    doc = nlp(sentence)
    target = find_active_target(doc)
    if target is None:
        return None
    root, nsubjpass, agent_token = target

    auxpass = next((c for c in root.children if c.dep_ == "auxpass"), None)
    agent_prep = next(
        (c for c in root.children if c.dep_ == "agent" or c.text.lower() == "by"),
        None
    )

    # DELETE agent(X0,X3) / INSERT nsubj(X0,X3): promote the agent
    new_subject_phrase = _phrase(sorted(agent_token.subtree, key=lambda t: t.i))
    low = new_subject_phrase.lower()
    new_subject_phrase = _OBJ_TO_SUBJ_PRONOUN.get(low, new_subject_phrase)
    plural_subject = _is_plural_np(agent_token)

    nsubjpass_tokens = sorted(nsubjpass.subtree, key=lambda t: t.i)
    new_object_phrase = _phrase(nsubjpass_tokens)
    low = new_object_phrase.lower()
    was_sentence_initial = nsubjpass_tokens[0].i == 0
    new_object_phrase = _SUBJ_TO_OBJ_PRONOUN.get(
        low, low if was_sentence_initial else new_object_phrase
    )

    # NODE OP: X0 inherits tense from X2 (auxpass), agrees in number with X3
    is_past = auxpass.text.lower() in ("was", "were")
    tag = "VBD" if is_past else ("VBZ" if not plural_subject else "VBP")
    infl = getInflection(root.lemma_, tag=tag)
    new_verb = infl[0] if infl else root.lemma_

    excluded = {root.i, auxpass.i} | {t.i for t in nsubjpass.subtree}
    excluded |= {t.i for t in agent_token.subtree}
    if agent_prep is not None:
        excluded.add(agent_prep.i)
    tight, free_before, free_after = _split_remaining_children(root, excluded)

    pieces = (
        free_before
        + [new_subject_phrase, new_verb]
        + tight
        + [new_object_phrase]
        + free_after
    )
    return _capitalize(_join_pieces(pieces))


def active_to_passive_mutation(prompt, num_variants=5):
    prompt = prompt.strip()
    if len(prompt) < 3:
        return []

    sentences = split_sentences(prompt)
    req_idx = find_request_sentence(sentences)
    request_sentence = sentences[req_idx]
    before = sentences[:req_idx]
    after = sentences[req_idx + 1:]

    mutated_request = explicit_active_to_passive(request_sentence)
    if not mutated_request:
        return []

    mutated = rebuild_prompt(before, mutated_request, after)
    if mutated.lower() == prompt.lower():
        return []

    return [{
        "mutation_id":     1,
        "mutation_type":   "active_to_passive",
        "original":        prompt,
        "mutated_prompt":  mutated
    }]


def passive_to_active_mutation(prompt, num_variants=5):
    """See active_to_passive_mutation's docstring re: num_variants."""
    prompt = prompt.strip()
    if len(prompt) < 3:
        return []

    sentences = split_sentences(prompt)
    req_idx = find_request_sentence(sentences)
    request_sentence = sentences[req_idx]
    before = sentences[:req_idx]
    after = sentences[req_idx + 1:]

    mutated_request = explicit_passive_to_active(request_sentence)
    if not mutated_request:
        return []

    mutated = rebuild_prompt(before, mutated_request, after)
    if mutated.lower() == prompt.lower():
        return []

    return [{
        "mutation_id":     1,
        "mutation_type":   "passive_to_active",
        "original":        prompt,
        "mutated_prompt":  mutated
    }]


def active_to_passive_applicable(prompt):
    prompt = prompt.strip()
    if len(prompt) < 3:
        return False
    sentences = split_sentences(prompt)
    req_idx = find_request_sentence(sentences)
    doc = nlp(sentences[req_idx])
    return find_passive_target(doc) is not None


def passive_to_active_applicable(prompt):
    prompt = prompt.strip()
    if len(prompt) < 3:
        return False
    sentences = split_sentences(prompt)
    req_idx = find_request_sentence(sentences)
    doc = nlp(sentences[req_idx])
    return find_active_target(doc) is not None
