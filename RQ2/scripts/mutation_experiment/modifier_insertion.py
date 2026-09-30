import json
import random
import spacy

nlp = spacy.load("en_core_web_sm")


#Load modifier dictionary
DICTIONARY_PATH = (
    "RQ2/data/dictionary/"
    "modifier_dictionary.json"
)


def load_modifier_dictionary(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Modifier dictionary not found at: {path}"
        )
    except json.JSONDecodeError:
        raise ValueError(
            f"Modifier dictionary is not valid JSON: {path}"
        )


_MODIFIER_DICTIONARY = load_modifier_dictionary(DICTIONARY_PATH)
MODIFIERS = _MODIFIER_DICTIONARY["adverb"]
ADJECTIVES = _MODIFIER_DICTIONARY["adjective"]

MODIFIERS_LOWER = {mod.lower() for mod in MODIFIERS}
ADJECTIVES_LOWER = {mod.lower() for mod in ADJECTIVES}


#Helpers
def clean_spacing(text):
    text = (
        text.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" !", "!")
        .replace(" ?", "?")
        .replace(" ;", ";")
        .replace(" :", ":")
    )
    return text.strip()


def fix_punctuation(mutated):
    while "??" in mutated:
        mutated = mutated.replace("??", "?")
    while ".." in mutated:
        mutated = mutated.replace("..", ".")
    return mutated.strip()


def build_candidate_pool(noun_idx):
    candidates = [("adverb", word) for word in MODIFIERS]

    if noun_idx is not None:
        candidates += [("adjective", word) for word in ADJECTIVES]

    return candidates


def sample_candidates(candidates, num_variants):
    return random.sample(candidates, min(num_variants, len(candidates)))


def get_object_idx(root, doc):
    
    for token in root.children:
        if (
            token.dep_ in ["dobj", "dative", "oprd"]
            and token.pos_ == "PRON"
            and token.i == root.i + 1
        ):
            return token.i
    return None


def get_anchor_idx(doc, root, aux_idx, subj_idx):
 

    # rule 1: question structure
    if (
        aux_idx is not None
        and subj_idx is not None
        and subj_idx > aux_idx
    ):
        return subj_idx + 1

    # rule 2: statement with aux
    if aux_idx is not None:
        return aux_idx + 1

    # rule 3: imperative + pronoun object
    obj_idx = get_object_idx(root, doc)
    if root.i == 0 and obj_idx is not None:
        return obj_idx + 1

    # rule 4: plain imperative
    if root.i == 0:
        return root.i + 1

    # rule 5: subject + verb, no aux
    if subj_idx is not None:
        return subj_idx + 1

    # fallback
    return root.i


def insert_modifier(prompt, modifier, doc, root, aux_idx, subj_idx):

    position = random.choices(
        ["start", "anchor", "end"],
        weights=[0.20, 0.55, 0.25]
    )[0]

    # start: "Modifier, tell me..."
    if position == "start":
        mutated = f"{modifier.capitalize()}, {prompt}"
        return fix_punctuation(clean_spacing(mutated))

    # end: "Tell me... modifier."
    if position == "end":
        stripped = prompt.rstrip(".?!")
        punct = prompt[len(stripped):]
        mutated = f"{stripped} {modifier}{punct}"
        return fix_punctuation(clean_spacing(mutated))

    # anchor: mid-sentence grammatically correct position
    if root is None:
        # no root — fall back to start
        mutated = f"{modifier.capitalize()}, {prompt}"
        return fix_punctuation(clean_spacing(mutated))

    anchor_idx = get_anchor_idx(doc, root, aux_idx, subj_idx)

    before = "".join(t.text_with_ws for t in doc[:anchor_idx])
    after = "".join(t.text_with_ws for t in doc[anchor_idx:])

    mutated = f"{before}{modifier} {after}"
    mutated = clean_spacing(mutated)
    mutated = fix_punctuation(mutated)

    return mutated


def get_noun_target_idx(root, doc):

   
    if root is None:
        return None

    candidate = None

    for child in root.children:
        if (
            child.dep_ in ["dobj", "attr"]
            and child.pos_ in ["NOUN", "PROPN"]
        ):
            candidate = child
            break

    if candidate is None:
        for token in doc[root.i + 1:]:
            if token.pos_ in ["NOUN", "PROPN"]:
                candidate = token
                break

    if candidate is None:
        return None

    if any(child.dep_ == "amod" for child in candidate.children):
        return None

    return candidate.i


def get_adjective_anchor_idx(doc, noun_idx):
 
  
    noun = doc[noun_idx]

    for child in noun.children:
        if (
            child.dep_ in ["det", "poss"]
            and child.i == noun_idx - 1
        ):
            return child.i + 1

    return noun_idx


def insert_adjective(prompt, adjective, doc, anchor_idx):
    
    
    before = "".join(t.text_with_ws for t in doc[:anchor_idx])
    after = "".join(t.text_with_ws for t in doc[anchor_idx:])

    mutated = f"{before}{adjective} {after}"
    mutated = clean_spacing(mutated)
    mutated = fix_punctuation(mutated)

    return mutated


#  Main mutation function

def modifier_mutation(prompt, num_variants=5):

    if (
        prompt is None
        or not isinstance(prompt, str)
    ):
        return []

    prompt = prompt.strip()

    # too short to insert modifier meaningfully
    if not prompt or len(prompt.split()) < 3:
        return []

    doc = nlp(prompt)

    root = None
    aux_idx = None
    subj_idx = None

    for token in doc:
        if (
            token.dep_ == "ROOT"
            and token.pos_ in ["VERB", "AUX"]
        ):
            root = token
        if token.dep_ in ["aux", "auxpass"]:
            aux_idx = token.i
        if token.dep_ in ["nsubj", "nsubjpass"]:
            subj_idx = token.i

    if root is not None:
        if any(
            child.dep_ == "advmod"
            for child in root.children
        ):
            return []


    if (
        len(doc) > 1
        and doc[1].text == ","
        and doc[0].text.lower() in MODIFIERS_LOWER
    ):
        return []

    noun_idx = get_noun_target_idx(root, doc)
    adjective_anchor_idx = (
        get_adjective_anchor_idx(doc, noun_idx)
        if noun_idx is not None
        else None
    )

    # combined adverb+adjective pool, no category weighting
    candidates = build_candidate_pool(noun_idx)

    # plain random sample
    selected = sample_candidates(candidates, num_variants)

    # fallback if pool too small: adverbs only, same as before
    if not selected:
        selected = [
            ("adverb", modifier)
            for modifier in random.sample(
                MODIFIERS,
                min(num_variants, len(MODIFIERS))
            )
        ]

    mutations = []
    seen = set()

    for i, (pos, modifier) in enumerate(selected):

        try:
            if pos == "adjective":
                mutated = insert_adjective(
                    prompt, modifier,
                    doc, adjective_anchor_idx
                )
            else:
                mutated = insert_modifier(
                    prompt, modifier,
                    doc, root,
                    aux_idx, subj_idx
                )

            if (
                mutated is None
                or not isinstance(mutated, str)
            ):
                continue

            mutated = mutated.strip()

            if not mutated:
                continue

            # reject duplicates or unchanged
            if (
                mutated in seen
                or mutated.lower() == prompt.lower()
            ):
                continue

            seen.add(mutated)

            mutations.append({
                "mutation_id":          i + 1,
                "modifier":             modifier,
                "pos":                  pos,
                "mutated_prompt":       mutated
            })

        except Exception as e:
            print("\nmodifier error:", e)
            continue

    return mutations


if __name__ == "__main__":
    random.seed(42)

    test_prompts = [
        "Would you show me how to do this?",
        "Show me the process.",
        "Explain your plan.",
        "Tell me how to do this.",
    ]

    for prompt in test_prompts:
        print("\n" + "=" * 60)
        print("ORIGINAL:", prompt)

        results = modifier_mutation(prompt, num_variants=5)

        for r in results:
            print(f"\n  [{r['pos']}: {r['modifier']}]")
            print(f"  {r['mutated_prompt']}")
