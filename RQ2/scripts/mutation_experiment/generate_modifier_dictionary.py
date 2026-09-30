# Generates RQ2/data/dictionary/modifier_dictionary.json from WordNet directly
import json
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

MODIFIER_DICTIONARY_PATH = "RQ2/data/dictionary/modifier_dictionary.json"

MIN_ZIPF_FREQUENCY_ADVERB = 3.5
MIN_ZIPF_FREQUENCY_ADJECTIVE = 4.5


def all_wordnet_adverbs():
    words = set()
    for synset in wn.all_synsets(pos=wn.ADV):
        for lemma in synset.lemmas():
            name = lemma.name()
            if (
                name.isalpha()
                and zipf_frequency(name, "en") >= MIN_ZIPF_FREQUENCY_ADVERB
            ):
                words.add(name.lower())
    return sorted(words)


def all_wordnet_adjectives():
    words = set()
    for pos in (wn.ADJ, wn.ADJ_SAT):
        for synset in wn.all_synsets(pos=pos):
            for lemma in synset.lemmas():
                name = lemma.name()
                if (
                    name.isalpha()
                    and zipf_frequency(name, "en") >= MIN_ZIPF_FREQUENCY_ADJECTIVE
                ):
                    words.add(name.lower())
    return sorted(words)


if __name__ == "__main__":
    adverbs = all_wordnet_adverbs()
    adjectives = all_wordnet_adjectives()

    modifier_dictionary = {
        "adverb": adverbs,
        "adjective": adjectives,
    }

    with open(MODIFIER_DICTIONARY_PATH, "w", encoding="utf-8") as f:
        json.dump(modifier_dictionary, f, indent=2, ensure_ascii=False)

    print(
        f"wrote {len(adverbs)} adverbs and {len(adjectives)} adjectives "
        f"to {MODIFIER_DICTIONARY_PATH}"
    )
