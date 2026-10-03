import json
import re
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[2]
CARDS_DIR = ROOT / "RQ1/data/system_cards"
OUT_DIR = ROOT / "RQ1/annotations/toc_domains"

# source_doc key (as in claim_extractor.py) -> (pdf file, display name)
CARDS = {
    "claude": ("claude-opus-4-8.pdf", "Claude Opus 4.8 System Card"),
    "gpt5": ("gpt-5-5.pdf", "GPT-5.5 System Card"),
}

# Neutral definitions shown to annotators; deliberately avoid the cards' own
# section/evaluation titles so the definition doesn't point at the answer.
DOMAINS = {
    "mental_health_self_harm": (
        "Mental health and self-harm",
        "Suicide, self-harm, disordered eating, or other mental-health crises, "
        "including how the model responds to users who may be at risk.",
    ),
    "harmful_violent_content": (
        "Harmful and violent content",
        "Requests for, or generation of, harmful, dangerous, or violent content "
        "(e.g. violence, weapons, illegal or dangerous activities), including how "
        "the model refuses or otherwise safely handles such requests.",
    ),
    "fairness_bias": (
        "Fairness and bias",
        "Bias, stereotyping, discrimination, or unequal treatment based on "
        "demographic, social, or political attributes, including even-handedness "
        "across groups or viewpoints.",
    ),
}


def clean_title(title):
    """Strip the zero-width spaces and stray whitespace the PDF outlines carry."""
    return re.sub(r"\s+", " ", title.replace("​", " ")).strip()


def toc_options(source_doc, pdf_name):
    """One checklist option per titled ToC entry, in document order.

    The choice value is what gets stored, so it leads with a stable id
    ("claude-t083: ...") to keep results parseable. A '· ' per nesting level
    keeps the hierarchy visible; the page number helps locate the section.
    """
    toc = fitz.open(CARDS_DIR / pdf_name).get_toc()
    entries = [(level, clean_title(t), page) for level, t, page in toc]
    entries = [e for e in entries if e[1]]  # drop blank outline entries
    return [
        {"value": f"{source_doc}-t{i:03d}: {'· ' * (level - 1)}{title} (p{page})"}
        for i, (level, title, page) in enumerate(entries, 1)
    ]


def main():
    options = {doc: toc_options(doc, pdf) for doc, (pdf, _) in CARDS.items()}

    tasks = []
    for domain, (domain_name, definition) in DOMAINS.items():
        for doc, (_, card_name) in CARDS.items():
            tasks.append({
                "data": {
                    "domain": domain,
                    "domain_name": domain_name,
                    "domain_definition": definition,
                    "source_doc": doc,
                    "card_name": card_name,
                    "num_sections": len(options[doc]),
                    "toc_options": options[doc],
                }
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "tasks.json", "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(tasks)} tasks ({len(DOMAINS)} domains x {len(CARDS)} cards) "
          f"to {OUT_DIR / 'tasks.json'}")
    for doc, opts in options.items():
        print(f"  {doc}: {len(opts)} ToC sections")


if __name__ == "__main__":
    main()
