import os
import re
import sys
import pymupdf as fitz
import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


#PART 1 — EXACT SECTION DEFINITIONS  (v4 fallback / baseline reference)
EXACT_SECTIONS = {
    "claude": {
        "mental_health_self_harm": [
            "4.3.1 Suicide and self-harm",
            "4.3.2 Disordered eating",
        ],
        "harmful_violent_content": [
            "4.1 Harmful request evaluations",
            "4.1.1 Single-turn harmful request evaluation results",
            "4.1.2 Single-turn benign request evaluation results",
            "4.1.3 Multi-turn testing results",
            "4.1.4 Harmful request evaluations discussion",
        ],
        "fairness_bias": [
            "4.4.1 Political bias and even-handedness",
            "4.4.2 Bias Benchmark for Question Answering",
            "4.4.3 Election integrity",
        ],
    },
    "gpt5": {
        "mental_health_self_harm": [
            "Dynamic Mental Health Benchmarks with Adversarial User Simulations",
            "Health",
        ],
        "harmful_violent_content": [
            "Disallowed Content",
            "Evaluations with Challenging Prompts",
            "Evaluations with Representative Prompts",
            "From Hard Refusals to Safe-Completions",
        ],
        "fairness_bias": [
            "Fairness and Bias: BBQ Evaluation",
            "Bias Evaluation",
            "First Person Fairness Evaluation",
        ],
    },
}

#Semantic category descriptions (v5)
CATEGORY_DESCRIPTIONS = {
    "mental_health_self_harm": (
        "sections about suicide, self-harm, disordered eating, mental health crises, "
        "safe messaging guidelines, and dynamic mental health "
        "benchmarks or adversarial mental health user simulations"
    ),
    "harmful_violent_content": (
        "sections about harmful content, violent content, dangerous requests, "
        "disallowed content, content moderation, safety evaluations, "
        "evaluations with challenging or representative prompts, hard refusals, "
        "safe completions, and harmful request evaluation results"
    ),
    "fairness_bias": (
        "sections about fairness, bias, discrimination, stereotyping, political bias, "
        "demographic disparities, first-person fairness, BBQ evaluation, "
        "bias benchmark for question answering, and election integrity"
    ),
}

# PART 2 — SPACY FEATURE LEXICONS  (paper Table I)
EXTERNAL_BEHAVIORAL_VERBS = {
    # Safety/refusal actions
    "refuse", "decline", "reject", "block", "filter", "flag",
    "detect", "prevent", "stop", "avoid", "warn", "redirect",
    # Output actions
    "output", "generate", "produce", "provide", "give", "return",
    "respond", "answer", "reply", "complete", "suggest", "recommend",
    # Compliance actions
    "follow", "comply", "adhere", "maintain", "enforce", "apply",
    "handle", "address", "support", "assist", "direct", "point",
    # Observable behavioral verbs used in system cards
    "contain", "include", "indicate", "represent", "reflect",
    "exhibit", "display", "show", "demonstrate", "retain",
    "position", "validate", "invite", "encourage",
}

INTERNAL_STATE_VERBS = {
    "prioritize", "intend", "aim", "design", "train", "learn",
    "understand", "believe", "consider", "ensure", "seek",
    "strive", "attempt", "try", "want", "need", "require",
    "expect", "plan", "commit", "dedicate", "focus", "target",
    "introduce", "develop", "build", "create",
}

AGGREGATE_STAT_VERBS = {
    "achieve", "score", "reach", "attain", "perform", "obtain",
    "record", "measure", "evaluate", "test", "assess", "improve",
    "reduce", "increase", "outperform", "exceed", "surpass",
    "match", "compare", "rate", "rank", "benchmark",
    # Reporting verbs used around benchmark numbers
    "report", "find", "observe", "note", "maintain", "publish",
}

STRONG_MODALS  = {"will", "must", "shall"}
DEONTIC_MODALS = {"should", "ought"}
POSS_MODALS    = {"may", "can", "might", "could"}

UNIVERSAL_QUANTIFIERS = {
    "always", "never", "all", "every", "none", "no",
    "entirely", "completely", "absolutely", "consistently",
}
HEDGED_QUANTIFIERS = {
    "may", "sometimes", "can", "might", "possibly", "often",
    "generally", "typically", "approximately", "around",
    "usually", "largely", "mostly", "tend", "likely", "unlikely",
}

MODEL_REFERENCE_WORDS = {
    "model", "system", "claude", "gpt", "assistant",
    "opus", "sonnet", "haiku", "gpt-5", "gpt-4", "chatgpt", "it",
}

#Noise patterns 
# Sentences matching any of these are discarded before SpaCy processes them.
NOISE_PATTERNS = [
    #page numbers / lone numbers 
    r'^\s*\d+\s*$',                        # lone page number
    r'^\s*\d+\.\d+\s+\d+\.\d+',           # bare number rows (table data)

    #TOC fragment chains
    # starts with a multi-level section number: "4.2 Prompt injection…", "6.2.3.1.1 Overall…"
    r'^\s*\d+(?:\.\d+)+\s+[A-Z]',

    #citations / references
    r'^\s*\[?\d+\]',                       # citation [1]
    r'et al\.',                             # author citation fragments
    r'ibid\.',                              # ibid
    r'arxiv',                               # arXiv refs
    r'doi\.org|doi:',                       # DOI refs

    #URLs / boilerplate
    r'https?://',                           # URLs
    r'©|copyright|all rights reserved',    # copyright

    #headings masquerading as claims
    r'^\s*[A-Z\s]{10,}\s*$',              # ALL-CAPS headings

    # table / figure metadata
    r'^\s*table\s+\d+',                    # "Table 1"
    r'^\s*figure\s+\d+',                   # "Figure 1"
    r'higher is better|lower is better',   # table notes
    r'closer to zero is better',           # table notes
    r'bold.* indicat|indicat.* bold',      # table footnotes
    r'second.best score',                   # table footnotes
    r'does not take into account',          # table footnotes
    r'rates are an average',                # table footnotes
    r'^\s*underlined',                      # table footnotes
    r'margin of error',                     # table footnotes
    r'^\s*[A-Za-z\s]+\(%\)',               # table column headers
    r'as shown in (table|figure)\s+\d+',   # inline table references
    r'(table|figure)\s+\d+\s+(shows?|presents?|summarizes?|reports?)',

    # eval config / methodology disclaimers
    r'^\s*\w+\s+with\s+(thinking|system prompt)',
    r'evaluations are run',
    r'results.*system cards? due to',
    r'single-turn requests posing',
    r'requests posing potential risk',
    r'reflected in the scores below',
    r'subject to some variation',
    r'values may vary slightly',

    #cross-references
    r'\bsee\s+(section|appendix|table|figure)\b',
    r'\bas (described|discussed|noted|shown|defined)\s+(above|below|in\s+section)\b',
    r'for more (information|details|context).{0,40}(see|refer)',
    r'we (note|refer|point) (that|the reader)',

    #acknowledgement / admin sentences 
    r'we (would like to )?(thank|acknowledge)',
    r'this (paper|report|document|work) (is|was) (organized|structured|presented)',
    r'the remainder of this (paper|report|section)',

    #table data rows (numbers/percentages concatenated by PDF extraction)
    # 3+ consecutive percentage values in sequence = table row, not a claim
    r'(?:\d+(?:\.\d+)?\s*%\s*){3,}',
    # 2+ parenthetical score tuples like "34.7 (41.6, 2880) 25.4 (41.4, 4049)"
    r'(?:\d+(?:\.\d+)?\s*\([^)]*\d[^)]*\)\s*){2,}',
    #table data rows (numbers/percentages concatenated by PDF extraction)
    # 3+ consecutive percentage values in sequence = table row, not a claim
    r'(?:\d+(?:\.\d+)?\s*%\s*){3,}',
    # 2+ parenthetical score tuples like "34.7 (41.6, 2880) 25.4 (41.4, 4049)"
    r'(?:\d+(?:\.\d+)?\s*\([^)]*\d[^)]*\)\s*){2,}',

    # ── example prompts embedded in system cards (user input, not model claims)
    r'\bexample prompt\b',
]


#PART 3 — PDF LOADING AND TOC EXTRACTION
def load_pdf(pdf_path: str) -> fitz.Document:
    print(f"\n{'='*60}")
    print(f"  Loading: {pdf_path}")
    print(f"{'='*60}")
    doc = fitz.open(pdf_path)
    print(f"  Pages: {len(doc)}")
    return doc


def extract_toc(doc: fitz.Document) -> list:
    toc = doc.get_toc()
    if toc:
        print(f"  TOC: {len(toc)} entries (embedded)")
        return toc
    print("  No embedded TOC — scanning visually...")
    return parse_toc_visually(doc)


def parse_toc_visually(doc: fitz.Document, max_pages: int = 10) -> list:
    r
    entries = []
    pattern = re.compile(
        r'^(\d+(?:\.\d+)*)\s+([A-Za-z][^\n]{4,70?}?)'
        r'\s*[.\s]{2,}\s*(\d{1,3})\s*$',
        re.MULTILINE
    )
    for idx in range(min(max_pages, len(doc))):
        for m in pattern.finditer(doc[idx].get_text()):
            num, title, page = m.groups()
            entries.append([len(num.split('.')), title.strip(), int(page)])
    print(f"  Visual TOC: {len(entries)} entries")
    return entries

# PART 4 — EXACT SECTION MAPPING (v4 core change)
def build_section_map(toc: list, source_doc: str,
                      doc: fitz.Document) -> dict:
    
    section_map  = {cat: [] for cat in EXACT_SECTIONS[source_doc]}
    seen_ranges  = {cat: set() for cat in EXACT_SECTIONS[source_doc]}

    # Build lookup: lowercase title → (category, original title)
    lookup = {}
    for cat, titles in EXACT_SECTIONS[source_doc].items():
        for t in titles:
            lookup[t.lower().strip()] = cat

    for i, (level, title, page) in enumerate(toc):
        title_clean = title.lower().strip()
        if title_clean not in lookup:
            continue

        category = lookup[title_clean]
        end_page = toc[i + 1][2] if i + 1 < len(toc) else page + 10

        # Deduplicate on page range
        rng = (page, end_page)
        if rng in seen_ranges[category]:
            continue
        seen_ranges[category].add(rng)

        # Classify section type
        t_lower = title.lower()
        if any(w in t_lower for w in ["result", "evaluation",
                                       "benchmark", "score"]):
            sec_type = "evaluation"
        elif any(w in t_lower for w in ["risk", "limitation",
                                         "concern", "caveat"]):
            sec_type = "bullet"
        else:
            sec_type = "prose"

        section_map[category].append({
            "title":        title,
            "start_page":   page,
            "end_page":     end_page,
            "section_type": sec_type,
        })

    # Report matches
    print(f"\n  Section map [{source_doc.upper()}]:")
    for cat, secs in section_map.items():
        status = f"{len(secs)} sections" if secs else "NONE MATCHED"
        print(f"    {cat}: {status}")
        for s in secs:
            print(f"      └─ '{s['title']}' "
                  f"pp.{s['start_page']}–{s['end_page']} "
                  f"[{s['section_type']}]")

    return section_map


## PART 4B — SEMANTIC + HIERARCHICAL SECTION MAPPING  (v5 default)
def build_section_map_semantic(toc: list, source_doc: str,
                                _doc: fitz.Document,
                                threshold_l1: float = 0.45,
                                threshold_l2: float = 0.33) -> dict:
    try:
        from sentence_transformers import SentenceTransformer, util as st_util
    except ImportError:
        print("  WARNING: sentence-transformers not installed. "
              "  Run: pip install sentence-transformers")
        return {}

    print("  Loading all-MiniLM-L6-v2 for semantic section matching...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    cats      = list(CATEGORY_DESCRIPTIONS.keys())
    desc_embs = embedder.encode(
        [CATEGORY_DESCRIPTIONS[c] for c in cats],
        convert_to_tensor=True, show_progress_bar=False,
    )
    titles     = [title for _, title, _ in toc]
    title_embs = embedder.encode(
        titles,
        convert_to_tensor=True, show_progress_bar=False,
    )

    # sims[i][j] = cosine similarity of toc[i] vs cats[j] description
    sims = st_util.cos_sim(title_embs, desc_embs)   # (n_titles, n_cats)

    # Collect all (toc_index, category) parent hits above their level threshold
    parent_hits = []
    print(f"\n  Semantic hits (threshold_l1={threshold_l1}, threshold_l2={threshold_l2}):")
    for i in range(len(toc)):
        level = toc[i][0]
        thr   = threshold_l1 if level == 1 else threshold_l2
        for j, cat in enumerate(cats):
            score = sims[i][j].item()
            if score >= thr:
                parent_hits.append((i, cat, score))
                print(f"    [{cat}] '{toc[i][1]}'  sim={score:.3f}  (L{level})")

    section_map  = {cat: [] for cat in cats}
    seen_ranges  = {cat: set() for cat in cats}

    def _add_entry(idx: int, category: str):
        _, title, page = toc[idx]
        end_page = toc[idx + 1][2] if idx + 1 < len(toc) else page + 10
        rng = (page, end_page)
        if rng in seen_ranges[category]:
            return
        seen_ranges[category].add(rng)
        t_lower = title.lower()
        if any(w in t_lower for w in ["result", "evaluation",
                                       "benchmark", "score"]):
            sec_type = "evaluation"
        elif any(w in t_lower for w in ["risk", "limitation",
                                         "concern", "caveat"]):
            sec_type = "bullet"
        else:
            sec_type = "prose"
        section_map[category].append({
            "title":        title,
            "start_page":   page,
            "end_page":     end_page,
            "section_type": sec_type,
        })

    for parent_idx, category, _score in parent_hits:
        parent_level = toc[parent_idx][0]
        # Add parent itself
        _add_entry(parent_idx, category)
        # Walk all deeper-nested children until level resets
        for k in range(parent_idx + 1, len(toc)):
            if toc[k][0] <= parent_level:
                break
            _add_entry(k, category)

    # Report
    print(f"\n  Section map [{source_doc.upper()}] (semantic):")
    for cat, secs in section_map.items():
        status = f"{len(secs)} sections" if secs else "NONE MATCHED"
        print(f"    {cat}: {status}")
        for s in secs:
            print(f"      └─ '{s['title']}' "
                  f"pp.{s['start_page']}–{s['end_page']} "
                  f"[{s['section_type']}]")

    return section_map



# PART 5 — TEXT EXTRACTION STRATEGIES

def get_section_text(doc: fitz.Document, section: dict) -> str:
    raw = ""
    start = max(0, section["start_page"] - 1)   # 1-based → 0-based
    # end_page is the start page of the NEXT section. Exclude it so content
    # from the next section doesn't bleed into this one. The floor guard
    # ensures we always read at least the section's own start page.
    end = max(start + 1, min(len(doc), section["end_page"] - 1))
    for idx in range(start, end):
        raw += doc[idx].get_text()

    # Normalise whitespace: newlines → spaces, collapse runs
    clean = re.sub(r'\n+', ' ', raw)
    clean = re.sub(r' {2,}', ' ', clean)
    return clean.strip()


def is_noise(text: str) -> bool:
    for pat in NOISE_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    # TOC chain: "3.1 Disallowed Content 3.1.1 Evaluations…" — 2+ section-number+title tokens
    if len(re.findall(r'\b\d+(?:\.\d+)+\s+[A-Z]', text)) >= 2:
        return True
    return False


def extract_bullet_claims(text: str) -> list:
    claims  = []
    pattern = re.compile(
        r'(?:^|(?<=\s))(?:[•\-\*–]|\d+[\.\)])\s+(.{30,}?)(?='
        r'\s*(?:[•\-\*–]|\d+[\.\)])\s|\Z)',
        re.DOTALL
    )
    for m in pattern.finditer(text):
        claim = re.sub(r'\s+', ' ', m.group(1).strip())
        if claim and not is_noise(claim):
            claims.append(claim)
    return claims


def extract_benchmark_claims(text: str) -> list:
    claims   = []
    numeric  = re.compile(r'\d+\.?\d*\s*%|\b0\.\d+\b|\b\d{2,}\b')
    perf_pat = re.compile(
        r'\b(achieves?|scores?|performs?|reaches?|reduces?|improves?|'
        r'outperforms?|demonstrates?|reports?|records?|attains?|'
        r'exceeds?|surpasses?|shows?|maintains?|lowers?|finds?)\b',
        re.IGNORECASE
    )
    for sent in re.split(r'(?<=[.!?])\s+', text):
        sent = sent.strip()
        if len(sent) < 25 or is_noise(sent):
            continue
        if numeric.search(sent) and perf_pat.search(sent):
            claims.append(re.sub(r'\s+', ' ', sent))
    return claims


# PART 6 — SPACY 3-STEP PIPELINE  (paper Section V-C1)
def step1_segment_sentences(text: str, nlp) -> list:
    return list(nlp(text).sents)


def step2_split_compounds(sentences: list) -> list:
    results = []
    for sent in sentences:
        splits = [
            tok.i for tok in sent
            if tok.dep_ == "conj" and any(
                c.dep_ in ("nsubj", "nsubjpass") for c in tok.children
            )
        ]
        if not splits:
            results.append(sent.text.strip())
        else:
            cur = sent.start
            for sp in splits:
                chunk = sent.doc[cur:sp].text.strip()
                if chunk:
                    results.append(chunk)
                cur = sp
            tail = sent.doc[cur:sent.end].text.strip()
            if tail:
                results.append(tail)
    return results


def step3_filter(sentences: list, nlp) -> list:
    valid = []
    for text in sentences:
        text = text.strip()
        if len(text) < 25:
            continue
        # Real sentences always start with a capital letter. A lowercase start
        # means this is a sentence fragment extracted mid-clause.
        if text and text[0].islower():
            continue
        if is_noise(text):
            continue
        has_verb = any(
            t.pos_ in ("VERB", "AUX") or t.tag_ == "MD"
            for t in nlp(text)
        )
        if not has_verb:
            continue
        if text.endswith('?'):
            continue
        valid.append(text)
    return valid


def run_spacy_pipeline(text: str, nlp) -> list:
    sents    = step1_segment_sentences(text, nlp)
    atomic   = step2_split_compounds(sents)
    filtered = step3_filter(atomic, nlp)
    return filtered


# PART 7 — FEATURE EXTRACTION  (paper Table I, 8 features, 4 groups)
def extract_features(claim_text: str, nlp) -> dict:
    doc    = nlp(claim_text)
    tokens = list(doc)
    lower  = {t.lower_ for t in tokens}

    # Group 2
    contains_numeric = (
        any(t.like_num for t in tokens) or
        bool(re.search(r'\d+\.?\d*\s*%|\b0\.\d+\b', claim_text))
    )
    has_universal = bool(lower & UNIVERSAL_QUANTIFIERS)
    has_hedged    = bool(lower & HEDGED_QUANTIFIERS)

    # Group 3
    root       = next((t for t in tokens if t.dep_ == "ROOT"), None)
    root_lemma = root.lemma_.lower() if root else ""

    if   root_lemma in EXTERNAL_BEHAVIORAL_VERBS: verb_type = "external_behavioral"
    elif root_lemma in INTERNAL_STATE_VERBS:       verb_type = "internal_state"
    elif root_lemma in AGGREGATE_STAT_VERBS:       verb_type = "aggregate_stat"
    else:                                           verb_type = "other"

    modals = {t.lower_ for t in tokens if t.tag_ == "MD"}
    if   modals & STRONG_MODALS:  modal_type = "strong"
    elif modals & DEONTIC_MODALS: modal_type = "deontic"
    elif modals & POSS_MODALS:    modal_type = "possible"
    else:                          modal_type = "none"

    # Group 4
    model_is_subject = False
    if root:
        for subj in (t for t in root.lefts
                     if t.dep_ in ("nsubj", "nsubjpass")):
            if {t.lower_ for t in subj.subtree} & MODEL_REFERENCE_WORDS:
                model_is_subject = True
                break

    tense = "unknown"
    if root:
        m = root.morph.get("Tense")
        if m:
            tense = "present" if m[0].lower() in ("pres", "present") else "past"
        else:
            aux = {t.lower_ for t in tokens if t.pos_ == "AUX"}
            if aux & {"is", "are", "does", "do"}:  tense = "present"
            elif aux & {"was", "were", "did"}:      tense = "past"

    return {
        "claim_text":               claim_text,
        "contains_numeric":         contains_numeric,
        "has_universal_quantifier": has_universal,
        "has_hedged_quantifier":    has_hedged,
        "verb_type":                verb_type,
        "modal_verb_type":          modal_type,
        "model_is_subject":         model_is_subject,
        "tense":                    tense,
    }


def build_tf_matrix(df: pd.DataFrame) -> pd.DataFrame:
    vec = CountVectorizer(min_df=2, max_features=300,
                          stop_words='english')
    mat = vec.fit_transform(df["claim_text"])
    tf  = pd.DataFrame(
        mat.toarray(),
        columns=[f"tf_{w}" for w in vec.get_feature_names_out()]
    )
    return pd.concat([df.reset_index(drop=True), tf], axis=1)


# PART 8 — ORCHESTRATION
def extract_from_section(doc, section, category,
                          source_doc, nlp) -> list:
    text     = get_section_text(doc, section)
    sec_type = section["section_type"]

    if   sec_type == "evaluation": texts = extract_benchmark_claims(text)
    elif sec_type == "bullet":     texts = extract_bullet_claims(text)
    else:                           texts = run_spacy_pipeline(text, nlp)

    records = []
    for t in texts:
        t = re.sub(r'\s+', ' ', t.strip())
        if not t or len(t) < 25:
            continue
        feat = extract_features(t, nlp)
        feat.update({
            "category":     category,
            "section":      section["title"],
            "source_doc":   source_doc,
            "section_type": sec_type,
            "label":        None,
        })
        records.append(feat)
    return records


def extract_all_claims(pdf_path: str, source_doc: str,
                        use_semantic: bool = True,
                        sem_threshold_l1: float = 0.45,
                        sem_threshold_l2: float = 0.33) -> pd.DataFrame:
    nlp = spacy.load("en_core_web_trf")
    doc = load_pdf(pdf_path)
    toc = extract_toc(doc)
    if not toc:
        print("  ERROR: Could not parse TOC.")
        return pd.DataFrame()

    if use_semantic:
        section_map = build_section_map_semantic(
            toc, source_doc, doc,
            threshold_l1=sem_threshold_l1,
            threshold_l2=sem_threshold_l2)
        if not any(section_map.values()):
            print("  Semantic matching found no sections — "
                  "falling back to v4 exact-title matching.")
            section_map = build_section_map(toc, source_doc, doc)
    else:
        section_map = build_section_map(toc, source_doc, doc)

    all_records = []
    for category, sections in section_map.items():
        print(f"\n  [{category}]")
        for section in sections:
            recs = extract_from_section(
                doc, section, category, source_doc, nlp)
            print(f"    '{section['title']}' → {len(recs)} claims")
            all_records.extend(recs)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    before = len(df)
    df = df.drop_duplicates(subset="claim_text", keep="first")
    if before - len(df):
        print(f"\n  Removed {before - len(df)} duplicates")

    df = build_tf_matrix(df)
    return df


# PART 9 — SUMMARY REPORT
def print_summary(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  EXTRACTION SUMMARY")
    print("="*60)
    print(f"  Total claims : {len(df)}")
    print()

    print("  Source breakdown:")
    for src, n in df["source_doc"].value_counts().items():
        print(f"    {src:8s} → {n}")
    print()

    print("  Category × source:")
    cross = df.groupby(["source_doc", "category"]).size().reset_index(
        name="n")
    for _, r in cross.iterrows():
        flag = "✓" if r["n"] > 5 else "⚠ LOW"
        print(f"    {r['source_doc']:8s} | "
              f"{r['category']:35s} | {r['n']:3d}  {flag}")
    print()

    print("  Features:")
    print(f"    contains_numeric   : {df['contains_numeric'].sum()} / {len(df)}")
    print(f"    model_is_subject   : {df['model_is_subject'].sum()} / {len(df)}")
    print("    verb_type:")
    for v, c in df["verb_type"].value_counts().items():
        print(f"      {v:25s} : {c}")
    print("    tense:")
    for t, c in df["tense"].value_counts().items():
        print(f"      {t:10s} : {c}")
    print()

    print("  Sample claims (2 per category per source):")
    for cat in df["category"].unique():
        print(f"\n  [{cat}]")
        for src in df["source_doc"].unique():
            sub = df[(df["category"]==cat) & (df["source_doc"]==src)]
            if sub.empty:
                print(f"    [{src}] — no claims")
                continue
            for _, row in sub.head(2).iterrows():
                txt = row["claim_text"]
                print(f"    [{src}] {txt[:105]}"
                      f"{'...' if len(txt)>105 else ''}")

    print("\n" + "="*60)
    print()
    print("  NOTE: Section matching in v5 uses semantic similarity")
    print("  (all-MiniLM-L6-v2) + hierarchical TOC walk. Generalizes")
    print("  to new PDFs without manual title updates. EXACT_SECTIONS")
    print("  is kept as a fallback reference only.")
    print("="*60)


# PART 10 — ENTRY POINT
if __name__ == "__main__":

    # RQ1/scripts/claim_extractor.py -> RQ1/, so ../data and ../outputs
    _rq1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CLAUDE_PDF = sys.argv[1] if len(sys.argv) > 1 \
        else os.path.join(_rq1, "data", "system_cards", "claude-opus-4-8.pdf")
    GPT5_PDF   = sys.argv[2] if len(sys.argv) > 2 \
        else os.path.join(_rq1, "data", "system_cards", "gpt-5-5.pdf")
    OUTPUT_DIR = os.path.join(_rq1, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    claude_df = extract_all_claims(CLAUDE_PDF, source_doc="claude")
    gpt5_df   = extract_all_claims(GPT5_PDF,   source_doc="gpt5")
    combined  = pd.concat([claude_df, gpt5_df], ignore_index=True)

    print_summary(combined)

    # OUTPUT 1 — Full feature matrix → fed into RIPPER after annotation
    features_path = os.path.join(OUTPUT_DIR, "claims_features.csv")
    combined.to_csv(features_path, index=False)
    print(f"  Saved: {features_path}  "
          f"({len(combined)} rows, {len(combined.columns)} cols)")

    # OUTPUT 2 — Annotation sheet → sent to human annotators
    # 'label' column is blank — annotators assign one of:
    #   claim_testable | claim_not_testable | not_claim
    anno_cols = [
        "claim_text", "category", "section", "source_doc",
        "contains_numeric", "has_universal_quantifier",
        "has_hedged_quantifier", "verb_type", "modal_verb_type",
        "model_is_subject", "tense", "label"
    ]
    anno = combined[[c for c in anno_cols if c in combined.columns]]
    annotation_path = os.path.join(OUTPUT_DIR, "claims_for_annotation.csv")
    anno.to_csv(annotation_path, index=False)
    print(f"  Saved: {annotation_path}  ({len(anno)} rows)")
    print()
    print("  Next: annotators fill in 'label' column with:")
    print("  claim_testable | claim_not_testable | not_claim")