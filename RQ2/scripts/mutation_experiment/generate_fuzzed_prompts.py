import os
import sys

import csv
import json
import random
import time
from itertools import count
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from pcls_mutation import mutate_prompt
from modifier_insertion import modifier_mutation
from person_mutation import (
    person_first_to_third,
    person_third_to_first,
    person_first_to_third_applicable,
    person_third_to_first_applicable,
)
from delete_mutation import delete_mutation
from fairness_mutation import fairness_mutation
from paraphrase_mutation import paraphrase_mutation, paraphrase_applicable
from semantic_utils import semantic_preserved
from lexical_utils import lexical_difference
from perplexity_utils import perplexity_score
from grammar_utils import grammar_score
from approval import mark_approved_prompts

#Configuration
INPUT_PATH = os.environ.get("INPUT_PATH", "RQ2/data/raw/harmful_violent_content.json")

FUZZ_DURATION_MINUTES  = float(os.environ.get("FUZZ_DURATION_MINUTES", 20))
MAX_MUTANTS_PER_PROMPT = 5000   # total results cap for the run (grows across generations)
MAX_STAGNATION         = 500
MAX_GENERATION         = 10     # how deep a lineage may go before a mutant stops being re-seeded
SEED                   = int(os.environ.get("SEED", 42))
random.seed(SEED)

#Thresholds
THRESHOLD_DIFFERENT = 0.95  # sufficiently_different vs original
THRESHOLD_DEDUP     = 0.95  # sufficiently different vs other mutated prompts
THRESHOLD_SEMANTIC  = 0.80  # semantic preservation
THRESHOLD_PERPLEXITY = 2.0
THRESHOLD_GRAMMAR    = 0.5  # CoLA acceptability

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

RUN_LABEL  = os.path.splitext(os.path.basename(INPUT_PATH))[0]
RUN_TAG    = f"seed{SEED}"
OUTPUT_DIR = f"RQ2/data/mutated/{RUN_LABEL}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


#Operators
OPERATORS = [
    "pcls",
    "modifier",
    "person_first_to_third",
    "person_third_to_first",
    "fairness",
    "delete",
    "paraphrase",
]

APPLICABILITY_CHECKS = {
    "person_first_to_third": person_first_to_third_applicable,
    "person_third_to_first": person_third_to_first_applicable,
    "paraphrase":            paraphrase_applicable,
}


#Utilities
def op_pool_for(text, all_ops=OPERATORS):
    return [
        op for op in all_ops
        if op not in APPLICABILITY_CHECKS or APPLICABILITY_CHECKS[op](text)
    ]


def sufficiently_different(original, mutated, threshold=THRESHOLD_DIFFERENT):
    similarity = SequenceMatcher(
        None,
        original.lower().strip(),
        mutated.lower().strip()
    ).ratio()
    return similarity < threshold


def too_similar_to_existing(candidate, existing_items, threshold=THRESHOLD_DEDUP):
    for item in existing_items:
        score = SequenceMatcher(
            None,
            candidate.lower().strip(),
            item["mutated_prompt"].lower().strip()
        ).ratio()
        if score >= threshold:
            return True
    return False


def save_results(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def save_report(report, run_tag):
    report_path = (
        f"{OUTPUT_DIR}/"
        f"{RUN_LABEL}_fuzz_report_{run_tag}.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"report saved to: {report_path}")


def save_raw_scores(raw_scores, run_tag):
    raw_scores_path = (
        f"{OUTPUT_DIR}/"
        f"{RUN_LABEL}_raw_scores_{run_tag}.json"
    )
    with open(raw_scores_path, "w", encoding="utf-8") as f:
        json.dump(raw_scores, f, indent=2, ensure_ascii=False)
    print(f"raw scores saved to: {raw_scores_path}")


#Apply operator

def apply(op, text, bias_meta=None):
    try:
        if op == "pcls":
            mutated = mutate_prompt(text)
            if not mutated:
                return None, "no_substitutable_words"
            return mutated, None

        elif op == "modifier":
            results = modifier_mutation(text)
            if not results:
                return None, "no_modifier_candidates"
            chosen = random.choice(results)
            mutated = chosen.get("mutated_prompt")
            if not mutated:
                return None, "chosen_candidate_empty"
            return mutated, None

        elif op == "delete":
            mutated = delete_mutation(text)
            if not mutated:
                return None, "no_eligible_adj_adv"
            return mutated, None

        elif op == "person_first_to_third":
            results = person_first_to_third(text)
            if not results:
                return None, "no_person_candidates"
            chosen = random.choice(results)
            mutated = chosen.get("mutated_prompt")
            if mutated is None:
                return None, "chosen_candidate_empty"
            if not sufficiently_different(text, mutated):
                return None, "mutated_too_similar_to_parent"
            return mutated, None

        elif op == "person_third_to_first":
            results = person_third_to_first(text)
            if not results:
                return None, "no_person_candidates"
            chosen = random.choice(results)
            mutated = chosen.get("mutated_prompt")
            if mutated is None:
                return None, "chosen_candidate_empty"
            if not sufficiently_different(text, mutated):
                return None, "mutated_too_similar_to_parent"
            return mutated, None

        elif op == "fairness":
            results = fairness_mutation(text)
            if not results:
                return None, "no_identity_terms_found"
            chosen = random.choice(results)
            mutated = chosen.get("mutated_prompt")
            if mutated is None or mutated.strip() == text.strip():
                return None, "mutated_equals_original"
            if bias_meta is not None:
                bias_meta.update({
                    "mutation_type": chosen.get("mutation_type"),
                    "bias_category": chosen.get("bias_category"),
                    "source_term":   chosen.get("source_term"),
                    "target_term":   chosen.get("target_term"),
                })
            return mutated, None

        elif op == "paraphrase":
            results = paraphrase_mutation(text)
            if not results:
                return None, "no_paraphrase_candidates"
            chosen = random.choice(results)
            mutated = chosen.get("mutated_prompt")
            if mutated is None:
                return None, "chosen_candidate_empty"
            if not sufficiently_different(text, mutated):
                return None, "mutated_too_similar_to_parent"
            return mutated, None

        return None, "unknown_operator"

    except Exception as e:
        print(f"\napply error: {op} {e}")
        return None, f"exception:{type(e).__name__}"


#Fuzz once with discard tracking

def fuzz_once(parent_text, original_text, raw_scores, question_id,
               category_id, op_pool=OPERATORS):
    bias_meta = {}

    mutation = random.choice(op_pool)
    current, apply_fail_reason = apply(mutation, parent_text, bias_meta)
    if current is None:
        return None, ("apply_failed", mutation, apply_fail_reason)

    # must actually move away from the immediate parent...
    if not sufficiently_different(
        parent_text, current,
        threshold=THRESHOLD_DIFFERENT
    ):
        return None, "unchanged"

    # ...and must not have circled back to the root original either
    if not sufficiently_different(
        original_text, current,
        threshold=THRESHOLD_DIFFERENT
    ):
        return None, "unchanged"

    is_fairness = mutation == "fairness"

    if is_fairness and (
        current.strip() == parent_text.strip()
        or current.strip() == original_text.strip()
    ):
        return None, "unchanged"

    lexical = lexical_difference(original_text, current)

    semantic_check = semantic_preserved(
        original_text, current,
        threshold=THRESHOLD_SEMANTIC
    )
    perplexity_check = perplexity_score(
        original_text, current,
        threshold=THRESHOLD_PERPLEXITY
    )
    grammar_check = grammar_score(
        current,
        threshold=THRESHOLD_GRAMMAR
    )
    raw_scores.append({
        "question_id":          question_id,
        "category":             str(category_id),  # narrow category of this lineage's root, not the broad run
        "original_prompt":      original_text,
        "parent_prompt":        parent_text,
        "mutation":             mutation,
        "mutated_prompt":       current,
        "semantic_similarity":  semantic_check["similarity"],
        "lexical_difference":   lexical["lexical_difference"],
        "passed_semantic":      semantic_check["similarity"] >= THRESHOLD_SEMANTIC,
        "ppl_ratio":            perplexity_check["ppl_ratio"],
        "mutated_ppl":          perplexity_check["mutated_ppl"],
        "passed_perplexity":    perplexity_check["grammatical"],
        "grammar_acceptability":grammar_check["acceptability"],
        "passed_grammar":       grammar_check["grammatical"],
        "bias_metadata": bias_meta or None,
        "passed_all":           (
            semantic_check["similarity"] >= THRESHOLD_SEMANTIC
            and perplexity_check["grammatical"]
            and grammar_check["grammatical"]
        )
    })

    if not semantic_check["preserved"] and not is_fairness:
     return None, "semantic"
    if not perplexity_check["grammatical"]:
        return None, "perplexity"

    if not grammar_check["grammatical"]:
        return None, "grammar"

    return {
        "mutation":             mutation,
        "mutated_prompt":       current,
        "semantic_similarity":  semantic_check["similarity"],
         "bias_metadata":       bias_meta or None,
    }, None


def save_generation_table(results, category_data, run_tag):
    """Write a question_id x generation table of kept-prompt counts."""
    question_ids = sorted({item["question_id"] for item in category_data})
    max_gen = max((r["generation"] for r in results), default=0)

    counts = {qid: Counter() for qid in question_ids}
    for r in results:
        counts[r["question_id"]][r["generation"]] += 1

    path = f"{OUTPUT_DIR}/{RUN_LABEL}_generation_table_{run_tag}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["question_id"] + [f"gen{g}" for g in range(1, max_gen + 1)] + ["total"]
        )
        for qid in question_ids:
            row = [counts[qid].get(g, 0) for g in range(1, max_gen + 1)]
            writer.writerow([qid] + row + [sum(row)])
    print(f"generation table saved to: {path}")


def save_operator_table(results, category_data, run_tag):
    question_ids = sorted({item["question_id"] for item in category_data})

    counts = {qid: Counter() for qid in question_ids}
    for r in results:
        counts[r["question_id"]][r["mutation"]] += 1

    path = f"{OUTPUT_DIR}/{RUN_LABEL}_operator_table_{run_tag}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id"] + OPERATORS + ["total"])
        for qid in question_ids:
            row = [counts[qid].get(op, 0) for op in OPERATORS]
            writer.writerow([qid] + row + [sum(row)])
    print(f"operator table saved to: {path}")


def save_apply_failure_table(apply_failures_by_op, run_tag):
    path = f"{OUTPUT_DIR}/{RUN_LABEL}_apply_failures_{run_tag}.csv"

    reasons = sorted({
        reason
        for op_reasons in apply_failures_by_op.values()
        for reason in op_reasons
    })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["operator"] + reasons + ["total"])
        for op_name in sorted(apply_failures_by_op):
            op_reasons = apply_failures_by_op[op_name]
            row = [op_reasons.get(reason, 0) for reason in reasons]
            writer.writerow([op_name] + row + [sum(row)])
    print(f"apply failure table saved to: {path}")


def save_grammatical(raw_scores, run_tag):
    """Write only the grammatically-correct mutants (passed perplexity and CoLA)."""
    grammatical = [
        r for r in raw_scores
        if r.get("passed_perplexity") and r.get("passed_grammar")
    ]
    path = (
        f"{OUTPUT_DIR}/"
        f"{RUN_LABEL}_grammatical_{run_tag}.json"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(grammatical, f, indent=2, ensure_ascii=False)
    print(f"grammatical ({len(grammatical)}) saved to: {path}")


#Main fuzzer
def fuzz_prompts():

    output_path = (
        f"{OUTPUT_DIR}/"
        f"{RUN_LABEL}_fuzzed_prompt_{RUN_TAG}.json"
    )

    category_data = data
    narrow_ids = {str(item["category"]) for item in category_data}

    if not category_data:
        print(f"Skipping empty input file {INPUT_PATH}")
        return

    print(
        f"\n{RUN_LABEL} ({RUN_TAG}): {len(category_data)} prompts "
        f"across categories {sorted(narrow_ids, key=int)}"
    )

    # global-per-run id generator so every stage gets a unique, trackable id
    id_gen = count()

    def make_pid(qid):
        return f"q{qid}-g{next(id_gen)}"

    seed_pool = []
    for item in category_data:
        qid  = item["question_id"]
        text = item["prompt"]
        pid  = make_pid(qid)
        seed_pool.append({
            "prompt_id":     pid,
            "question_id":   qid,
            "category":      str(item["category"]),
            "parent_id":     None,
            "generation":    0,
            "text":          text,
            "original_text": text,
            "op_path":       []
        })

    results            = []
    seen               = set()
    raw_scores         = []
    stagnation_counter = 0
    start_time         = time.time()
    iteration          = 0

    total_generated      = 0
    discarded_apply      = 0
    discarded_no_op      = 0
    discarded_unchanged  = 0
    discarded_semantic   = 0
    discarded_perplexity = 0
    discarded_grammar    = 0
    discarded_duplicate  = 0
    discarded_similar    = 0
    apply_failures_by_op = defaultdict(Counter)

    while True:

        iteration += 1
        elapsed_minutes = (time.time() - start_time) / 60

        if elapsed_minutes >= FUZZ_DURATION_MINUTES:
            break

        if stagnation_counter >= MAX_STAGNATION:
            print(f"\nsearch exhausted for {RUN_LABEL} ({RUN_TAG})")
            break

        if len(results) >= MAX_MUTANTS_PER_PROMPT:
            break

        # pick any seed: an original, or a previously successful mutant
        parent        = random.choice(seed_pool)
        question_id   = parent["question_id"]
        original_text = parent["original_text"]

        op_pool = op_pool_for(parent["text"])

        if not op_pool:
            stagnation_counter += 1
            discarded_no_op += 1
            continue

        total_generated += 1

        mutant, discard_reason = fuzz_once(
            parent["text"],
            original_text,
            raw_scores,
            question_id,
            parent["category"],
            op_pool=op_pool
        )

        if mutant is None:
            stagnation_counter += 1
            if isinstance(discard_reason, tuple) and discard_reason[0] == "apply_failed":
                _, failed_op, failed_why = discard_reason
                discarded_apply += 1
                apply_failures_by_op[failed_op][failed_why] += 1
            elif discard_reason == "unchanged":
                discarded_unchanged += 1
            elif discard_reason == "semantic":
                discarded_semantic += 1
            elif discard_reason == "perplexity":
                discarded_perplexity += 1
            elif discard_reason == "grammar":
                discarded_grammar += 1
            continue

        unique_key = str(question_id) + "_" + mutant["mutated_prompt"]

        if unique_key in seen:
            stagnation_counter += 1
            discarded_duplicate += 1
            continue

        same_question_results = [
            r for r in results
            if r["question_id"] == question_id
        ]

        if too_similar_to_existing(
            mutant["mutated_prompt"],
            same_question_results,
            threshold=THRESHOLD_DEDUP
        ):
            stagnation_counter += 1
            discarded_similar += 1
            continue

        seen.add(unique_key)
        stagnation_counter = 0

        child_generation = parent["generation"] + 1
        child_pid        = make_pid(question_id)
        child_op_path    = parent["op_path"] + [mutant["mutation"]]

        results.append({
            "prompt_id":            child_pid,
            "question_id":          question_id,
            "category":             parent["category"],
            "parent_id":            parent["prompt_id"],
            "generation":           child_generation,
            "original_prompt":      original_text,
            "parent_prompt":        parent["text"],
            "mutation":             mutant["mutation"],
            "op_path":              child_op_path,
            "mutated_prompt":       mutant["mutated_prompt"],
            "semantic_similarity":  mutant["semantic_similarity"],
            "bias_metadata":        mutant.get("bias_metadata"),
        })

        if child_generation < MAX_GENERATION:
            seed_pool.append({
                "prompt_id":     child_pid,
                "question_id":   question_id,
                "category":      parent["category"],
                "parent_id":     parent["prompt_id"],
                "generation":    child_generation,
                "text":          mutant["mutated_prompt"],
                "original_text": original_text,
                "op_path":       child_op_path
            })

        save_results(results, output_path)

        print(
            f"[{elapsed_minutes:6.2f}m] {child_pid:<12s} "
            f"gen{child_generation:<2d} op={mutant['mutation']:<9s} "
            f"<- {parent['prompt_id']:<12s} "
            f"{mutant['mutated_prompt'][:80]}"
        )

    total_kept      = len(results)
    total_discarded = total_generated - total_kept
    keep_rate       = round(
        total_kept / max(total_generated, 1) * 100, 2
    )

    gen_distribution = dict(
        sorted(Counter(r["generation"] for r in results).items())
    )

    print("\n" + "=" * 60)
    print(f"{RUN_LABEL} ({RUN_TAG}) COMPLETE")
    print("=" * 60)
    print(f"total generated        : {total_generated}")
    print(f"total kept             : {total_kept}")
    print(f"total discarded        : {total_discarded}")
    print(f"  -> apply failed      : {discarded_apply}")
    for op_name in sorted(apply_failures_by_op):
        reasons = apply_failures_by_op[op_name]
        print(f"       {op_name:<14s}: {sum(reasons.values())}")
        for reason, cnt in reasons.most_common():
            print(f"           - {reason:<28s}: {cnt}")
    print(f"  -> unchanged         : {discarded_unchanged}")
    print(f"  -> semantic fail     : {discarded_semantic}")
    print(f"  -> perplexity fail   : {discarded_perplexity}")
    print(f"  -> grammar fail      : {discarded_grammar}")
    print(f"  -> duplicate         : {discarded_duplicate}")
    print(f"  -> too similar       : {discarded_similar}")
    print(f"no applicable operator : {discarded_no_op}")
    print(f"keep rate              : {keep_rate}%")
    print(f"seed pool final size   : {len(seed_pool)}")
    print(f"generation distribution: {gen_distribution}")
    print("=" * 60)

    report = {
        "input_file":              INPUT_PATH,
        "category_ids":            sorted(narrow_ids, key=int),
        "seed":                    SEED,
        "total_generated":         total_generated,
        "total_kept":              total_kept,
        "total_discarded":         total_discarded,
        "discarded_apply":         discarded_apply,
        "apply_failures_by_operator": {
            op_name: dict(reasons)
            for op_name, reasons in apply_failures_by_op.items()
        },
        "discarded_unchanged":     discarded_unchanged,
        "discarded_semantic":      discarded_semantic,
        "discarded_perplexity":    discarded_perplexity,
        "discarded_grammar":       discarded_grammar,
        "discarded_duplicate":     discarded_duplicate,
        "discarded_similar":       discarded_similar,
        "discarded_no_applicable_op": discarded_no_op,
        "keep_rate_pct":           keep_rate,
        "seed_pool_final_size":    len(seed_pool),
        "generation_distribution": gen_distribution,
        "max_generation":          MAX_GENERATION,
        "thresholds": {
            "semantic":    THRESHOLD_SEMANTIC,
            "perplexity":  THRESHOLD_PERPLEXITY,
            "grammar":     THRESHOLD_GRAMMAR,
            "different":   THRESHOLD_DIFFERENT,
            "dedup":       THRESHOLD_DEDUP
        }
    }

    save_report(report, RUN_TAG)
    save_raw_scores(raw_scores, RUN_TAG)
    # save_grammatical(raw_scores, RUN_TAG)

    approval_stats = mark_approved_prompts(results)
    save_results(results, output_path)
    print(f"approved ({approval_stats['approved']} of {approval_stats['total']}) stamped onto {output_path}")
    print("  (counts below overlap -- a record can fail more than one gate, so they don't sum to total-approved)")
    print(f"  -> failed lexical difference (vs original)  : {approval_stats['removed_lexical']}")
    print(f"  -> failed semantic similarity (vs original) : {approval_stats['removed_semantic']}")
    print(f"  -> failed diversity (vs approved siblings)  : {approval_stats['removed_diversity']}")

    save_generation_table(results, category_data, RUN_TAG)
    save_operator_table(results, category_data, RUN_TAG)
    save_apply_failure_table(apply_failures_by_op, RUN_TAG)


if __name__ == "__main__":
    fuzz_prompts()
