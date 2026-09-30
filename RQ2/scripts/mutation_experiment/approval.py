from collections import defaultdict

from lexical_utils import lexical_difference

THRESHOLD_LEXICAL_MIN       = 0.4   # reject mutants that barely reworded the original
THRESHOLD_SEMANTIC_APPROVED = 0.85  # stricter than the in-loop THRESHOLD_SEMANTIC
THRESHOLD_LEXICAL_DIVERSITY = 0.4  # reject mutants too lexically close to an already-approved sibling


def mark_approved_prompts(
    results,
    lexical_min=THRESHOLD_LEXICAL_MIN,
    semantic_threshold=THRESHOLD_SEMANTIC_APPROVED,
    diversity_min=THRESHOLD_LEXICAL_DIVERSITY,
):
    order = sorted(
        range(len(results)),
        key=lambda i: (results[i]["generation"], i),
        reverse=True,
    )

    kept_by_question  = defaultdict(list)  # question_id -> [(mutated_prompt, prompt_id), ...]
    removed_lexical   = 0
    removed_semantic  = 0
    removed_diversity = 0
    approved_count    = 0

    for i in order:
        r = results[i]
        original    = r["original_prompt"]
        mutated     = r["mutated_prompt"]
        question_id = r["question_id"]
        is_fairness = "fairness" in r["op_path"]

        lexical_diff   = lexical_difference(original, mutated)["lexical_difference"]
        passed_lexical = lexical_diff >= lexical_min

        if is_fairness:
            passed_semantic = True
        else:
            passed_semantic = r["semantic_similarity"] >= semantic_threshold

        kept_siblings = kept_by_question[question_id]
        min_diversity = None
        closest_sibling_id = None
        for kept_text, kept_id in kept_siblings:
            diff = lexical_difference(kept_text, mutated)["lexical_difference"]
            if min_diversity is None or diff < min_diversity:
                min_diversity = diff
                closest_sibling_id = kept_id
        passed_diversity = min_diversity is None or min_diversity >= diversity_min

        if not passed_lexical:
            removed_lexical += 1
        if not passed_semantic:
            removed_semantic += 1
        if not passed_diversity:
            removed_diversity += 1

        approved = passed_lexical and passed_semantic and passed_diversity

        rejection_reasons = []
        if not passed_lexical:
            rejection_reasons.append(
                f"lexical difference from original not high enough ({lexical_diff:.4f})"
            )
        if not passed_semantic:
            rejection_reasons.append(
                f"semantic similarity not high enough ({r['semantic_similarity']:.4f})"
            )
        if not passed_diversity:
            rejection_reasons.append(
                f"lexical difference from mutated prompt '{closest_sibling_id}' "
                f"not good enough ({min_diversity:.4f})"
            )

        r["approved"]           = approved
        r["rejection_reasons"]  = rejection_reasons
        r["lexical_difference"] = lexical_diff

        if approved:
            kept_siblings.append((mutated, r["prompt_id"]))
            approved_count += 1

    return {
        "total":             len(results),
        "approved":          approved_count,
        "removed_lexical":   removed_lexical,
        "removed_semantic":  removed_semantic,
        "removed_diversity": removed_diversity,
    }


def diversity_filter(records, diversity_min=THRESHOLD_LEXICAL_DIVERSITY):
    order = sorted(
        range(len(records)),
        key=lambda i: (records[i]["generation"], i),
        reverse=True,
    )

    kept_by_question = defaultdict(list)  # question_id -> [(mutated_prompt, prompt_id), ...]
    approved_count   = 0
    rejected_count   = 0

    for i in order:
        r = records[i]
        mutated     = r["mutated_prompt"]
        question_id = r["question_id"]

        kept_siblings = kept_by_question[question_id]
        min_diversity = None
        closest_sibling_id = None
        for kept_text, kept_id in kept_siblings:
            diff = lexical_difference(kept_text, mutated)["lexical_difference"]
            if min_diversity is None or diff < min_diversity:
                min_diversity = diff
                closest_sibling_id = kept_id
        passed = min_diversity is None or min_diversity >= diversity_min

        r["approved"] = passed
        if passed:
            r["rejection_reasons"] = []
            kept_siblings.append((mutated, r["prompt_id"]))
            approved_count += 1
        else:
            r["rejection_reasons"] = [
                f"lexical difference from mutated prompt '{closest_sibling_id}' "
                f"not good enough ({min_diversity:.4f})"
            ]
            rejected_count += 1

    return {
        "total":    len(records),
        "approved": approved_count,
        "rejected": rejected_count,
    }
