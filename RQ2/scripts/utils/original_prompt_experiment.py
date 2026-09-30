import json
import os
import sys

CATEGORIES = ["fairness_bias", "harmful_violent_content", "mental_health_self_harm"]
MODELS = ["claude", "gpt"]
RAW_DIR = "RQ2/data/raw"
BASE = "RQ2/data/evaluation/original_prompt_experiment"


def load_raw(category):
    with open(f"{RAW_DIR}/{category}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def input_path(category):
    return f"{BASE}/inputs/{category}/{category}_original_prompts.json"


def stage_paths(category, model):
    stem = f"{category}_original_prompt_result_{model}"
    return {
        "responses": f"{BASE}/responses/{stem}.json",
        "wildguard": f"{BASE}/responses/{stem}_wildguard_result.json",
        "ensemble": f"{BASE}/judge_ensemble/{stem}_wildguard_result_ensemble.json",
    }


def build():
    total = 0
    for category in CATEGORIES:
        raw = load_raw(category)
        prompts = [r["prompt"] for r in raw]
        if len(set(prompts)) != len(prompts):
            # original mode in the experiment scripts dedups on prompt text
            sys.exit(f"{category}: duplicate prompt text, would be silently dropped")

        records = [
            {
                "prompt_id": f"q{r['question_id']}-orig",
                "question_id": r["question_id"],
                "category": r["category"],
                "category_name": category,
                "prompt_type": "original",
                "original_prompt": r["prompt"],
                "mutated_prompt": r["prompt"],
            }
            for r in raw
        ]

        path = input_path(category)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"{category}: {len(records)} prompts -> {path}")
        total += len(records)
    print(f"total original prompts: {total}")


def verify():
    ok = True
    for category in CATEGORIES:
        raw = {f"q{r['question_id']}-orig": r["prompt"] for r in load_raw(category)}
        for model in MODELS:
            for stage, path in stage_paths(category, model).items():
                label = f"{category}/{model}/{stage}"
                if not os.path.isfile(path):
                    print(f"MISSING  {label}: {path}")
                    ok = False
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    rows = json.load(f)

                ids = [r["prompt_id"] for r in rows]
                missing = set(raw) - set(ids)
                extra = set(ids) - set(raw)
                changed = [
                    r["prompt_id"]
                    for r in rows
                    if r["prompt_id"] in raw
                    and (
                        r.get("original_prompt") != raw[r["prompt_id"]]
                        or r.get("mutated_prompt") != raw[r["prompt_id"]]
                    )
                ]
                key = f"{model}_response"
                null_resp = sum(1 for r in rows if r.get(key) is None)

                bad = bool(missing or extra or changed or len(ids) != len(set(ids)))
                ok = ok and not bad
                print(
                    f"{'FAIL' if bad else 'ok  '}  {label}: {len(rows)}/{len(raw)} rows, "
                    f"{null_resp} null responses, {len(changed)} prompts differ from raw, "
                    f"{len(missing)} missing, {len(extra)} unexpected"
                )
        path = f"{BASE}/prompt_harmfulness/{category}_original_prompt_harmfulness.json"
        label = f"{category}/prompt_harmfulness"
        if not os.path.isfile(path):
            print(f"MISSING  {label}: {path}")
            ok = False
            continue
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        changed = [
            r["prompt_id"]
            for r in rows
            if raw.get(r["prompt_id"]) != r.get("original_prompt")
        ]
        unlabeled = sum(
            1
            for r in rows
            if r.get("wildguard_prompt_harmfulness") is None
            or r.get("qwen3guard_prompt_safety") is None
        )
        bad = bool(changed or unlabeled or len(rows) != len(raw))
        ok = ok and not bad
        print(
            f"{'FAIL' if bad else 'ok  '}  {label}: {len(rows)}/{len(raw)} rows, "
            f"{len(changed)} prompts differ from raw, {unlabeled} unlabeled"
        )
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    commands = {"build": build, "verify": verify}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit("usage: original_prompt_experiment.py build|verify")
    commands[sys.argv[1]]()
