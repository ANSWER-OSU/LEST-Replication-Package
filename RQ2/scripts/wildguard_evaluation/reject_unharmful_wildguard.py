import json
from pathlib import Path

WILDGUARD_DIR = Path("RQ2/data/evaluation/wildguard")

# fairness_bias is left out: wildguard marks some harmful fairness/bias prompts as unharmful
TARGET_FILES = [
    WILDGUARD_DIR / "harmful_violent_content" / "harmful_violent_content_fuzzed_prompt_combined_seeds_wildguard.json",
    WILDGUARD_DIR / "mental_health_self_harm" / "mental_health_self_harm_fuzzed_prompt_combined_seeds_wildguard.json",
]

REJECTION_REASON = "wildguard_unharmful"


def process_file(path: Path) -> None:
    with open(path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    changed = 0
    for prompt in prompts:
        if prompt.get("wildguard_prompt_harmfulness") == "unharmful" and prompt.get("approved") is not False:
            prompt["approved"] = False
            reasons = prompt.setdefault("rejection_reasons", [])
            if REJECTION_REASON not in reasons:
                reasons.append(REJECTION_REASON)
            changed += 1

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)

    print(f"{path}: {changed} prompts set to approved=False (of {len(prompts)} total)")


def main() -> None:
    for path in TARGET_FILES:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        process_file(path)


if __name__ == "__main__":
    main()
