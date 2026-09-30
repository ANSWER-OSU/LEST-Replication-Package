import argparse
import os
import json

os.environ["WILDGARD_USE_VLLM"] = "0"

import torch
from wildguard import load_wildguard

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
SAVE_EVERY = 20
DEFAULT_INPUT = (
    "RQ2/data/evaluation/model_response/fairness_bias_mutated_prompt_result_claude.json"
)


def response_text(item):
    for key in ("claude_response", "gpt_response"):
        value = item.get(key)
        if value:
            return value
    return None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_and_resume(input_path, output_path):
    records = load_json(input_path)
    if os.path.isfile(output_path):
        done_rows = load_json(output_path)
        done = [r for r in done_rows if r.get("wildguard_response_error") is None]
        done_ids = {r["prompt_id"] for r in done}
        remaining = [r for r in records if r["prompt_id"] not in done_ids]
        n_retry = len(done_rows) - len(done)
        print(
            f"resuming: {len(done)} already scored, {len(remaining)} remaining "
            f"({n_retry} retrying after a previous error)"
        )
        return done, remaining
    return [], records


def main():
    parser = argparse.ArgumentParser(
        description="Classify a model_response file's target-model completions with WildGuard (refusal voter)."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=os.environ.get("INPUT_PATH", DEFAULT_INPUT),
        help="model_response-style JSON file to score",
    )
    args = parser.parse_args()

    input_path = args.input_path
    stem = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(os.path.dirname(input_path) or ".", f"{stem}_wildguard_result.json")

    done_results, records = load_and_resume(input_path, output_path)

    if not records:
        save_json(output_path, done_results)
        print("nothing to do, saved:", output_path)
        return

    print(f"loading wildguard... (device={DEVICE})")
    wildguard = load_wildguard(use_vllm=False, device=DEVICE)
    print("wildguard loaded")

    results = list(done_results)
    try:
        for i, item in enumerate(records, start=1):
            response = response_text(item)
            if response is None:
                results.append(
                    {
                        **item,
                        "wildguard_response_harmfulness": None,
                        "wildguard_response_refusal": None,
                        "wildguard_response_parsing_error": None,
                        "wildguard_response_error": "no_response",
                    }
                )
            else:
                try:
                    prediction = wildguard.classify(
                        [{"prompt": item["mutated_prompt"], "response": response}]
                    )[0]
                    refusal = prediction.get("response_refusal")
                    results.append(
                        {
                            **item,
                            "wildguard_response_harmfulness": prediction.get("response_harmfulness"),
                            "wildguard_response_refusal": refusal,
                            "wildguard_response_parsing_error": prediction.get("is_parsing_error"),
                            "wildguard_response_error": None if refusal else "parse_failed",
                        }
                    )
                except Exception as e:
                    print(f"error on {item.get('prompt_id')}: {e}")
                    results.append(
                        {
                            **item,
                            "wildguard_response_harmfulness": None,
                            "wildguard_response_refusal": None,
                            "wildguard_response_parsing_error": True,
                            "wildguard_response_error": f"exception: {e}",
                        }
                    )

            if i % SAVE_EVERY == 0 or i == len(records):
                save_json(output_path, results)
                print(f"[{i}/{len(records)}] saved")
    finally:
        save_json(output_path, results)

    print("finished")
    print("saved to:", output_path)


if __name__ == "__main__":
    main()
