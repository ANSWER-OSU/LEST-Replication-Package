import argparse
import json
import os

DEFAULT_INPUT = (
    "RQ2/data/evaluation/model_response/fairness_bias_mutated_prompt_result_claude_wildguard_result.json"
)
OUTPUT_ROOT = os.environ.get("ENSEMBLE_OUTPUT_ROOT", "RQ2/data/evaluation/judge_ensemble")


def parse_args(judge_name, default_input=DEFAULT_INPUT):
    parser = argparse.ArgumentParser(
        description=f"Score target-model responses with {judge_name} for the refusal/compliance ensemble."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=os.environ.get("INPUT_PATH", default_input),
        help="wildguard_result-style JSON file to score",
    )
    args = parser.parse_args()

    input_path = args.input_path
    stem = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"{OUTPUT_ROOT}/{stem}_{judge_name}.json"
    return input_path, output_path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def response_text(item):
    for key in ("claude_response", "gpt_response"):
        value = item.get(key)
        if value:
            return value
    return None


def _row_key(r):
    if "_source_file" in r:
        return (r["_source_file"], r["prompt_id"])
    return r["prompt_id"]


def load_and_resume(input_path, output_path, error_key):
    records = load_json(input_path)
    if os.path.isfile(output_path):
        done_rows = load_json(output_path)
        done = [r for r in done_rows if r.get(error_key) is None]
        done_keys = {_row_key(r) for r in done}
        remaining = [r for r in records if _row_key(r) not in done_keys]
        n_retry = len(done_rows) - len(done)
        print(
            f"resuming: {len(done)} already scored, {len(remaining)} remaining "
            f"({n_retry} retrying after a previous error)"
        )
        return done, remaining
    return [], records
