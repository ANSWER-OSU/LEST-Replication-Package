import argparse
import os
import json

import torch
from wildguard import load_wildguard

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description="Classify a fuzzer output file's mutated prompts with WildGuard."
)
parser.add_argument(
    "input_path",
    nargs="?",
    default=os.environ.get(
        "INPUT_PATH",
        "RQ2/data/mutated/fairness_bias/fairness_bias_fuzzed_prompt_combined_seeds.json",
    ),
    help="fuzzer output JSON file to classify",
)
args = parser.parse_args()

INPUT_PATH  = args.input_path
CATEGORY    = os.path.basename(os.path.dirname(INPUT_PATH))
INPUT_STEM  = os.path.splitext(os.path.basename(INPUT_PATH))[0]
OUTPUT_DIR  = f"RQ2/data/evaluation/wildguard/{CATEGORY}"
OUTPUT_PATH = f"{OUTPUT_DIR}/{INPUT_STEM}_wildguard.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16


def save(results):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


print(f"loading wildguard... (device={DEVICE})")
wildguard = load_wildguard(
    use_vllm=False, device=DEVICE, batch_size=BATCH_SIZE, ephemeral_model=False
)
wildguard.tokenizer.padding_side = "left"
print("wildguard loaded")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    prompts = json.load(f)

prompts = [item for item in prompts if item.get("approved") is True]
print(f"approved prompts: {len(prompts)}")

if os.path.isfile(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        done_results = json.load(f)
    done_ids = {r["prompt_id"] for r in done_results}
    prompts = [item for item in prompts if item["prompt_id"] not in done_ids]
    print(f"resuming: {len(done_ids)} already classified, {len(prompts)} remaining")
else:
    done_results = []

print("total prompts to classify:", len(prompts))

if prompts:
    def save_progress(predictions):
        results = done_results + [
            {
                **item,
                "wildguard_prompt_harmfulness": prediction["prompt_harmfulness"],
                "wildguard_parsing_error": prediction["is_parsing_error"],
            }
            for item, prediction in zip(prompts, predictions)
        ]
        save(results)
        print(f"[{len(predictions)}/{len(prompts)}] saved")

    wildguard.classify(
        [{"prompt": item["mutated_prompt"]} for item in prompts],
        save_func=save_progress,
    )
else:
    save(done_results)

print("finished")
print("saved to:", OUTPUT_PATH)
