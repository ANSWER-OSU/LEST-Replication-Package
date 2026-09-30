import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "judge_ensemble"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wildguard import load_wildguard

import original_prompt_experiment as ope
from ensemble_io import load_json, save_json
from run_qwen3guard import MODEL_NAME as QWEN_MODEL, parse_qwen3guard

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = f"{ope.BASE}/prompt_harmfulness"


def load_records():
    records = []
    for category in ope.CATEGORIES:
        records += load_json(ope.input_path(category))
    return records


def wildguard_labels(prompts):
    print(f"loading wildguard... (device={DEVICE})")
    wildguard = load_wildguard(
        use_vllm=False, device=DEVICE, batch_size=16, ephemeral_model=False
    )
    wildguard.tokenizer.padding_side = "left"  # see run_wildguard.py
    predictions = wildguard.classify([{"prompt": p} for p in prompts])
    del wildguard
    torch.cuda.empty_cache()
    return [(p["prompt_harmfulness"], p["is_parsing_error"]) for p in predictions]


def qwen3guard_labels(prompts):
    print(f"loading {QWEN_MODEL}... (device={DEVICE})")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, torch_dtype="auto", device_map=DEVICE
    ).eval()
    labels = []
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=128)
        output = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        _, safety, categories = parse_qwen3guard(output)
        labels.append((safety, categories))
    return labels


def main():
    records = load_records()
    prompts = [r["original_prompt"] for r in records]
    print(f"{len(prompts)} original prompts")

    wg = wildguard_labels(prompts)
    qw = qwen3guard_labels(prompts)

    out = {category: [] for category in ope.CATEGORIES}
    for record, (harm, parse_error), (safety, categories) in zip(records, wg, qw):
        out[record["category_name"]].append(
            {
                **record,
                "wildguard_prompt_harmfulness": harm,
                "wildguard_prompt_parsing_error": parse_error,
                "qwen3guard_prompt_safety": safety,
                "qwen3guard_prompt_categories": categories,
            }
        )
    for category, rows in out.items():
        path = f"{OUT_DIR}/{category}_original_prompt_harmfulness.json"
        save_json(path, rows)
        print(f"{category}: {len(rows)} rows -> {path}")

    print("wildguard prompt_harmfulness:", dict(Counter(h for h, _ in wg)))
    print("qwen3guard prompt safety:", dict(Counter(s for s, _ in qw)))


if __name__ == "__main__":
    main()
