import os
import json
import logging
import time
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

#configuaration
MODE = os.environ.get("MODE", "mutated")  # "original" or "mutated"


INPUT_PATH = os.environ.get(
    "INPUT_PATH",
    "RQ2/data/evaluation/wildguard/fairness_bias/fairness_bias_fuzzed_prompt_combined_seeds_wildguard_without_sampled.json",
)

CATEGORY = os.path.basename(os.path.dirname(INPUT_PATH))

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "RQ2/data/evaluation/model_response")

OUTPUT_PATH = f"{OUTPUT_DIR}/{CATEGORY}_{MODE}_prompt_result_claude.json"
LOG_PATH = f"{OUTPUT_DIR}/{CATEGORY}_{MODE}_prompt_result_claude.log"


os.makedirs(OUTPUT_DIR, exist_ok=True)

# log to both console and a file (append, so resumed runs keep the same log)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# client
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

MODEL_NAME = "claude-opus-4-8"


def call_claude(prompt, max_retries=3):

    for attempt in range(max_retries):

        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return response.content[0].text

        except Exception as e:

            log.info(
                f"retry {attempt + 1}/{max_retries}: {e}"
            )

            time.sleep(2)

    return None


def save(results):

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )


log.info("loading dataset...")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

if MODE == "original":

    seen = set()
    filtered_data = []

    for item in data:
        original = item["original_prompt"]

        if original not in seen:
            seen.add(original)
            filtered_data.append(item)

    data = filtered_data

valid_ids = {item["prompt_id"] for item in data}
if os.path.isfile(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    in_scope = [r for r in loaded if r.get("prompt_id") in valid_ids]
    dropped = len(loaded) - len(in_scope)
    if dropped:
        log.info(f"dropped {dropped} stale/incompatible entries from {OUTPUT_PATH}")
    results = [r for r in in_scope if r.get("claude_response") is not None]
    n_retry = len(in_scope) - len(results)
    done_ids = {r["prompt_id"] for r in results}
    data = [item for item in data if item["prompt_id"] not in done_ids]
    log.info(
        f"resuming: {len(done_ids)} already done, {len(data)} remaining "
        f"({n_retry} retrying after a previous null response)"
    )
else:
    results = []

log.info(f"total prompts: {len(data)}")


for i, item in enumerate(data, start=1):

    prompt = item["mutated_prompt"] if MODE == "mutated" else item["original_prompt"]

    response = call_claude(prompt)

    results.append({
        **item,
        "model": MODEL_NAME,
        "claude_response": response
    })

    # incremental save (safe for crashes)
    if i % 5 == 0:
        save(results)

    log.info(f"[{i}/{len(data)}] done")


save(results)

log.info("finished")
log.info(f"saved to: {OUTPUT_PATH}")
