import os
import json
import logging
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

#configuaration
MODE = os.environ.get("MODE", "mutated")  # "original" or "mutated"


# paths
INPUT_PATH = os.environ.get(
    "INPUT_PATH",
    "RQ2/outputs/prompts/combined_sample.json",
)

CATEGORY = os.path.basename(os.path.dirname(INPUT_PATH)) or "default"

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "RQ2/data/evaluation/model_response")

OUTPUT_PATH = f"{OUTPUT_DIR}/{CATEGORY}_{MODE}_prompt_result_gpt.json"
LOG_PATH = f"{OUTPUT_DIR}/{CATEGORY}_{MODE}_prompt_result_gpt.log"


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

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL_NAME = "gpt-5.5"


def call_gpt(prompt, max_retries=3):

    for attempt in range(max_retries):

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return response.choices[0].message.content

        except Exception as e:

            log.info(
                f"retry {attempt+1}/{max_retries}: {e}"
            )

            time.sleep(2)

    return None


def save(data):

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
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
    results = [r for r in loaded if r.get("prompt_id") in valid_ids]
    dropped = len(loaded) - len(results)
    if dropped:
        log.info(f"dropped {dropped} stale/incompatible entries from {OUTPUT_PATH}")
    done_ids = {r["prompt_id"] for r in results}
    data = [item for item in data if item["prompt_id"] not in done_ids]
    log.info(f"resuming: {len(done_ids)} already done, {len(data)} remaining")
else:
    results = []

log.info(f"total prompts: {len(data)}")


for i, item in enumerate(data, start=1):

    prompt = item["mutated_prompt"] if MODE == "mutated" else item["original_prompt"]

    response = call_gpt(prompt)

    results.append({
        **item,
        "model": MODEL_NAME,
        "gpt_response": response
    })

    # incremental save (safe for crashes)
    if i % 5 == 0:
        save(results)

    log.info(f"[{i}/{len(data)}] done")


save(results)

log.info("finished")
log.info(f"saved to: {OUTPUT_PATH}")
