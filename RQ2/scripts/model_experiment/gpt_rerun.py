import os
import json
import logging
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

#This file is used for rerunning the prompts that were originally complied with so that we have 5 total runs for each

INPUT_PATHS = os.environ.get(
    "INPUT_PATHS",
    "RQ2/outputs/complied_prompts/gpt-5.5_original_complied.json,"
    "RQ2/outputs/complied_prompts/gpt-5.5_mutated_complied.json",
).split(",")

START_RUN = int(os.environ.get("START_RUN", "2"))
NUM_RUNS = int(os.environ.get("NUM_RUNS", "4"))
RUNS = range(START_RUN, START_RUN + NUM_RUNS)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "RQ2/data/evaluation/complied_rerun")
LOG_PATH = f"{OUTPUT_DIR}/complied_rerun_gpt.log"

STALE_FIELDS = ("model_response", "tier", "models_flagged")


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
                f"retry {attempt + 1}/{max_retries}: {e}"
            )

            time.sleep(2)

    return None


def save(results, output_path):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )


def run_once(data, run, output_path):

    # this run's copy of every record, with a run-tagged prompt_id
    items = []
    for item in data:
        record = {k: v for k, v in item.items() if k not in STALE_FIELDS}
        record["base_prompt_id"] = item["prompt_id"]
        record["prompt_id"] = f"{item['prompt_id']}-run{run}"
        record["run"] = run
        # judge stages read mutated_prompt; the export calls it "prompt"
        record["mutated_prompt"] = item["prompt"]
        items.append(record)

    valid_ids = {r["prompt_id"] for r in items}
    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        in_scope = [r for r in loaded if r.get("prompt_id") in valid_ids]
        dropped = len(loaded) - len(in_scope)
        if dropped:
            log.info(f"dropped {dropped} stale/incompatible entries from {output_path}")
        results = [r for r in in_scope if r.get("gpt_response") is not None]
        n_retry = len(in_scope) - len(results)
        done_ids = {r["prompt_id"] for r in results}
        items = [r for r in items if r["prompt_id"] not in done_ids]
        log.info(
            f"resuming: {len(done_ids)} already done, {len(items)} remaining "
            f"({n_retry} retrying after a previous null response)"
        )
    else:
        results = []

    log.info(f"run {run}: {len(items)} prompts -> {output_path}")

    for i, record in enumerate(items, start=1):

        response = call_gpt(record["mutated_prompt"])

        results.append({
            **record,
            "model": MODEL_NAME,
            "gpt_response": response
        })

        # incremental save (safe for crashes)
        if i % 5 == 0:
            save(results, output_path)

        log.info(f"[run {run}] [{i}/{len(items)}] done")

    save(results, output_path)
    log.info(f"run {run} finished, saved to: {output_path}")


for input_path in INPUT_PATHS:

    log.info(f"loading dataset {input_path}...")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # e.g. gpt-5.5_mutated_complied.json -> gpt-5.5_mutated_complied_run2_gpt.json
    stem = os.path.splitext(os.path.basename(input_path))[0]

    for run in RUNS:
        output_path = f"{OUTPUT_DIR}/{stem}_run{run}_gpt.json"
        run_once(data, run, output_path)

log.info("finished")
