import os

from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset

from ensemble_io import parse_args, load_and_resume, response_text, save_json

SAVE_EVERY = 8  # one evaluate_dataset() call = one checkpoint; keep it equal to BATCH_SIZE
BATCH_SIZE = int(os.environ.get("STRONGREJECT_BATCH_SIZE", "8"))


def main():
    input_path, output_path = parse_args("strongreject")
    done_results, records = load_and_resume(
        input_path, output_path, "strongreject_error"
    )

    if not records:
        save_json(output_path, done_results)
        print("nothing to do, saved:", output_path)
        return

    results = list(done_results)
    try:
        for start in range(0, len(records), SAVE_EVERY):
            chunk = records[start : start + SAVE_EVERY]

            scoreable = [
                item
                for item in chunk
                if response_text(item) is not None and "original_prompt" in item
            ]
            unscoreable = [
                (item, "no_response" if response_text(item) is None else "missing_original_prompt")
                for item in chunk
                if response_text(item) is None or "original_prompt" not in item
            ]

            chunk_results = [
                {
                    **item,
                    "strongreject_score": None,
                    "strongreject_prompt_source": "original_prompt",
                    "strongreject_error": error,
                }
                for item, error in unscoreable
            ]

            if scoreable:
                try:
                    dataset = Dataset.from_dict(
                        {
                            "forbidden_prompt": [item["original_prompt"] for item in scoreable],
                            "response": [response_text(item) for item in scoreable],
                        }
                    )
                    evaluated = evaluate_dataset(
                        dataset, ["strongreject_finetuned"], batch_size=BATCH_SIZE
                    )
                    for item, score in zip(scoreable, evaluated["score"]):
                        chunk_results.append(
                            {
                                **item,
                                "strongreject_score": score,
                                "strongreject_prompt_source": "original_prompt",
                                "strongreject_error": None,
                            }
                        )
                except Exception as e:
                    print(f"error scoring chunk starting at {start}: {e}")
                    for item in scoreable:
                        chunk_results.append(
                            {
                                **item,
                                "strongreject_score": None,
                                "strongreject_prompt_source": "original_prompt",
                                "strongreject_error": f"exception: {e}",
                            }
                        )

            results.extend(chunk_results)
            save_json(output_path, results)
            print(f"[{min(start + SAVE_EVERY, len(records))}/{len(records)}] saved")
    finally:
        save_json(output_path, results)

    print("finished")
    print("saved to:", output_path)


if __name__ == "__main__":
    main()
