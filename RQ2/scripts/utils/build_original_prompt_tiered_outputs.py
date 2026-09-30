import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import build_filtered_harmful_outputs as filtered_harmful
import build_model_summary_csv as model_summary
import build_tier_summary_csv as tier_summary

INPUT_DIRS = [
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "evaluation",
        "original_prompt_experiment",
        "judge_ensemble",
    )
]
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "evaluation",
    "original_prompt_experiment",
    "tiered",
)


def main():
    filtered_harmful.main(input_dirs=INPUT_DIRS, output_dir=OUTPUT_DIR, require_harmful=False)
    model_summary.main(
        input_dirs=INPUT_DIRS,
        out_path=os.path.join(OUTPUT_DIR, "model_summary.csv"),
        require_harmful=False,
    )
    tier_summary.main(
        input_dirs=INPUT_DIRS,
        out_path=os.path.join(OUTPUT_DIR, "tier_summary.csv"),
        require_harmful=False,
    )


if __name__ == "__main__":
    main()
