import os

# wildguard-"unharmful" prompts are dropped for mental health and self harm and harmful and violent content
# but not for fairness and bias as we find that wildguard marks some of the harmful fairness and bias prompts as unharmful
UNHARMFUL_KEPT_CATEGORIES = {"fairness_bias"}


def dataset_category(record, path):
    return (
        record.get("category_name")
        or os.path.basename(path).split("_mutated_prompt")[0].split("_original_prompt")[0]
    )


def is_tested_prompt(record, category):
    if record.get("approved") is False:
        return False
    return (
        record.get("wildguard_prompt_harmfulness") == "harmful"
        or category in UNHARMFUL_KEPT_CATEGORIES
    )
