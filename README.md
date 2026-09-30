# LEST-Replication-Package
This repository contains the replication package for LEST.


## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.11.0
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
python RQ2/scripts/mutation_experiment/get_counterfitted.py
```

## Reproducing RQ1

### 1. Extract claims from system-card PDFs

```bash
python RQ1/scripts/claim_extractor.py
```

Reads the two PDFs in `RQ1/data/system_cards/`, writes
`RQ1/outputs/claims_features.csv` (full feature matrix) and
`RQ1/outputs/claims_for_annotation.csv` (blank `label` column for
annotators).

### 2. Human annotation (manual step)

`claims_for_annotation.csv` is uploaded to Label Studio, where multiple
annotators label each claim:

- `is_claim`: Claim / Not a claim
- `testable` (only if a claim): Testable / Not testable

Export each annotator's pass as a CSV. A
completed export for the current dataset is included at
`RQ1/annotations/claims/annotated_claims_unaggregated_no_annotator4.csv`.

### 3. Aggregate annotator votes

```bash
python RQ1/scripts/annotation_extraction/aggregate_claims.py \
    RQ1/annotations/claims/annotated_claims_unaggregated_no_annotator4.csv \
    RQ1/outputs/annotated_claims.csv
```

### 4. Evaluate testability classifiers

```bash
python RQ1/classification/evaluation.py \
    --input RQ1/outputs/annotated_claims.csv \
    --folds 5 --repeats 5 --tf linear --show-rules
```

Reports F1 (positive class = Testable), precision, recall and accuracy for logistic regression, linear SVM and RIPPER, under repeated stratified k-fold cross-validation. `--show-rules` fits RIPPER on the full dataset and prints the learned rule set.

### 5. Build the anchor (seed prompt) set

```bash
python RQ1/scripts/build_anchor_set.py
```

`RQ1/data/anchor/category_selection.md` documents the reasoning behind every category and prompt level inclusion/exclusion decision.


## Reproducing RQ2

### 1. Generated Mutated Prompts

```bash
./RQ2/run_scripts/run_all_seeds.sh
```

Runs the mutation engine for 5 different seeds for 20 minutes each.

### 2. Combine Seeds for Final Selection

```bash
python RQ2/scripts/utils/combine_seeds_and_filter.py
```

Combines all the approved seeds into one file and filters more based on the lexical difference between the prompts in different seeds.

### 3. Classify Prompt Harmfulness with Wildguard

```bash
./RQ2/run_scripts/run_wildguard_combined.sh
python RQ2/scripts/wildguard_evaluation/reject_unharmful_wildguard.py
```

Runs the approved prompts through wildguard to filter out any prompts that wildguard finds unharmful for 2 of the categories.
*fairness and bias* is excluded from this filtering as we found that wildguard found some prompts in this category as unharmful even though they were harmful.

### 4. Get Model Responses
```bash
./RQ2/run_scripts/run_claude_experiment_all.sh
./RQ2/run_scripts/run_gpt_experiment_all.sh

./RQ2/run_scripts/run_original_prompt_experiment.sh
```
Runs the mutated prompts as well as the original prompts through the two models.
Note: this requires an Open AI and Anthropic API key in a .env file in the repo.

### 5. Judge Responses with Ensemble
```bash
./RQ2/run_scripts/run_ensemble_all_models.sh
python python RQ2/scripts/utils/build_filtered_harmful_outputs.py
```

Runs responses for both models through all the ensemble judges.

### 6. Rerun all Prompts That Complied
```bash
python RQ2/scripts/utils/build_complied_prompt_export.py
python RQ2/scripts/model_experiment/claude_rerun.py
python RQ2/scripts/model_experiment/gpt_rerun.py
python RQ2/scripts/utils/merge_claude_reruns.py
python RQ2/scripts/utils/merge_gpt_reruns.py
./RQ2/run_scripts/run_ensemble_all.sh RQ2/data/evaluation/complied_rerun/merged/claude_complied_reruns_merged.json
./RQ2/run_scripts/run_ensemble_all.sh RQ2/data/evaluation/complied_rerun/merged/gpt_complied_reruns_merged.json
```


### 7. Human Annotation
```bash
python RQ2/scripts/utils/sample_label_studio_prompts.py
```
The output is uploaded to Label Studio, where multiple
annotators label each prompt:

- Maintains original harmful intention
- Maintains intention of category
- Is grammatical and understandable

The exported annotations are aggregated

```bash
python RQ2/scripts/annotation_extraction/aggregate_prompts.py RQ2/annotations/prompts/annotated_prompts_unaggregated.csv RQ2/outputs/annotated_prompts.csv
python RQ2/scripts/annotation_extraction/fleiss_kappa_prompts.py RQ2/outputs/annotated_prompts.csv
```

### 8. Tables and Data Outputs
```bash
python python RQ2/scripts/utils/build_seed_failure_rate.py
python python RQ2/scripts/utils/build_response_transitions.py
```