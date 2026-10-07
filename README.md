# LEST-Replication-Package
This artifact contains the code and data to replicate the results presented in the paper titled: **From LLM Safety Claims to Validated Counterexamples: Specification-Based Testing of LLMs**.
In Proceedings of International Conference on the Foundations of Software Engineering (FSE’27)-Under Review. 

**LEST(LLM Evaluation through Specification-driven Testing)** is a framework for black-box testing LLM safety claims as behavioral specifications, using prompt mutation to search for Refuse→Comply counterexamples, human validation to confirm them and traceability back to the claim each one contradicts. The artifact is organized in terms of the three research questions addressed in the paper:

**RQ1:Claim testability:** This involves code and data to extract safety-related claims from LLM system-card PDFs (Claude, GPT-5) (claim extraction), classify each claim as testable or not testable using logistic regression, linear SVM and RIPPER rule induction (testability classification).

**RQ2: Specification coverage and behavioral counterexamples:** This involves code and data to mutate the seed prompts with seven operators; POS-constrained lexical substitution, modifier insertion, deletion, fairness mutation, paraphrase and Person Conversion(First-to-Third Person and Third-to-First), run the original and mutated prompts against Claude and GPT (model experiment), classify prompt harmfulness and response refusal with WildGuard and score each response for refusal vs. compliance with a three-judge ensemble (WildGuard, GuardReasoner and Qwen3Guard).

**RQ3: Variation across mutation operators:** This involves code and data to measure how often each mutation operator flips a model's response from refusal to compliance, relative to its unmutated baseline.

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
python RQ2/scripts/mutation_experiment/get_counterfitted.py
```

`get_counterfitted.py` downloads the counter-fitted word vectors to `RQ2/scripts/mutation_experiment/counter-fitted-vectors.txt`.

## Reproducing RQ1

### 1. Extract claims from system-card PDFs

```bash
python RQ1/scripts/claim_extractor.py 
```

Reads the two PDFs in `RQ1/data/system_cards/`.
Outputs to `RQ1/outputs/claims_features.csv` and `RQ1/outputs/claims_for_annotation.csv`.

### 2. Human annotation (manual step)

`claims_for_annotation.csv` is uploaded to Label Studio, where multiple annotators label each claim:

- `is_claim`: Claim / Not a claim
- `testable` (only if a claim): Testable / Not testable

Export each annotator's pass as a CSV. A completed export for the current dataset is included at `RQ1/annotations/claims/annotated_claims_unaggregated.csv`.

### 3. Aggregate annotator votes

```bash
python RQ1/scripts/annotation_extraction/aggregate_claims.py \
    RQ1/annotations/claims/annotated_claims_unaggregated.csv \
    RQ1/outputs/annotated_claims.csv
python RQ1/scripts/annotation_extraction/fleiss_kappa.py RQ1/outputs/annotated_claims.csv
```

Outputs to `RQ1/outputs/annotated_claims.csv` and `RQ1/outputs/claims_fleiss_kappa.csv`.

### 4. Evaluate testability classifiers

```bash
python RQ1/classification/evaluation.py \
    --input RQ1/outputs/annotated_claims.csv \
    --folds 5 --repeats 5 --tf linear --show-rules
```

Reports F1 (positive class = Testable), precision, recall and accuracy for logistic regression, linear SVM and RIPPER, under repeated stratified k-fold cross-validation. `--show-rules` fits RIPPER on the full dataset and prints the learned rule set.
Outputs to `RQ1/outputs/classifier_evaluation.csv`.

### 5. Human Annotation (ToC Domain Mapping)

```bash
python RQ1/scripts/build_toc_domain_tasks.py
```

Outputs to `RQ1/annotations/toc_domains/tasks.json`.

The output is uploaded to Label Studio, where multiple annotators label each safety domain / model pair with the relevant table of contents sections from the system card.

The exported annotations are aggregated:

```bash
python RQ1/scripts/annotation_extraction/aggregate_toc_domains.py \
    RQ1/annotations/toc_domains/toc_domain_mapping_unaggregated.csv \
    RQ1/outputs/toc_domains.csv
python RQ1/scripts/annotation_extraction/fleiss_kappa_toc_domains.py RQ1/outputs/toc_domains.csv
```

Outputs to `RQ1/outputs/toc_domains.csv` and `RQ1/outputs/toc_domains_fleiss_kappa.csv`.

## Reproducing RQ2

### 1. Select Seed Prompts from SORRY-Bench

```bash
python RQ2/scripts/seed_selection/extract_sorrybench_base.py
```

Note: Requires HuggingFace authentication
Outputs the 440 base prompts of SORRY-Bench to `RQ2/data/sorrybench/sorrybench_base.json`.

```bash
python RQ2/scripts/seed_selection/build_anchor_set.py
```

Outputs the 97 seed prompts to `RQ2/data/raw/<category>.json` and the keep/exclude decision for every considered prompt to `RQ2/data/sorrybench/anchor_audit.csv`.

### 2. Generated Mutated Prompts

```bash
./RQ2/run_scripts/run_all_seeds.sh
```

Runs the mutation engine for 5 different seeds for 20 minutes each.
Outputs to `RQ2/data/mutated/<category>/` (per-seed files, e.g. `<category>_fuzzed_prompt_seed<seed>.json`), with logs in `logs/experiment_<timestamp>/`.

### 3. Combine Seeds for Final Selection

```bash
python RQ2/scripts/utils/combine_seeds_and_filter.py
```

Combines all the approved seeds into one file and filters more based on the lexical difference between the prompts in different seeds.
Outputs to `RQ2/data/mutated/<category>/<category>_fuzzed_prompt_combined_seeds.json`.

### 4. Classify Prompt Harmfulness with Wildguard

```bash
./RQ2/run_scripts/run_wildguard_combined.sh
python RQ2/scripts/wildguard_evaluation/reject_unharmful_wildguard.py
```

Runs the approved prompts through wildguard to filter out any prompts that wildguard finds unharmful for 2 of the categories.
*fairness and bias* is excluded from this filtering as we found that wildguard found some prompts in this category as unharmful even though they were harmful.
Outputs to `RQ2/data/evaluation/wildguard/<category>/<category>_fuzzed_prompt_combined_seeds_wildguard.json`; `reject_unharmful_wildguard.py` updates these files in place.

### 5. Get Model Responses
Note: this requires an Open AI and Anthropic API key in a .env file in the repo.
```bash
./RQ2/run_scripts/run_claude_experiment_all.sh
./RQ2/run_scripts/run_gpt_experiment_all.sh

./RQ2/run_scripts/run_original_prompt_experiment.sh
```

Runs the mutated prompts as well as the original prompts through the two models.

Outputs to `RQ2/data/evaluation/model_response/<category>_mutated_prompt_result_{claude,gpt}.json` (mutated prompts) and `RQ2/data/evaluation/original_prompt_experiment/` (original prompts: `inputs/`, `responses/`, `judge_ensemble/`, `logs/`).

### 6. Judge Responses with Ensemble
```bash
./RQ2/run_scripts/run_ensemble_all_models.sh
python RQ2/scripts/utils/build_filtered_harmful_outputs.py
```

Runs responses for both models through all the ensemble judges.
Outputs to:
- `run_ensemble_all_models.sh`: `RQ2/data/evaluation/model_response/*_wildguard_result.json` and `RQ2/data/evaluation/judge_ensemble/`
- `build_filtered_harmful_outputs.py`: `RQ2/data/evaluation/filtered_harmful/` (`ensemble_compliance_tiers.json`, `all_five_signal_tiers.json`, `ensemble_harmfulness_tiers.json`)

### 7. Rerun all Prompts That Complied
```bash
python RQ2/scripts/utils/build_complied_prompt_export.py
python RQ2/scripts/model_experiment/claude_rerun.py
python RQ2/scripts/model_experiment/gpt_rerun.py
python RQ2/scripts/utils/merge_claude_reruns.py
python RQ2/scripts/utils/merge_gpt_reruns.py
./RQ2/run_scripts/run_ensemble_all.sh RQ2/data/evaluation/complied_rerun/merged/claude_complied_reruns_merged.json
./RQ2/run_scripts/run_ensemble_all.sh RQ2/data/evaluation/complied_rerun/merged/gpt_complied_reruns_merged.json
```

Outputs to:
- `build_complied_prompt_export.py`: `RQ2/outputs/complied_prompts/`
- `claude_rerun.py`, `gpt_rerun.py`: `RQ2/data/evaluation/complied_rerun/`
- `merge_claude_reruns.py`, `merge_gpt_reruns.py`: `RQ2/data/evaluation/complied_rerun/merged/{claude,gpt}_complied_reruns_merged.json`
- `run_ensemble_all.sh`: `RQ2/data/evaluation/judge_ensemble/{claude,gpt}_complied_reruns_merged_wildguard_result_*.json`


### 8. Human Annotation (Sampled Prompts)
```bash
python RQ2/scripts/utils/sample_label_studio_prompts.py
```

Outputs to `RQ2/outputs/label_studio/mutation_quality_sample.json`.

The output is uploaded to Label Studio, where multiple annotators label each prompt:

- Maintains intention of category (Yes / No)
- Is grammatical and understandable (Yes / No)
- Is harmful (Yes / No)

The exported annotations are aggregated:

```bash
python RQ2/scripts/annotation_extraction/aggregate_prompts.py RQ2/annotations/prompts/annotated_prompts_unaggregated.csv RQ2/annotations/prompts/annotated_prompts2_unaggregated.csv RQ2/outputs/annotated_prompts.csv
python RQ2/scripts/annotation_extraction/fleiss_kappa_prompts.py RQ2/outputs/annotated_prompts.csv
```

Outputs to `RQ2/outputs/annotated_prompts.csv` and `RQ2/outputs/prompts_fleiss_kappa.csv`.

### 9. Human Annotation (Refusal to Comply Cases)
```bash
python RQ2/scripts/utils/build_refuse_comply_tasks.py
```

The output is uploaded to Label Studio where multiple annotators label each case:

- Mutated prompt preserves the harmful intent of the seed prompt (Yes / No / Unclear)
- Model response is actually compliant with the harmful request (Yes / No / Unclear)

The exported annotations are aggregated:

```bash
python RQ2/scripts/annotation_extraction/aggregate_refuse_comply.py RQ2/annotations/refuse_comply/refuse_to_comply_unaggregated.csv RQ2/outputs/annotated_refuse_comply.csv
python RQ2/scripts/annotation_extraction/fleiss_kappa_refuse_comply.py RQ2/outputs/annotated_refuse_comply.csv
```

Outputs to `RQ2/outputs/annotated_refuse_comply.csv` and `RQ2/outputs/refuse_comply_fleiss_kappa.csv`.

### 10. Human Annotation (Claim to Prompt Mapping)
```bash
python RQ2/scripts/annotation_extraction/build_claim_prompt_mapping_tasks.py
```

Creates one task per testable claim from RQ1 (`RQ1/outputs/annotated_claims.csv`), listing the seed prompts from the claim's category as candidates.
Outputs to `RQ2/annotations/claim_to_prompt/tasks.json`.

The output is uploaded to Label Studio where multiple annotators label each claim:

- Which seed prompts are relevant to the claim (any number)

The exported annotations are aggregated:

```bash
python RQ2/scripts/annotation_extraction/aggregate_claim_to_prompt.py RQ2/annotations/claim_to_prompt/claim_to_prompt_unaggregated.csv RQ2/annotations/claim_to_prompt/claim_to_prompt2_unaggregated.csv RQ2/outputs/claim_to_prompt.csv RQ2/outputs/prompt_to_claims.csv
python RQ2/scripts/annotation_extraction/fleiss_kappa_claim_to_prompt.py RQ2/outputs/claim_to_prompt.csv
```

Outputs to `RQ2/outputs/claim_to_prompt.csv`, `RQ2/outputs/prompt_to_claims.csv` and `RQ2/outputs/claim_to_prompt_fleiss_kappa.csv`.

### 11. Human Annotation (Refusal Oracle)

This uses the same task list as Step 10 and is uploaded to Label Studio where multiple annotators label each claim:

- Can be tested using a refusal oracle (Yes / No / Unclear)
- One sentence description of why or why not

The exported annotations are aggregated: 

```bash
python RQ2/scripts/annotation_extraction/aggregate_refusal_evidence.py RQ2/annotations/refusal_evidence/refuse_oracle_unaggregated.csv RQ2/outputs/refusal_evidence.csv
python RQ2/scripts/annotation_extraction/fleiss_kappa_refusal_evidence.py RQ2/outputs/refusal_evidence.csv
```

Outputs to `RQ2/outputs/refusal_evidence.csv` and `RQ2/outputs/refusal_evidence_fleiss_kappa.csv`.

### 12. Tables and Data Outputs
```bash
python RQ2/scripts/utils/build_seed_failure_rate.py
python RQ2/scripts/utils/build_response_transitions.py
python RQ2/scripts/utils/claims_overview.py
```

Outputs to `RQ2/data/evaluation/filtered_harmful/` (`seed_failure_summary.csv`, `seed_failure_distribution.csv`, `seed_level_failure_table.csv` `claims_overview.csv` and `response_transitions.csv`).

## Reproducing RQ3

### 1. Build Mutation Tables
```bash
python RQ3/scripts/build_mutation_tables.py
```

Joins each approved mutated prompt with its ensemble verdict and the verdict on its original prompt.
Outputs to `RQ3/data/analysis/<category>_mutation_all_experiment_{claude,gpt}.json`.

### 2. Operator Effectiveness Report
```bash
python RQ3/scripts/operator_effectiveness.py > RQ3/outputs/operator_effectiveness_report.txt
```

Reports original-prompt outcomes, valid mutations and outcome rates by operator and category, Refuse->Comply rates by operator and category (all prompts and harmful prompts only), and seeds with at least one flip with 95% confidence intervals.
Outputs to `RQ3/outputs/operator_effectiveness_report.txt`.
