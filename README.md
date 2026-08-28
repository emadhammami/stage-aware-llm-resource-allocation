# Stage-Aware LLM Resource Allocation

Reproducibility artifact for an empirical study of stage-aware resource allocation in multi-stage LLM workflows.

## Overview

This repository contains code, frozen manifests, analysis plans, selected result artifacts, and figure-generation scripts for a study of budgeted multi-stage LLM workflows.

The main idea is a stage-aware controller that:
- protects minimum budget required for reachable downstream stages,
- releases reservations for unreachable stages,
- returns unused capacity,
- reallocates available budget subject to a hard task-level limit.

## Study components

### 1. Pilot60
A small frozen engineering study used for pre-confirmatory validation.

Main characteristics:
- 60 runs
- 3 open-weight model families
- 4 allocation policies
- QuixBugs and HotpotQA tasks

### 2. Confirmatory study
A frozen confirmatory evaluation with pre-specified design and analysis.

Main characteristics:
- 4000 runs
- 3 open-weight model families
- primary comparison on HotpotQA
- paired noninferiority analysis for quality
- additional resource and operating-envelope diagnostics

## Models

- `Qwen/Qwen3-8B`
- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-4-E4B-it`

## Policies

- Legacy static allocation
- Greedy allocation
- Fixed reservation
- Proposed stage-aware reserve-release-reallocate policy

## Repository contents

### Frozen design artifacts
- `research/stage_aware/confirmatory_matrix_design_v1.yaml`
- `research/stage_aware/confirmatory_manifest_v1.json`
- `research/stage_aware/confirmatory_analysis_plan_v1.yaml`
- `research/stage_aware/confirmatory_budget_regimes_v1.yaml`
- `research/stage_aware/confirmatory_hotpot_selection_v1.yaml`
- `research/stage_aware/confirmatory_hotpot100_tasks_v1.json`
- `research/stage_aware/confirmatory_reference_budgets_v1.json`

### Analysis and figure scripts
- `analysis/stage_aware/confirmatory_analyzer_v1.py`
- `analysis/stage_aware/make_figure1_operating_envelope.py`
- `analysis/stage_aware/make_figure2_primary_confirmatory.py`
- `analysis/stage_aware/make_remaining_paper_figures.py`

### Frozen figures
- `results/figures/figure1_qwen_operating_envelope.pdf`
- `results/figures/figure2_primary_confirmatory_cross_model.pdf`
- `results/figures/figure3_primary_resources_cross_model.pdf`

PNG versions are also included.

## Public result artifacts

### Pilot60 artifact
- `FLLM_PILOT60_RESULTS_V4.zip`
- `FLLM_PILOT60_RESULTS_V4_FREEZE_RECORD.txt`

### Confirmatory artifact
- `FLLM_CONFIRMATORY_RESULTS_V1_1C178171.zip`
- `FLLM_CONFIRMATORY_RESULTS_V1_FREEZE_RECORD.txt`
- `FLLM_CONFIRMATORY_ANALYSIS_V1_FREEZE_RECORD.txt`

### Derived analysis artifact
- `results/stage_aware_confirmatory_v1/confirmatory_analysis_v1.json`
- `results/stage_aware_confirmatory_v1/exploratory_operating_envelope_v1.json`

## Notes on interpretation

The confirmatory analysis distinguishes:
- confirmatory pre-specified comparisons,
- post hoc exploratory operating-envelope diagnostics.

Exploratory operating-envelope outputs should not be interpreted as confirmatory causal evidence.

## Reproducibility

Important frozen references are tracked through tags and SHA256 hashes in the corresponding freeze-record files.

## License

See `LICENSE`.
