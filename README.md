# Stage-Aware LLM Resource Allocation

Reproducibility artifact for an empirical study of stage-aware resource allocation in multi-stage LLM workflows.

## Overview

This repository contains code, frozen manifests, pre-specified analysis inputs, selected result artifacts, figure-generation scripts, and public releases for a study of budgeted multi-stage LLM workflows.

The proposed controller is a **Reserve-Release-Reallocate (RRR)** policy that:

- protects minimum budget required for reachable downstream stages,
- releases reservations for stages that become unreachable,
- returns unused capacity,
- reallocates available budget while preserving the hard task-level budget invariant.

The confirmatory study uses a quality guardrail: RRR is not required to outperform the legacy policy on accuracy, but it must preserve reliable correctness within a pre-specified non-inferiority margin.

## Study components

### 1. Pilot60

A frozen 60-run engineering and instrumentation study used only for pre-confirmatory mechanism validation.

- 60 runs
- 3 open-weight model families
- 4 allocation policies
- QuixBugs and HotpotQA
- no confirmatory inference

### 2. Confirmatory study

A frozen 4,000-run evaluation with a design and analysis plan defined before confirmatory model runs.

- 4,000 completed runs
- 3 open-weight model families
- primary comparison: Legacy static allocation vs RRR
- primary benchmark: HotpotQA
- primary budget condition: transition
- 100 HotpotQA tasks
- 3 repetitions per task/model/policy in the primary comparison
- paired task-level bootstrap with 10,000 resamples
- primary guardrail: reliable correctness
- absolute non-inferiority margin: 5 percentage points

The raw confirmatory results were frozen before outcome analysis.

## Primary confirmatory result

The pre-specified primary endpoint was **reliable correctness**. Repetitions were aggregated within task before task-level inference, and model-level effects were macro-averaged across the three models.

| Metric | Legacy | RRR | Absolute change | 95% CI for change | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Reliable correctness | 31.00% | 31.33% | +0.33 pp | [0.00, +1.00] pp | Non-inferior; not superior |

The lower confidence bound is above the pre-specified -5 pp non-inferiority margin, so the non-inferiority guardrail passes. The lower bound is not above zero, so superiority is **not** established.

### Reliable correctness by model

| Model | Legacy | RRR | Change |
| --- | ---: | ---: | ---: |
| Qwen/Qwen3-8B | 36.00% | 36.00% | 0.00 pp |
| google/gemma-4-E4B-it | 32.00% | 32.00% | 0.00 pp |
| meta-llama/Llama-3.1-8B-Instruct | 25.00% | 26.00% | +1.00 pp |

### Primary resource measurements

| Resource metric | Legacy | RRR | Change |
| --- | ---: | ---: | ---: |
| Mean consumed tokens | 1388.17 | 1389.27 | +1.10 |
| Mean unused capacity | 1341.21 | 1340.10 | -1.10 |
| Mean provider calls | 3.5467 | 3.5500 | +0.0033 |
| Downstream-stage completion | 91.00% | 91.33% | +0.33 pp |
| Structural-shortfall rate | 9.00% | 8.33% | -0.67 pp |
| Reservation-shortfall rate | 0.00% | 7.33% | +7.33 pp |

At the primary transition-budget condition, RRR preserved quality but did **not** produce a material reduction in token consumption. Resource and stage-reachability effects were small and model-dependent.

## Models

- `Qwen/Qwen3-8B`
- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-4-E4B-it`

These are referred to as **open-weight models**; their individual licenses and usage terms remain those of their respective model providers.

## Policies

- Legacy static allocation
- Greedy allocation
- Fixed reservation
- Proposed Reserve-Release-Reallocate (RRR) policy

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

**Figure interpretation**

- **Figure 2** is the primary cross-model confirmatory quality result.
- **Figure 3** reports primary cross-model resource measurements.
- **Figure 1** is a post-hoc exploratory operating-envelope diagnostic and must not be interpreted as confirmatory causal evidence.

## Public result artifacts

### Pilot60

- `FLLM_PILOT60_RESULTS_V4.zip`
- `FLLM_PILOT60_RESULTS_V4_FREEZE_RECORD.txt`

Release: [pilot60-v4](https://github.com/emadhammami/stage-aware-llm-resource-allocation/releases/tag/pilot60-v4)

### Confirmatory v1

- `FLLM_CONFIRMATORY_RESULTS_V1_1C178171.zip`
- `FLLM_CONFIRMATORY_RESULTS_V1_FREEZE_RECORD.txt`
- `FLLM_CONFIRMATORY_ANALYSIS_V1_FREEZE_RECORD.txt`
- `results/stage_aware_confirmatory_v1/confirmatory_analysis_v1.json`

Release: [confirmatory-v1](https://github.com/emadhammami/stage-aware-llm-resource-allocation/releases/tag/confirmatory-v1)

Frozen release hashes:

- raw confirmatory archive: `A62340C2997ED9DBAC7622B4F5830B86451997227744D051B5B4F10CA8FF2AF9`
- confirmatory analysis release asset: `3F21FDA9559D32A03B5A0CE7BA0DC4D8CC38B82D64454EC87D88A4D7E43B74A4`
- raw events: `1C17817145CD64276D88096EE4559097E95A729E50B306DBA30C8075C0DD3E71`

### Post-hoc exploratory artifact

- `results/stage_aware_confirmatory_v1/exploratory_operating_envelope_v1.json`

This artifact is explicitly exploratory and is not part of the pre-specified confirmatory inference.

## Reproducibility

### Install

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Public CI-equivalent checks

```bash
ruff check .
python -m pytest --basetemp=.pytest_tmp -m "not integration" --ignore=tests/stage_aware/test_artifact.py --ignore=tests/stage_aware/test_phase0_pipeline.py
```

Tests marked `integration` require downloaded benchmark data and are intentionally excluded from the self-contained public CI job. The two explicitly ignored Phase-0 tests depend on a legacy artifact that is not distributed as part of this public release.

### Reproduce the frozen confirmatory analysis

Download and extract the `confirmatory-v1` release archive, then run the frozen analyzer against the extracted raw events file:

```bash
python -m analysis.stage_aware.confirmatory_analyzer_v1 --events <path-to-extracted-events.jsonl> --output reproduced_confirmatory_analysis_v1.json
```

The analyzer verifies the frozen manifest hash, analysis-plan hash, and raw-events SHA256 before producing output. A mismatch causes the analysis to stop rather than silently analyze modified inputs.

### Verify a release artifact

Linux/macOS:

```bash
sha256sum FLLM_CONFIRMATORY_RESULTS_V1_1C178171.zip
```

PowerShell:

```powershell
Get-FileHash .\FLLM_CONFIRMATORY_RESULTS_V1_1C178171.zip -Algorithm SHA256
```

## Interpretation boundary

The confirmatory result supports the claim that RRR **preserved reliable correctness within the pre-specified non-inferiority margin** at the primary transition-budget condition.

It does **not** support a claim of general accuracy superiority or general token-efficiency superiority.

The severe-budget and broader operating-envelope observations are useful for understanding where stage-aware reservation may help or become counterproductive, but those subgroup observations are post-hoc exploratory results and should be presented as such.

## License

See `LICENSE`.
