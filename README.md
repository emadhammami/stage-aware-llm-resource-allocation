# Stage-Aware LLM Resource Allocation

Reproducibility artifact for an empirical study of stage-aware resource allocation in multi-stage LLM workflows.

## Status

This repository contains the frozen pre-execution design for the Pilot60 engineering study. Scientific Pilot60 results are not yet included.

## Method

The proposed controller uses a reserve-release-reallocate strategy. It protects minimum resources for reachable downstream stages, returns unused capacity, releases reservations for unreachable stages, and reallocates available capacity while respecting a hard task-level budget.

## Pilot60

The frozen matrix contains:

- 5 tasks
- 4 allocation policies
- 3 open-weight model families
- 1 repetition
- 60 runs

Models:

- Qwen/Qwen3-8B
- meta-llama/Llama-3.1-8B-Instruct
- google/gemma-4-E4B-it

Policies:

- Legacy static allocation
- Greedy allocation
- Fixed reservation
- Prompt-aware reserve-release-reallocate

Benchmarks:

- 3 QuixBugs tasks
- 2 HotpotQA tasks

Exact run identities, revisions, budgets, and policies are stored in `pilot60_manifest.json`.

## Calibration

Initial task budgets use the frozen outcome-blind calibration stored in:

`initial_budget_calibration.json`

The calibration uses resource information rather than correctness or answer outcomes.

## Repository structure

- `workflow_control/` â€” stage-aware controller and runtime
- `benchmark/` â€” benchmark adapters and Pilot60 launcher
- `analysis/stage_aware/` â€” analysis and diagnostics
- `research/stage_aware/` â€” experiment configuration
- `tests/stage_aware/` â€” stage-aware tests
- `tests/workflow_control/` â€” controller and runtime tests
- `agent/` â€” shared workflow components required by the runtime

## Validation

The frozen pre-execution implementation passed:

- pytest: test suite completed successfully; provenance-dependent Phase-0 tests are skipped when the optional source artifact is absent
- Ruff: all checks passed
- mypy: no issues in the configured scientific scope

Pilot60 manifest SHA-256:

`41a1dd14e5b322eea5476d07607c4e1773e2753d783d8a1a7514b383e5236bd2`

## Installation

Core development environment:

`python -m pip install -e ".[dev]"`

Local open-weight model environment:

`python -m pip install -e ".[dev,local-models]"`

Large model weights and local caches are not stored in this repository.

## License

MIT
