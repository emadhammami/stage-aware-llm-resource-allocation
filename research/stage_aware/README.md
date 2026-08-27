# Online Stage-Aware Budget Control

## Motivation

Conditional multi-stage LLM workflows share a hard task-level token budget across planning,
execution, recovery, and verification. A stage that receives a large output allowance can leave
too little capacity for a reachable downstream prompt and minimum viable output, even when the
workflow route itself is unchanged.

## Research Question

This study examines whether deterministic online reservation, release, return, and reallocation
can preserve feasible downstream execution under a fixed global budget. Allocation policy is the
independent variable; routing, prompts, validation evidence, retry rules, model configuration, and
workflow topology remain controlled.

## Relation to Previous Work

The preceding Evidence-Gated implementation determines which downstream stage should execute from
validation evidence. The present project instead studies how much capacity a current stage may use
while protecting reachable downstream work. It retains the earlier workflow topology and routing
semantics.

## Controller Concept

The proposed controller follows a deterministic cycle:

1. reserve prompt-plus-minimum-output capacity for reachable downstream stages;
2. reconcile the allocation with realized provider input-plus-output usage;
3. release reservations when a branch becomes unreachable;
4. return unused output allowance to the shared pool; and
5. make returned or released capacity available to later stages.

Allocation decisions use no model judgment and do not use correctness labels.

## Allocation Policies

- `legacy_static` uses the publication output caps and no explicit reservation.
- `greedy` gives the current stage all available capacity up to its hard cap.
- `fixed_reserved` protects 25% of the total task budget for a potentially reachable Critic.
- `adaptive_stage_aware` funds the current minimum, protects reachable downstream minima with
  Critic priority, and reconciles capacity at later events.

The stage limits and the material-action rule are frozen in
`research/stage_aware/phase0_config.yaml`.

## Phase 0

Phase 0 is an offline mechanism-feasibility audit. No new model inference is performed in Phase 0.
It reads the complete publication artifact, validates every summary/raw correspondence, and applies
the four allocation policies to the controlled Evidence-Gated 4000-token cohort. Policy replay
stops when a proposed cap cannot contain the historical output; downstream state is then marked
counterfactual-unknown.

## Data Provenance

The immutable publication archive is verified before and after replay. Its recorded implementation
commit is `93ce1a5901bf900033c8cc7c547b8e8926bb7103`. Numeric provenance is explicit:
`observed_exact`, `reconstructed_exact`, `deterministic_estimate`, `missing`, or
`counterfactual_unknown`.

## Reproduction

From the repository root:

```bash
python -m analysis.stage_aware.run_phase0
pytest tests/stage_aware
pytest
ruff check .
```

The replay command regenerates all Phase-0 CSV files, figures, and the checksum manifest from the
publication archive and frozen configuration.

## Outputs

`results/stage_aware_phase0/` contains:

- the versioned allocation and reservation ledgers;
- per-run replay dispositions and policy summaries;
- stage-cost and cap-binding-proxy tables;
- exact and estimated structural-feasibility tables;
- policy-action, starvation, reservation, provenance, and data-quality summaries;
- a manifest with source and generated-file checksums; and
- publication-quality PNG and PDF figures.

## Limitations

Offline replay cannot determine how a changed allocation would alter generated text, program
correctness, validation outcome, Critic decision, retry success, or workflow success. Historical
output fitting a proposed cap is a compatibility check, not proof that the cap was non-binding.
Provider finish reason is absent from the publication telemetry, and structurally estimated future
prompts remain labeled as estimates.
