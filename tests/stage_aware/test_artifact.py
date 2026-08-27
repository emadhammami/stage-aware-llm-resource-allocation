from __future__ import annotations

from pathlib import Path

import pytest

from analysis.stage_aware.artifact import load_publication_artifact, sha256_file
from analysis.stage_aware.config import load_config
from analysis.stage_aware.types import Provenance

FINAL_CORE_ARTIFACT = Path("final_core_results.zip")
pytestmark = pytest.mark.skipif(
    not FINAL_CORE_ARTIFACT.exists(),
    reason="optional Phase-0 provenance artifact final_core_results.zip is not present",
)

def test_publication_artifact_is_complete_exact_and_read_only():
    artifact = "final_core_results.zip"
    config = load_config("research/stage_aware/phase0_config.yaml")
    checksum_before = sha256_file(artifact)
    bundle = load_publication_artifact(artifact, config)
    checksum_after = sha256_file(artifact)
    assert checksum_before == checksum_after == config.artifact_sha256
    assert len(bundle.rows) == 240
    assert len(bundle.runs) == 240
    assert len(bundle.canonical_runs) == 40
    assert sum(len(run.calls) for run in bundle.runs) == 637
    assert all(
        call.prompt_provenance == Provenance.OBSERVED_EXACT
        and call.usage_provenance == Provenance.OBSERVED_EXACT
        for run in bundle.runs
        for call in run.calls
    )


def test_finish_reason_absence_is_reported_as_missing():
    config = load_config("research/stage_aware/phase0_config.yaml")
    bundle = load_publication_artifact("final_core_results.zip", config)
    metric = next(
        row for row in bundle.quality_metrics if row["metric"] == "finish_reason_records"
    )
    assert metric["value"] == 0
    assert metric["provenance"] == Provenance.MISSING.value
