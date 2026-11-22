from __future__ import annotations

import json

from alpha_foundry.agentic.artifacts import ArtifactStore
from alpha_foundry.agentic.schemas import ArtifactKind, ArtifactStatus
from alpha_foundry.agentic.workspace import RunWorkspace


def test_artifact_store_roundtrip_and_lineage(tmp_path) -> None:
    workspace = RunWorkspace(tmp_path / "runs", run_id="run_test")
    store = ArtifactStore(workspace.run_dir, workspace.run_id)
    first = store.put(
        ArtifactKind.HYPOTHESIS_CARD,
        {"spec_id": "rev_5", "hypothesis_id": "hyp_rev_5"},
        producer="test",
        status=ArtifactStatus.PROPOSED,
    )
    second = store.put(
        ArtifactKind.FEASIBILITY_REPORT,
        {"spec_id": "rev_5", "feasible": True},
        producer="test",
        parents=[first.artifact_id],
    )
    assert store.get(first.artifact_id).payload["spec_id"] == "rev_5"
    assert store.latest(ArtifactKind.FEASIBILITY_REPORT).artifact_id == second.artifact_id
    lineage = store.lineage(second.artifact_id)
    assert [item.artifact_id for item in lineage] == [first.artifact_id, second.artifact_id]
    assert json.loads((workspace.artifacts_dir / f"{first.artifact_id}.json").read_text())["kind"] == "hypothesis_card"


def test_artifact_content_hash_is_stable(tmp_path) -> None:
    workspace = RunWorkspace(tmp_path / "runs", run_id="run_hash")
    store = ArtifactStore(workspace.run_dir, workspace.run_id)
    one = store.put(ArtifactKind.RUN_PLAN, {"b": 2, "a": 1}, producer="test")
    two = store.put(ArtifactKind.RUN_PLAN, {"a": 1, "b": 2}, producer="test")
    assert one.content_hash == two.content_hash
