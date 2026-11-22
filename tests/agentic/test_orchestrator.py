from __future__ import annotations

import json

from conftest import make_fields

from alpha_foundry.agentic.orchestrator import ResearchOrchestrator
from alpha_foundry.agentic.schemas import ArtifactKind
from alpha_foundry.agentic.session import ResearchSession
from alpha_foundry.agentic.workspace import RunWorkspace


def test_deterministic_agentic_run_end_to_end(app_config) -> None:
    workspace = RunWorkspace(app_config.paths.runs_dir, run_id="run_e2e")
    workspace.create_manifest(
        project=app_config.project_name,
        mode="deterministic",
        raw_config=app_config.raw,
        data_fingerprint="synthetic",
        repo_root=app_config.root,
    )
    session = ResearchSession(
        app_config,
        make_fields(n_days=320, n_symbols=40),
        {"data_fingerprint": "synthetic", "n_selected_symbols": 40},
        run_id=workspace.run_id,
        run_dir=workspace.run_dir,
    )
    orchestrator = ResearchOrchestrator(
        app_config,
        workspace,
        session,
        mode="deterministic",
        factor_limit=4,
        release_holdout=True,
    )
    report_path, summary_path = orchestrator.run()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert report_path.exists()
    assert summary["holdout_released"] is True
    assert summary["artifact_count"] > 10
    assert session.store.latest(ArtifactKind.HOLDOUT_REPORT) is not None
    assert session.store.latest(ArtifactKind.PORTFOLIO_DECISION) is not None
    manifest = json.loads((workspace.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"


def test_mini_agent_mode_falls_back_without_credentials(app_config) -> None:
    workspace = RunWorkspace(app_config.paths.runs_dir, run_id="run_fallback")
    workspace.create_manifest(
        project=app_config.project_name,
        mode="mini-agent",
        raw_config=app_config.raw,
        data_fingerprint="synthetic",
        repo_root=app_config.root,
    )
    session = ResearchSession(
        app_config,
        make_fields(n_days=320, n_symbols=40),
        {"data_fingerprint": "synthetic"},
        run_id=workspace.run_id,
        run_dir=workspace.run_dir,
    )
    orchestrator = ResearchOrchestrator(
        app_config,
        workspace,
        session,
        mode="mini-agent",
        factor_limit=2,
        release_holdout=False,
    )
    report_path, _ = orchestrator.run()
    manifest = json.loads((workspace.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert report_path.exists()
    assert manifest["mode"] == "mini-agent_fallback"
