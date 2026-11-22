from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import RunManifest
from .utils import config_hash, git_commit, to_jsonable


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:6]}"


class RunWorkspace:
    """Filesystem contract for one reproducible research campaign."""

    def __init__(self, runs_dir: Path, run_id: str | None = None):
        self.runs_dir = Path(runs_dir)
        self.run_id = run_id or new_run_id()
        self.run_dir = self.runs_dir / self.run_id
        self.artifacts_dir = self.run_dir / "artifacts"
        self.agent_logs_dir = self.run_dir / "agent_logs"
        self.manifest_path = self.run_dir / "manifest.json"
        self.run_config_path = self.run_dir / "run_config.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.agent_logs_dir.mkdir(parents=True, exist_ok=True)

    def create_manifest(
        self,
        project: str,
        mode: str,
        raw_config: dict[str, Any],
        data_fingerprint: str,
        repo_root: Path,
    ) -> RunManifest:
        self.run_config_path.write_text(
            json.dumps(raw_config, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        manifest = RunManifest(
            run_id=self.run_id,
            project=project,
            mode=mode,
            config_hash=config_hash(raw_config),
            data_fingerprint=data_fingerprint,
            git_commit=git_commit(repo_root),
        )
        self.save_manifest(manifest)
        return manifest

    def save_manifest(self, manifest: RunManifest) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_manifest(self) -> RunManifest:
        return RunManifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))

    def update_manifest(self, **updates: Any) -> RunManifest:
        manifest = self.load_manifest()
        payload = manifest.model_dump()
        payload.update(to_jsonable(updates))
        updated = RunManifest.model_validate(payload)
        self.save_manifest(updated)
        return updated

    def add_stage(self, name: str, status: str, detail: dict[str, Any] | None = None) -> None:
        manifest = self.load_manifest()
        stages = list(manifest.stages)
        entry = {"name": name, "status": status, "detail": to_jsonable(detail or {})}
        if status != "running" and stages and stages[-1].get("name") == name and stages[-1].get("status") == "running":
            stages[-1] = entry
        else:
            stages.append(entry)
        self.update_manifest(stages=stages)

    def finish(self, status: str = "completed", summary: dict[str, Any] | None = None) -> RunManifest:
        manifest = self.load_manifest()
        return self.update_manifest(
            status=status,
            finished_at=datetime.now(timezone.utc).isoformat(),
            summary=to_jsonable(summary or manifest.summary),
        )
