from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import AppConfig
from .ledger import EventLedger
from .schemas import SplitName, utc_now


class SplitGuardError(PermissionError):
    """Raised when a role attempts to access a locked data split."""


class SplitGuard:
    """Capability gate for train/validation/test access.

    The guard is deliberately boring: it records every access and requires an
    explicit, logged release before the holdout can be evaluated. In a hosted
    deployment the same interface can be backed by filesystem permissions or a
    separate evaluation service.
    """

    def __init__(self, config: AppConfig, run_dir: Path, ledger: EventLedger):
        self.config = config
        self.run_dir = Path(run_dir)
        self.ledger = ledger
        self.unlock_path = self.run_dir / "holdout_unlock.json"
        self._released = self.unlock_path.exists()

    @property
    def holdout_released(self) -> bool:
        return self._released

    def dates(self, split: SplitName | str) -> tuple[str, str]:
        split_value = split if isinstance(split, SplitName) else SplitName(split)
        splits = self.config.splits
        if split_value is SplitName.TRAIN:
            return splits.train_start, splits.train_end
        if split_value is SplitName.VALIDATION:
            return splits.validation_start, splits.validation_end
        return splits.test_start, splits.test_end

    def assert_allowed(
        self,
        split: SplitName | str,
        actor: str,
        purpose: str,
    ) -> SplitName:
        split_value = split if isinstance(split, SplitName) else SplitName(split)
        if split_value is SplitName.TEST and not self._released:
            self.ledger.append(
                "split_access_denied",
                actor,
                {"split": split_value.value, "purpose": purpose},
            )
            raise SplitGuardError(
                "Holdout split is locked. Call release_holdout only after the run is frozen."
            )
        self.ledger.append(
            "split_access",
            actor,
            {"split": split_value.value, "purpose": purpose},
        )
        return split_value

    def release_holdout(self, actor: str, reason: str, force: bool = False) -> dict[str, Any]:
        if self._released and not force:
            raise SplitGuardError("Holdout has already been released for this run.")
        record = {
            "run_id": self.ledger.run_id,
            "released_at": utc_now(),
            "released_by": actor,
            "reason": reason,
            "force": bool(force),
            "test_start": self.config.splits.test_start,
            "test_end": self.config.splits.test_end,
        }
        self.unlock_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._released = True
        self.ledger.append("holdout_released", actor, record)
        return record
