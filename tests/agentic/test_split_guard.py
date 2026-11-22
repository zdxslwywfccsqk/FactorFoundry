from __future__ import annotations

import pytest

from alpha_foundry.agentic.ledger import EventLedger
from alpha_foundry.agentic.splits import SplitGuard, SplitGuardError


def test_holdout_is_locked_until_explicit_release(app_config, tmp_path) -> None:
    ledger = EventLedger(tmp_path, "run_guard")
    guard = SplitGuard(app_config, tmp_path, ledger)
    with pytest.raises(SplitGuardError):
        guard.assert_allowed("test", actor="critic", purpose="peek")
    assert any(event["event_type"] == "split_access_denied" for event in ledger.query())
    record = guard.release_holdout(actor="orchestrator", reason="frozen selection")
    assert record["released_by"] == "orchestrator"
    assert guard.assert_allowed("test", actor="evaluator", purpose="final evaluation").value == "test"
    with pytest.raises(SplitGuardError):
        guard.release_holdout(actor="orchestrator", reason="again")
    assert (tmp_path / "holdout_unlock.json").exists()
