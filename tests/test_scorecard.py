from __future__ import annotations

import json
from pathlib import Path

from alpha_foundry.scorecard import score_run


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_run(root: Path, *, strong: bool = True, snapshot: bool = True) -> Path:
    run_dir = root / "run_test"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": "run_test",
        "project": "FactorFoundry",
        "mode": "deterministic",
        "status": "completed",
        "config_hash": "config-hash",
        "data_fingerprint": "data-hash",
        "git_commit": "abc123",
        "holdout_released": True,
        "artifact_count": 4,
    }
    _write_json(run_dir / "manifest.json", manifest)
    if snapshot:
        _write_json(
            run_dir / "run_config.json",
            {
                "project": {"seed": 7},
                "backtest": {
                    "execution_lag_bars": 1,
                    "cost_bps_per_side": 10.0,
                },
            },
        )

    pbo = 0.10 if strong else 0.52
    q_value = 0.05 if strong else 0.19
    dsr = 0.95 if strong else 0.13
    ci_low = 0.25 if strong else -0.10
    summary = {
        "run_id": "run_test",
        "artifact_count": 4,
        "holdout_released": True,
        "critic": {
            "n_candidates": 10,
            "n_accepted": 1,
            "decisions": [
                {
                    "factor_id": "good_factor",
                    "accepted": True,
                    "q_value": q_value,
                    "dsr": dsr,
                },
                *[
                    {
                        "factor_id": f"rejected_{index}",
                        "accepted": False,
                        "q_value": 0.6,
                        "dsr": 0.02,
                    }
                    for index in range(9)
                ],
            ],
            "pbo": {"pbo": pbo, "n_trials": 300},
        },
        "portfolio": {
            "selected_factors": ["good_factor"],
            "validation_metrics": {
                "n_periods": 120,
                "mean_ic": 0.03,
                "icir": 0.8,
                "subperiod_ic_sign_ratio": 1.0,
                "ic_hit_rate": 0.55,
                "net_sharpe": 1.0,
                "turnover_annual": 8.0,
                "sharpe_ci_low": ci_low,
                "sharpe_ci_high": 1.5,
            },
        },
        "holdout": {
            "metrics": {
                "n_periods": 80,
                "mean_ic": 0.02,
                "subperiod_ic_sign_ratio": 1.0,
                "net_sharpe": 0.9,
                "net_ann_return": 0.15,
                "turnover_annual": 9.0,
                "sharpe_ci_low": ci_low,
                "sharpe_ci_high": 1.4,
            }
        },
        "scorecard_evidence": {
            "leakage": {
                "label_boundary_guard": True,
                "no_lookahead_full_prefix": True,
                "label_shuffle_same_universe": True,
                "pit_universe": True,
            },
            "data": {
                "delisting_control": True,
                "t_plus_one": True,
                "limit_control": True,
                "st_control": True,
                "suspension_control": True,
                "selection_missing_window": True,
            },
            "execution": {
                "execution_lag_bars": 1,
                "trade_price": "next_close",
                "implementability": "long_only",
                "borrow_cost_bps": 0.0,
                "capacity_estimate": True,
            },
            "cost": {
                "mode": "turnover_l1",
                "initial_position_charged": True,
                "rebalance_charged": True,
                "components": ["commission", "stamp_duty", "spread", "impact"],
            },
            "statistics": {"holdout_pristine": True},
            "robustness": {
                "cost_sensitivity": [{"bps": 5}, {"bps": 10}, {"bps": 20}],
                "parameter_perturbation": [{"window": 10}, {"window": 20}, {"window": 40}],
                "regime_results": [{"regime": "low_vol"}, {"regime": "high_vol"}],
            },
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "artifacts" / "run_test-audit_report-0004.json",
        {
            "artifact_id": "run_test-audit_report-0004",
            "kind": "audit_report",
            "created_at": "2026-01-01T00:00:04",
            "content_hash": "audit-hash",
            "payload": {"factor_id": "good_factor", "passed": True},
        },
    )
    _write_json(
        run_dir / "artifacts" / "run_test-backtest_report-0003.json",
        {
            "artifact_id": "run_test-backtest_report-0003",
            "kind": "backtest_report",
            "split": "validation",
            "created_at": "2026-01-01T00:00:03",
            "content_hash": "backtest-hash",
            "payload": {
                "factor_id": "portfolio_main",
                "protocol": {"cost_bps_per_side": 10.0, "tradable_filter": True},
            },
        },
    )
    _write_json(
        run_dir / "artifacts" / "run_test-critic_review-0002.json",
        {
            "artifact_id": "run_test-critic_review-0002",
            "kind": "critic_review",
            "created_at": "2026-01-01T00:00:02",
            "content_hash": "critic-hash",
            "payload": summary["critic"],
        },
    )
    _write_json(
        run_dir / "artifacts" / "run_test-portfolio_decision-0001.json",
        {
            "artifact_id": "run_test-portfolio_decision-0001",
            "kind": "portfolio_decision",
            "created_at": "2026-01-01T00:00:01",
            "content_hash": "portfolio-hash",
            "payload": summary["portfolio"],
        },
    )
    return run_dir


def test_complete_run_can_be_certified(tmp_path: Path) -> None:
    result = score_run(_make_run(tmp_path))
    assert result.score == 100.0
    assert result.grade == "S"
    assert result.alpha_candidate
    assert result.certification == "Paper Alpha"
    assert all(gate.passed for gate in result.gates)


def test_weak_statistics_block_alpha_certification(tmp_path: Path) -> None:
    result = score_run(_make_run(tmp_path, strong=False))
    statistical_gate = next(gate for gate in result.gates if gate.gate_id == "G5")
    assert not statistical_gate.passed
    assert not result.alpha_candidate
    assert result.certification == "Benchmark only"
    assert result.grade != "A"


def test_missing_config_snapshot_fails_reproducibility_gate(tmp_path: Path) -> None:
    result = score_run(_make_run(tmp_path, snapshot=False))
    reproducibility_gate = next(gate for gate in result.gates if gate.gate_id == "G1")
    assert not reproducibility_gate.passed
    assert not result.alpha_candidate


def test_markdown_contains_verdict_and_gaps(tmp_path: Path) -> None:
    result = score_run(_make_run(tmp_path, strong=False))
    rendered = result.to_markdown()
    assert "Alpha Evidence Scorecard" in rendered
    assert "Alpha Candidate: NO" in rendered
    assert "Blocking evidence gaps" in rendered


def test_json_output_is_strict_json(tmp_path: Path) -> None:
    result = score_run(_make_run(tmp_path, strong=False))
    rendered = json.dumps(result.to_dict(), allow_nan=False)
    assert '"certification": "Benchmark only"' in rendered
