from __future__ import annotations

from conftest import make_fields

from alpha_foundry.backtest import evaluate_factor, forward_returns
from alpha_foundry.config import BacktestConfig


def test_perfect_forecast_factor_has_positive_ic_and_return() -> None:
    fields = make_fields(n_days=260, n_symbols=50)
    close = fields["close"]
    config = BacktestConfig(horizon=5, quantile=0.1, cost_bps_per_side=10.0, min_names=10)
    fwd = forward_returns(close, config.horizon)
    result = evaluate_factor(fwd, fwd, config)
    assert result.metrics["mean_ic"] > 0.95
    assert result.metrics["gross_ann_return"] > 0
    assert result.metrics["net_sharpe"] < result.metrics["gross_sharpe"]


def test_reversed_factor_has_negative_return() -> None:
    fields = make_fields(n_days=260, n_symbols=50)
    close = fields["close"]
    config = BacktestConfig(horizon=5, quantile=0.1, cost_bps_per_side=10.0, min_names=10)
    fwd = forward_returns(close, config.horizon)
    result = evaluate_factor(-fwd, fwd, config)
    assert result.metrics["mean_ic"] < -0.95
    assert result.metrics["gross_ann_return"] < 0
