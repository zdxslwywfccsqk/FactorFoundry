from __future__ import annotations

from conftest import make_fields

from alpha_foundry.backtest import eligible_signal_dates, evaluate_factor, forward_returns
from alpha_foundry.config import BacktestConfig


def test_perfect_forecast_factor_has_positive_ic_and_return() -> None:
    fields = make_fields(n_days=260, n_symbols=50)
    close = fields["close"]
    config = BacktestConfig(
        horizon=5, quantile=0.1, cost_bps_per_side=10.0, min_names=10
    )
    fwd = forward_returns(close, config.horizon, config.execution_lag)
    result = evaluate_factor(fwd, fwd, config)
    assert result.metrics["mean_ic"] > 0.95
    assert result.metrics["gross_ann_return"] > 0
    assert result.metrics["net_sharpe"] < result.metrics["gross_sharpe"]


def test_reversed_factor_has_negative_return() -> None:
    fields = make_fields(n_days=260, n_symbols=50)
    close = fields["close"]
    config = BacktestConfig(
        horizon=5, quantile=0.1, cost_bps_per_side=10.0, min_names=10
    )
    fwd = forward_returns(close, config.horizon, config.execution_lag)
    result = evaluate_factor(-fwd, fwd, config)
    assert result.metrics["mean_ic"] < -0.95
    assert result.metrics["gross_ann_return"] < 0


def test_forward_returns_uses_execution_lag() -> None:
    fields = make_fields(n_days=30, n_symbols=3)
    close = fields["close"]
    lagged = forward_returns(close, horizon=5, execution_lag=1)
    expected = close.iloc[6] / close.iloc[1] - 1.0
    assert lagged.iloc[0].equals(expected)


def test_signal_dates_do_not_cross_split_end() -> None:
    fields = make_fields(n_days=40, n_symbols=3)
    index = fields["close"].index
    end = index[24]
    dates = eligible_signal_dates(
        index,
        start=index[0],
        end=end,
        horizon=5,
        execution_lag=1,
    )
    assert not dates.empty
    assert dates[-1] <= index[24 - 1 - 5]


def test_initial_long_short_cost_uses_full_l1_turnover() -> None:
    fields = make_fields(n_days=120, n_symbols=30)
    close = fields["close"]
    config = BacktestConfig(
        horizon=5,
        execution_lag=1,
        min_names=10,
        commission_bps=2.5,
        stamp_duty_bps=5.0,
        spread_bps=2.0,
        impact_bps=2.0,
    )
    fwd = forward_returns(close, config.horizon, config.execution_lag)
    result = evaluate_factor(fwd, fwd, config)
    assert result.metrics["initial_turnover"] > 1.9
    assert result.metrics["initial_cost_return"] > 0.0015


def test_long_only_mode_has_no_short_exposure() -> None:
    fields = make_fields(n_days=120, n_symbols=30)
    close = fields["close"]
    config = BacktestConfig(
        horizon=5,
        execution_lag=1,
        min_names=10,
        portfolio_mode="long_only",
    )
    fwd = forward_returns(close, config.horizon, config.execution_lag)
    result = evaluate_factor(fwd, fwd, config)
    assert result.metrics["avg_long_exposure"] > 0.95
    assert result.metrics["avg_short_exposure"] == 0.0
