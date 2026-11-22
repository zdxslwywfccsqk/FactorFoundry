from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BacktestConfig


@dataclass
class BacktestResult:
    metrics: dict[str, float | int]
    returns: pd.Series
    gross_returns: pd.Series
    ic_series: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame


def forward_returns(
    close: pd.DataFrame,
    horizon: int,
    execution_lag: int = 0,
) -> pd.DataFrame:
    """Return the close-to-close return from t+lag to t+lag+horizon."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if execution_lag < 0:
        raise ValueError("execution_lag must be non-negative")
    entry = close.shift(-execution_lag)
    exit_price = close.shift(-(execution_lag + horizon))
    return exit_price.div(entry).sub(1.0).replace([np.inf, -np.inf], np.nan)


def eligible_signal_dates(
    index: pd.Index,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    horizon: int,
    execution_lag: int,
) -> pd.DatetimeIndex:
    """Select signal dates whose realised label ends inside the requested split."""
    dates = pd.DatetimeIndex(index)
    positions = np.arange(len(dates))
    if start is not None:
        start_position = int(dates.searchsorted(pd.Timestamp(start), side="left"))
        positions = positions[positions >= start_position]
    if end is not None:
        end_position = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
        positions = positions[positions + execution_lag + horizon <= end_position]
    return dates[positions[::horizon]]


def _periods_per_year(horizon: int) -> float:
    return 252.0 / float(horizon)


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = wealth.div(wealth.cummax()).sub(1.0)
    return float(drawdown.min())


def _geometric_annual_return(returns: pd.Series, periods_per_year: float) -> float:
    if returns.empty:
        return np.nan
    wealth = float((1.0 + returns.fillna(0.0)).prod())
    return wealth ** (periods_per_year / len(returns)) - 1.0


def _transaction_cost(
    weights: pd.Series,
    previous_weights: pd.Series | None,
    config: BacktestConfig,
) -> tuple[float, dict[str, float]]:
    previous = (
        previous_weights.reindex(weights.index).fillna(0.0)
        if previous_weights is not None
        else pd.Series(0.0, index=weights.index)
    )
    delta = weights.sub(previous)
    buys = float(delta.clip(lower=0.0).sum())
    sells = float((-delta.clip(upper=0.0)).sum())
    turnover = buys + sells
    short_exposure = float((-weights.clip(upper=0.0)).sum())

    if config.commission_bps + config.stamp_duty_bps + config.spread_bps + config.impact_bps > 0:
        return turnover, {
            "commission": config.commission_bps * (buys + sells) / 10000.0,
            "stamp_duty": config.stamp_duty_bps * sells / 10000.0,
            "spread": config.spread_bps * (buys + sells) / 10000.0,
            "impact": config.impact_bps * (buys + sells) / 10000.0,
            "borrow": config.borrow_bps_annual
            * short_exposure
            * config.horizon
            / 252.0
            / 10000.0,
        }
    return turnover, {
        "all_in": turnover * config.cost_bps_per_side / 10000.0,
    }


def _empty_result() -> BacktestResult:
    empty = pd.Series(dtype=float)
    return BacktestResult(
        metrics={
            "n_periods": 0,
            "mean_ic": np.nan,
            "icir": np.nan,
            "net_sharpe": np.nan,
            "net_ann_return": np.nan,
            "max_drawdown": np.nan,
        },
        returns=empty,
        gross_returns=empty,
        ic_series=empty,
        turnover=empty,
        weights=pd.DataFrame(),
    )


def _record_liquidation(
    dt: pd.Timestamp,
    all_symbols: pd.Index,
    previous_weights: pd.Series,
    config: BacktestConfig,
    gross_values: list[float],
    turnover_values: list[float],
    cost_values: dict[str, list[float]],
    n_names_values: list[int],
    coverage_values: list[float],
    long_exposure_values: list[float],
    short_exposure_values: list[float],
    return_dates: list[pd.Timestamp],
    weight_rows: dict[pd.Timestamp, pd.Series],
) -> pd.Series:
    weights = pd.Series(0.0, index=all_symbols, dtype=float)
    turnover, cost_components = _transaction_cost(weights, previous_weights, config)
    gross_values.append(0.0)
    turnover_values.append(turnover)
    for name, value in cost_components.items():
        cost_values.setdefault(name, []).append(value)
    return_dates.append(pd.Timestamp(dt))
    weight_rows[pd.Timestamp(dt)] = weights
    n_names_values.append(0)
    coverage_values.append(0.0)
    long_exposure_values.append(0.0)
    short_exposure_values.append(0.0)
    return weights


def evaluate_factor(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    config: BacktestConfig,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    tradable: pd.DataFrame | None = None,
) -> BacktestResult:
    factor = factor.replace([np.inf, -np.inf], np.nan)
    fwd_ret = fwd_ret.replace([np.inf, -np.inf], np.nan)
    dates = eligible_signal_dates(
        factor.index,
        start=start,
        end=end,
        horizon=config.horizon,
        execution_lag=config.execution_lag,
    )

    gross_values: list[float] = []
    ic_values: list[float] = []
    ic_dates: list[pd.Timestamp] = []
    turnover_values: list[float] = []
    cost_values: dict[str, list[float]] = {}
    n_names_values: list[int] = []
    coverage_values: list[float] = []
    long_exposure_values: list[float] = []
    short_exposure_values: list[float] = []
    return_dates: list[pd.Timestamp] = []
    weight_rows: dict[pd.Timestamp, pd.Series] = {}
    all_symbols = factor.columns
    previous_weights: pd.Series | None = None

    for dt in dates:
        raw_x = factor.loc[dt].copy()
        base_count = int(raw_x.notna().sum())
        x = raw_x.copy()
        if tradable is not None and dt in tradable.index:
            x = x.where(tradable.loc[dt] > 0)
        y = fwd_ret.loc[dt].reindex(x.index) if dt in fwd_ret.index else pd.Series(dtype=float)
        available = pd.concat([x.rename("factor"), y.rename("forward")], axis=1).dropna()

        if len(available) < config.min_names:
            if previous_weights is not None and float(previous_weights.abs().sum()) > 0:
                previous_weights = _record_liquidation(
                    dt,
                    all_symbols,
                    previous_weights,
                    config,
                    gross_values,
                    turnover_values,
                    cost_values,
                    n_names_values,
                    coverage_values,
                    long_exposure_values,
                    short_exposure_values,
                    return_dates,
                    weight_rows,
                )
            continue

        x_avail = available["factor"]
        y_avail = available["forward"]
        weights = pd.Series(0.0, index=all_symbols, dtype=float)
        if config.portfolio_mode == "rank":
            centered = x_avail.rank(pct=True) - 0.5
            denom = float(centered.abs().sum())
            if denom <= 0:
                continue
            active_weights = centered / denom * 2.0
            weights.loc[active_weights.index] = active_weights
            gross = float((active_weights * y_avail).sum())
        elif config.portfolio_mode == "quantile":
            ranks = x_avail.rank(pct=True)
            top = ranks[ranks >= 1.0 - config.quantile]
            bottom = ranks[ranks <= config.quantile]
            if len(top) == 0 or len(bottom) == 0:
                continue
            long_ret = float(y_avail.loc[top.index].mean())
            short_ret = float(y_avail.loc[bottom.index].mean())
            gross = long_ret - short_ret
            weights.loc[top.index] = 1.0 / len(top)
            weights.loc[bottom.index] = -1.0 / len(bottom)
        elif config.portfolio_mode == "long_only":
            ranks = x_avail.rank(pct=True)
            denom = float(ranks.sum())
            if denom <= 0:
                continue
            active_weights = ranks / denom
            weights.loc[active_weights.index] = active_weights
            gross = float((active_weights * y_avail).sum())
        else:
            raise ValueError(f"Unknown portfolio_mode: {config.portfolio_mode}")

        turnover, cost_components = _transaction_cost(weights, previous_weights, config)
        gross_values.append(gross)
        turnover_values.append(turnover)
        for name, value in cost_components.items():
            cost_values.setdefault(name, []).append(value)
        return_dates.append(pd.Timestamp(dt))
        weight_rows[pd.Timestamp(dt)] = weights
        previous_weights = weights
        n_names_values.append(int(len(available)))
        coverage_values.append(float(len(available) / max(1, base_count)))
        long_exposure_values.append(float(weights.clip(lower=0.0).sum()))
        short_exposure_values.append(float((-weights.clip(upper=0.0)).sum()))

        if x_avail.nunique() > 1 and y_avail.nunique() > 1:
            ic = x_avail.corr(y_avail, method="spearman")
            if np.isfinite(ic):
                ic_values.append(float(ic))
                ic_dates.append(pd.Timestamp(dt))

    if not gross_values:
        return _empty_result()

    gross_series = pd.Series(gross_values, index=pd.DatetimeIndex(return_dates), name="gross")
    turnover_series = pd.Series(
        turnover_values, index=pd.DatetimeIndex(return_dates), name="turnover"
    )
    cost_frame = pd.DataFrame(
        {
            name: pd.Series(values, index=pd.DatetimeIndex(return_dates))
            for name, values in cost_values.items()
            if values
        }
    ).fillna(0.0)
    costs = cost_frame.sum(axis=1)
    net_series = (gross_series - costs).rename("net")
    ic_series = pd.Series(
        ic_values, index=pd.DatetimeIndex(ic_dates), dtype=float, name="ic"
    )
    weights_frame = pd.DataFrame(weight_rows).T.fillna(0.0)
    periods = _periods_per_year(config.horizon)

    mean_ic = float(ic_series.mean()) if not ic_series.empty else np.nan
    ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else np.nan
    icir = mean_ic / ic_std * np.sqrt(periods) if ic_std and np.isfinite(ic_std) else np.nan
    ic_tstat = (
        mean_ic / ic_std * np.sqrt(len(ic_series))
        if ic_std and np.isfinite(ic_std)
        else np.nan
    )
    ic_hit_rate = float((ic_series > 0).mean()) if not ic_series.empty else np.nan

    net_mean = float(net_series.mean())
    net_std = float(net_series.std(ddof=1)) if len(net_series) > 1 else np.nan
    gross_mean = float(gross_series.mean())
    gross_std = float(gross_series.std(ddof=1)) if len(gross_series) > 1 else np.nan
    net_sharpe = net_mean / net_std * np.sqrt(periods) if net_std and np.isfinite(net_std) else np.nan
    gross_sharpe = (
        gross_mean / gross_std * np.sqrt(periods)
        if gross_std and np.isfinite(gross_std)
        else np.nan
    )

    if len(ic_series) >= 6:
        chunks = np.array_split(ic_series.to_numpy(), 3)
        signs = [float(np.nanmean(chunk) > 0) for chunk in chunks if np.isfinite(chunk).any()]
        subperiod_sign_ratio = float(np.mean(signs)) if signs else np.nan
    else:
        subperiod_sign_ratio = np.nan

    costs_by_component = {
        name: float(cost_frame[name].mean() * periods) if name in cost_frame else 0.0
        for name in ("commission", "stamp_duty", "spread", "impact", "borrow", "all_in")
    }
    initial_cost = float(costs.iloc[0]) if len(costs) else np.nan
    initial_turnover = float(turnover_series.iloc[0]) if len(turnover_series) else np.nan
    metrics: dict[str, float | int] = {
        "n_periods": int(len(net_series)),
        "avg_names": float(np.mean(n_names_values)),
        "coverage": float(np.mean(coverage_values)),
        "execution_lag": int(config.execution_lag),
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "icir": float(icir) if np.isfinite(icir) else np.nan,
        "ic_tstat": float(ic_tstat) if np.isfinite(ic_tstat) else np.nan,
        "ic_hit_rate": ic_hit_rate,
        "subperiod_ic_sign_ratio": subperiod_sign_ratio,
        "gross_ann_return": _geometric_annual_return(gross_series, periods),
        "gross_ann_vol": gross_std * np.sqrt(periods) if np.isfinite(gross_std) else np.nan,
        "gross_sharpe": float(gross_sharpe) if np.isfinite(gross_sharpe) else np.nan,
        "net_ann_return": _geometric_annual_return(net_series, periods),
        "net_ann_vol": net_std * np.sqrt(periods) if np.isfinite(net_std) else np.nan,
        "net_sharpe": float(net_sharpe) if np.isfinite(net_sharpe) else np.nan,
        "max_drawdown": _max_drawdown(net_series),
        "hit_rate": float((net_series > 0).mean()),
        "avg_turnover": float(turnover_series.mean()),
        "initial_turnover": initial_turnover,
        "initial_cost_return": initial_cost,
        "turnover_annual": float(turnover_series.mean() * periods),
        "cost_drag_annual": float(costs.mean() * periods),
        "commission_drag_annual": costs_by_component["commission"],
        "stamp_duty_drag_annual": costs_by_component["stamp_duty"],
        "spread_drag_annual": costs_by_component["spread"],
        "impact_drag_annual": costs_by_component["impact"],
        "borrow_drag_annual": costs_by_component["borrow"],
        "avg_long_exposure": float(np.mean(long_exposure_values)),
        "avg_short_exposure": float(np.mean(short_exposure_values)),
        "avg_gross_return": gross_mean,
        "net_calmar": (
            float(_geometric_annual_return(net_series, periods) / abs(_max_drawdown(net_series)))
            if _max_drawdown(net_series) < 0
            else np.nan
        ),
    }
    return BacktestResult(
        metrics=metrics,
        returns=net_series,
        gross_returns=gross_series,
        ic_series=ic_series,
        turnover=turnover_series,
        weights=weights_frame,
    )
