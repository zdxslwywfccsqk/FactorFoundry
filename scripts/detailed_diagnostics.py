from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alpha_foundry.backtest import evaluate_factor, forward_returns
from alpha_foundry.config import AppConfig
from alpha_foundry.data import load_fields
from alpha_foundry.factors import compute_factor, generate_factor_specs
from alpha_foundry.portfolio import combine_signals, signal_from_factor
from alpha_foundry.stats import (
    bootstrap_sharpe_ci,
    deflated_sharpe,
    probability_of_backtest_overfitting,
)


def yearly_stats(result, periods: float) -> list[dict]:
    rows = []
    for year, group in result.returns.groupby(result.returns.index.year):
        wealth = float((1.0 + group).prod())
        std = float(group.std(ddof=1)) if len(group) > 1 else np.nan
        sharpe = (
            float(group.mean() / std * np.sqrt(periods))
            if std and np.isfinite(std) and std > 0
            else np.nan
        )
        ic = result.ic_series[result.ic_series.index.year == year]
        turnover = result.turnover[result.turnover.index.year == year]
        rows.append(
            {
                "year": int(year),
                "n_periods": int(len(group)),
                "cumulative_return": wealth - 1.0,
                "net_sharpe_annualized": sharpe,
                "mean_ic": float(ic.mean()) if len(ic) else np.nan,
                "hit_rate": float((group > 0).mean()),
                "avg_turnover": float(turnover.mean()),
            }
        )
    return rows


def drawdown_details(result) -> dict:
    wealth = (1.0 + result.returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    trough = drawdown.idxmin()
    peak = wealth.loc[:trough].idxmax()
    return {
        "trough_date": str(trough.date()),
        "peak_date": str(peak.date()),
        "max_drawdown": float(drawdown.min()),
    }


def leg_decomposition(result, fwd_ret: pd.DataFrame) -> dict:
    rows = []
    for dt, weights in result.weights.iterrows():
        forward = fwd_ret.loc[dt].reindex(weights.index).fillna(0.0)
        rows.append(
            (
                float((weights.clip(lower=0) * forward).sum()),
                float((weights.clip(upper=0) * forward).sum()),
            )
        )
    values = np.asarray(rows)
    return {
        "avg_long_leg_5d": float(values[:, 0].mean()),
        "avg_short_leg_5d": float(values[:, 1].mean()),
        "avg_gross_5d": float((values[:, 0] + values[:, 1]).mean()),
    }


def quintile_spread(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    tradable: pd.DataFrame,
    close: pd.DataFrame,
    start: str,
    end: str,
    horizon: int,
) -> dict:
    rows = []
    dates = close.index[(close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))][::horizon]
    for dt in dates:
        x = factor.loc[dt].where(tradable.loc[dt] > 0)
        pair = pd.concat([x.rename("x"), fwd_ret.loc[dt].rename("y")], axis=1).dropna()
        if len(pair) < 50:
            continue
        quintile = pd.qcut(pair["x"].rank(method="first"), 5, labels=False)
        means = pair.groupby(quintile)["y"].mean()
        rows.append([float(means.get(i, np.nan)) for i in range(5)])
    values = np.asarray(rows, dtype=float)
    spread = values[:, 4] - values[:, 0]
    return {
        "n_periods": int(len(values)),
        "mean_forward_return_by_quintile_5d": [float(x) for x in np.nanmean(values, axis=0)],
        "q5_minus_q1_mean_5d": float(np.nanmean(spread)),
        "q5_minus_q1_tstat": float(
            np.nanmean(spread) / np.nanstd(spread, ddof=1) * np.sqrt(len(spread))
        )
        if len(spread) > 2
        else np.nan,
        "q5_minus_q1_hit_rate": float(np.nanmean(spread > 0)),
    }


def universe_funnel(config: AppConfig) -> dict:
    raw = pd.read_parquet(config.paths.raw_dir / "prices" / "prices.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    close = raw.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    amount = raw.pivot_table(index="date", columns="symbol", values="amount", aggfunc="last").sort_index()
    selection = slice(None, pd.Timestamp(config.data.selection_end_date))
    history = close.loc[selection].notna().sum()
    avg_dollar = amount.loc[selection].mean()
    median_price = close.loc[selection].median()
    missing_ratio = close.isna().mean()
    checks = {
        "history_ge_180": history >= config.data.min_history_days,
        "avg_dollar_volume_ge_20m": avg_dollar >= config.data.min_avg_dollar_volume,
        "median_price_ge_1": median_price >= config.data.min_price,
        "missing_ratio_le_0_25": missing_ratio <= config.data.max_missing_ratio,
    }
    funnel = {"raw_symbols": int(close.shape[1])}
    for name, mask in checks.items():
        funnel[f"{name}_count"] = int(mask.sum())
    combined = np.logical_and.reduce([mask.to_numpy() for mask in checks.values()])
    funnel["all_conditions_count"] = int(combined.sum())
    return funnel


def main() -> None:
    config = AppConfig.load(ROOT / "config.json")
    fields, _ = load_fields(config.paths.processed_dir)
    specs = {spec.spec_id: spec for spec in generate_factor_specs()}
    close = fields["close"]
    tradable = fields["tradable"]
    horizon = config.backtest.horizon
    periods = 252.0 / horizon
    fwd_ret = forward_returns(close, horizon, config.backtest.execution_lag)

    validation_returns = {}
    trial_sharpes = []
    for spec_id, spec in specs.items():
        factor = compute_factor(spec, fields)
        result = evaluate_factor(
            factor,
            fwd_ret,
            config.backtest,
            start=config.splits.validation_start,
            end=config.splits.validation_end,
            tradable=tradable,
        )
        validation_returns[spec_id] = result.returns
        std = result.returns.std(ddof=1)
        trial_sharpes.append(
            float(result.returns.mean() / std) if len(result.returns) > 2 and std > 0 else 0.0
        )

    selected_id = "illiquidity_20"
    selected_spec = specs[selected_id]
    selected_factor = compute_factor(selected_spec, fields)
    combined = combine_signals({selected_id: signal_from_factor(selected_factor)})
    split_ranges = {
        "train": (config.splits.train_start, config.splits.train_end),
        "validation": (config.splits.validation_start, config.splits.validation_end),
        "test": (config.splits.test_start, config.splits.test_end),
    }
    portfolio_runs = {
        name: evaluate_factor(
            combined,
            fwd_ret,
            config.backtest,
            start=start,
            end=end,
            tradable=tradable,
        )
        for name, (start, end) in split_ranges.items()
    }

    accepted_ids = ["corrclosevol_20", "illiquidity_20"]
    accepted_runs = {}
    for spec_id in accepted_ids:
        factor = compute_factor(specs[spec_id], fields)
        accepted_runs[spec_id] = {
            name: evaluate_factor(
                factor,
                fwd_ret,
                config.backtest,
                start=start,
                end=end,
                tradable=tradable,
            )
            for name, (start, end) in split_ranges.items()
        }

    ci_scale = math.sqrt(periods)
    bootstrap = {}
    for split in ("validation", "test"):
        interval = bootstrap_sharpe_ci(
            portfolio_runs[split].returns, n_boot=5000, seed=config.seed
        )
        interval["sharpe_ci_low_annualized"] = interval["sharpe_ci_low"] * ci_scale
        interval["sharpe_ci_high_annualized"] = interval["sharpe_ci_high"] * ci_scale
        bootstrap[split] = interval

    metric_frame = pd.read_csv(config.paths.experiments_dir / "factor_results.csv")
    metric_correlation = {}
    for validation_column in (
        "validation_mean_ic",
        "validation_icir",
        "validation_net_sharpe",
    ):
        test_column = validation_column.replace("validation_", "test_")
        pair = metric_frame[[validation_column, test_column]].dropna()
        metric_correlation[validation_column] = {
            "pearson": float(pair[validation_column].corr(pair[test_column])),
            "spearman": float(
                pair[validation_column].corr(pair[test_column], method="spearman")
            ),
            "n": int(len(pair)),
        }

    output = {
        "selected_factor": {
            "factor_id": selected_id,
            "name": selected_spec.name,
            "family": selected_spec.family,
            "category": selected_spec.category,
            "formula": selected_spec.formula,
            "hypothesis": selected_spec.hypothesis,
            "direction": selected_spec.direction,
            "window": selected_spec.window,
        },
        "splits": {
            name: {
                "metrics": portfolio_runs[name].metrics,
                "yearly": yearly_stats(portfolio_runs[name], periods),
                "drawdown": drawdown_details(portfolio_runs[name]),
                "legs": leg_decomposition(portfolio_runs[name], fwd_ret),
            }
            for name in split_ranges
        },
        "bootstrap_sharpe_95ci": bootstrap,
        "deflated_sharpe_detail": deflated_sharpe(
            accepted_runs[selected_id]["validation"].returns, len(specs), trial_sharpes
        ),
        "pbo_detail": probability_of_backtest_overfitting(
            pd.concat(validation_returns, axis=1), seed=config.seed
        ),
        "accepted_return_correlation": {
            "validation": float(
                accepted_runs["corrclosevol_20"]["validation"].returns.corr(
                    accepted_runs["illiquidity_20"]["validation"].returns
                )
            ),
            "test": float(
                accepted_runs["corrclosevol_20"]["test"].returns.corr(
                    accepted_runs["illiquidity_20"]["test"].returns
                )
            ),
        },
        "quintile_spread": {
            "validation": quintile_spread(
                selected_factor,
                fwd_ret,
                tradable,
                close,
                config.splits.validation_start,
                config.splits.validation_end,
                horizon,
            ),
            "test": quintile_spread(
                selected_factor,
                fwd_ret,
                tradable,
                close,
                config.splits.test_start,
                config.splits.test_end,
                horizon,
            ),
        },
        "universe_filter_funnel": universe_funnel(config),
        "cross_factor_metric_correlation": metric_correlation,
    }

    output_path = config.paths.output_dir / "detailed_diagnostics.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
