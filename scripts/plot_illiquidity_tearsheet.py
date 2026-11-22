from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy.stats import kurtosis, norm, skew

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alpha_foundry.backtest import eligible_signal_dates, evaluate_factor, forward_returns
from alpha_foundry.config import AppConfig
from alpha_foundry.data import load_fields
from alpha_foundry.factors import compute_factor, generate_factor_specs
from alpha_foundry.portfolio import signal_from_factor

FACTOR_ID = "illiquidity_20"
PERIODS_PER_YEAR = 252.0 / 5.0
SPLIT_COLORS = {
    "Train": "#d9d9d9",
    "Validation": "#9ecae1",
    "Test": "#fdae6b",
}


def compound(values: pd.Series) -> float:
    return float(np.prod(1.0 + values.to_numpy(dtype=float)) - 1.0)


def realized_period_returns(
    result, close_index: pd.DatetimeIndex, horizon: int
) -> pd.Series:
    positions = close_index.get_indexer(result.returns.index)
    valid = positions >= 0
    if not valid.any():
        return result.returns.copy()
    realized_index = close_index[
        np.minimum(positions[valid] + horizon, len(close_index) - 1)
    ]
    series = pd.Series(
        result.returns.to_numpy(dtype=float)[valid],
        index=realized_index,
        name="realized_return",
    )
    return series.groupby(level=0).apply(compound).sort_index()


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def extra_metrics(returns: pd.Series) -> dict[str, float]:
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) < 3:
        return {}
    downside = np.minimum(values, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside**2)))
    positives = values[values > 0]
    negatives = values[values < 0]
    var95 = float(np.quantile(values, 0.05))
    cvar95 = float(values[values <= var95].mean()) if (values <= var95).any() else np.nan
    profit_factor = (
        float(positives.sum() / abs(negatives.sum())) if negatives.size else np.inf
    )
    return {
        "sortino": (
            float(values.mean() / downside_dev * np.sqrt(PERIODS_PER_YEAR))
            if downside_dev > 0
            else np.nan
        ),
        "skew": float(skew(values, bias=False)),
        "excess_kurtosis": float(kurtosis(values, fisher=True, bias=False)),
        "var95_5d": var95,
        "cvar95_5d": cvar95,
        "profit_factor": profit_factor,
        "best_5d": float(values.max()),
        "worst_5d": float(values.min()),
    }


def yearly_table(result, close_index: pd.DatetimeIndex, horizon: int) -> dict[int, dict]:
    realized = realized_period_returns(result, close_index, horizon)
    rows: dict[int, dict] = {}
    for year in sorted(realized.index.year.unique()):
        ret = realized[realized.index.year == year]
        wealth = (1.0 + ret.fillna(0.0)).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        std = float(ret.std(ddof=1)) if len(ret) > 1 else np.nan
        ic = result.ic_series[result.ic_series.index.year == year]
        turnover = result.turnover[result.turnover.index.year == year]
        rows[int(year)] = {
            "n_periods": int(len(ret)),
            "return": compound(ret),
            "ann_vol": float(std * np.sqrt(PERIODS_PER_YEAR)) if np.isfinite(std) else np.nan,
            "sharpe": (
                float(ret.mean() / std * np.sqrt(PERIODS_PER_YEAR))
                if std and np.isfinite(std) and std > 0
                else np.nan
            ),
            "max_drawdown": float(drawdown.min()),
            "mean_ic": float(ic.mean()) if len(ic) else np.nan,
            "hit_rate": float((ret > 0).mean()),
            "ann_turnover": (
                float(turnover.mean() * PERIODS_PER_YEAR)
                if len(turnover)
                else np.nan
            ),
        }
    return rows


def mean_ic_at_horizon(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    tradable: pd.DataFrame,
    start: str,
    end: str,
    horizon: int,
    step: int,
    execution_lag: int,
) -> tuple[float, int]:
    fwd = forward_returns(close, horizon, execution_lag)
    dates = eligible_signal_dates(
        close.index,
        start=start,
        end=end,
        horizon=step,
        execution_lag=execution_lag,
    )
    ics = []
    for dt in dates:
        x = factor.loc[dt].where(tradable.loc[dt] > 0)
        pair = pd.concat([x.rename("x"), fwd.loc[dt].rename("y")], axis=1).dropna()
        if len(pair) < 20 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
            continue
        value = pair["x"].corr(pair["y"], method="spearman")
        if np.isfinite(value):
            ics.append(float(value))
    return (float(np.mean(ics)) if ics else np.nan, len(ics))


def shade_splits(ax, config: AppConfig, labels: bool = False) -> None:
    spans = [
        ("Train", config.splits.train_start, config.splits.train_end),
        ("Validation", config.splits.validation_start, config.splits.validation_end),
        ("Test", config.splits.test_start, config.splits.test_end),
    ]
    for label, start, end in spans:
        ax.axvspan(
            pd.Timestamp(start),
            pd.Timestamp(end),
            color=SPLIT_COLORS[label],
            alpha=0.28,
            linewidth=0,
            zorder=0,
        )
        if labels:
            mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
            ax.text(
                mid,
                1.01,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#666666",
            )


def style_axis(ax) -> None:
    ax.grid(True, color="#dddddd", linewidth=0.5, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#444444")


def format_pct(value: float, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:.{digits}f}%"


def format_num(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def main() -> None:
    config = AppConfig.load(ROOT / "config.json")
    fields, _ = load_fields(config.paths.processed_dir)
    close = fields["close"]
    tradable = fields["tradable"]
    horizon = config.backtest.horizon
    fwd_ret = forward_returns(close, horizon, config.backtest.execution_lag)

    specs = {spec.spec_id: spec for spec in generate_factor_specs()}
    spec = specs[FACTOR_ID]
    factor = compute_factor(spec, fields)
    signal = signal_from_factor(factor)

    ranges = {
        "full": (config.splits.train_start, config.splits.test_end),
        "train": (config.splits.train_start, config.splits.train_end),
        "validation": (config.splits.validation_start, config.splits.validation_end),
        "test": (config.splits.test_start, config.splits.test_end),
    }
    runs = {
        name: evaluate_factor(
            signal,
            fwd_ret,
            config.backtest,
            start=start,
            end=end,
            tradable=tradable,
        )
        for name, (start, end) in ranges.items()
    }
    extras = {name: extra_metrics(result.returns) for name, result in runs.items()}

    full = runs["full"]
    full_returns = full.returns
    net_wealth = (1.0 + full_returns.fillna(0.0)).cumprod()
    gross_wealth = (1.0 + full.gross_returns.fillna(0.0)).cumprod()
    drawdown = net_wealth / net_wealth.cummax() - 1.0
    realized = realized_period_returns(full, close.index, horizon)
    yearly = yearly_table(full, close.index, horizon)

    diagnostics_path = ROOT / "output" / "detailed_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    factor_frame = pd.read_csv(ROOT / "experiments" / "factor_results.csv")
    factor_row = factor_frame.loc[factor_frame["factor_id"].eq(FACTOR_ID)].iloc[0]
    bootstrap = diagnostics["bootstrap_sharpe_95ci"]
    quintile = diagnostics["quintile_spread"]

    ic_decay: dict[str, dict[int, float]] = {}
    for split_name, key in (
        ("train", "train"),
        ("validation", "validation"),
        ("test", "test"),
    ):
        start, end = ranges[key]
        ic_decay[split_name] = {
            h: mean_ic_at_horizon(
                factor,
                close,
                tradable,
                start,
                end,
                h,
                horizon,
                config.backtest.execution_lag,
            )[0]
            for h in (1, 3, 5, 10, 20)
        }

    fig = plt.figure(figsize=(22, 32), facecolor="white")
    gs = fig.add_gridspec(
        8,
        4,
        height_ratios=[1.0, 4.2, 2.6, 2.6, 2.7, 2.7, 2.7, 3.6],
        wspace=0.30,
        hspace=0.42,
    )
    ax_title = fig.add_subplot(gs[0, :])
    ax_equity = fig.add_subplot(gs[1, :3])
    ax_kpi = fig.add_subplot(gs[1, 3])
    ax_dd = fig.add_subplot(gs[2, :2])
    ax_roll = fig.add_subplot(gs[2, 2:])
    ax_month = fig.add_subplot(gs[3, :2])
    ax_year = fig.add_subplot(gs[3, 2:])
    ax_ic = fig.add_subplot(gs[4, :2])
    ax_ic_hist = fig.add_subplot(gs[4, 2])
    ax_ic_decay = fig.add_subplot(gs[4, 3])
    ax_quint = fig.add_subplot(gs[5, 0])
    ax_turn = fig.add_subplot(gs[5, 1])
    ax_dist = fig.add_subplot(gs[5, 2])
    ax_legs = fig.add_subplot(gs[5, 3])
    ax_expo = fig.add_subplot(gs[6, 0])
    ax_beta = fig.add_subplot(gs[6, 1])
    ax_hold = fig.add_subplot(gs[6, 2:])
    ax_metrics = fig.add_subplot(gs[7, :2])
    ax_year_heat = fig.add_subplot(gs[7, 2:])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )

    # Title
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.97,
        f"{FACTOR_ID} — 20-day Low-Liquidity Premium",
        fontsize=24,
        fontweight="bold",
        va="top",
        color="#17202a",
    )
    ax_title.text(
        0.0,
        0.53,
        "Formula: -log(mean(dollar volume, 20))   |   Direction: +1   |   Universe: 155 A-shares   |   Rebalance: 5 trading days   |   Cost: 10 bps per side",
        fontsize=10,
        va="top",
        color="#4d5656",
    )
    ax_title.text(
        0.0,
        0.25,
        "Train: 2019-07-01 to 2021-12-31   |   Validation: 2022-01-01 to 2024-12-31   |   Test: 2025-01-01 to 2026-09-30",
        fontsize=9,
        va="top",
        color="#4d5656",
    )
    ax_title.text(
        0.0,
        0.05,
        "Full-sample metrics include the training and validation periods and are diagnostic only; the locked test set is the out-of-sample evidence.",
        fontsize=8,
        style="italic",
        va="top",
        color="#7f8c8d",
    )

    # Equity curve
    ax_equity.plot(net_wealth.index, net_wealth.values, color="#1f4e79", linewidth=2.1, label="Net")
    ax_equity.plot(
        gross_wealth.index,
        gross_wealth.values,
        color="#7f8c8d",
        linewidth=1.1,
        linestyle="--",
        label="Gross",
    )
    ax_equity.axhline(1.0, color="#999999", linewidth=0.8, linestyle=":")
    shade_splits(ax_equity, config, labels=True)
    ax_equity.set_title("Cumulative performance (5-day rebalanced, net of costs)", loc="left")
    ax_equity.set_ylabel("Wealth (x)")
    ax_equity.legend(loc="upper left", frameon=False)
    ax_equity.text(
        0.99,
        0.04,
        f"Net final: {net_wealth.iloc[-1]:.2f}x\nGross final: {gross_wealth.iloc[-1]:.2f}x",
        transform=ax_equity.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    style_axis(ax_equity)

    # KPI panel
    ax_kpi.axis("off")
    ax_kpi.text(0.02, 0.98, "Key metrics", fontsize=13, fontweight="bold", va="top", color="#17202a")
    m = full.metrics
    e = extras["full"]
    test_metrics = runs["test"].metrics
    kpis = [
        ("Full net ann. return", format_pct(m["net_ann_return"])),
        ("Full net Sharpe", format_num(m["net_sharpe"], 2)),
        ("Full max drawdown", format_pct(m["max_drawdown"])),
        ("Full ann. volatility", format_pct(m["net_ann_vol"])),
        ("Test net ann. return", format_pct(test_metrics["net_ann_return"])),
        ("Test net Sharpe", format_num(test_metrics["net_sharpe"], 2)),
        ("Mean Rank IC", format_num(m["mean_ic"], 3)),
        ("ICIR", format_num(m["icir"], 2)),
        ("Ann. turnover", f"{m['turnover_annual']:.1f}x"),
        ("Cost drag / yr", format_pct(m["cost_drag_annual"], 2)),
    ]
    for idx, (label, value) in enumerate(kpis):
        col = idx % 2
        row = idx // 2
        x = 0.02 + col * 0.50
        y = 0.88 - row * 0.095
        ax_kpi.text(x, y, label, fontsize=7, color="#5d6d7e", va="top")
        ax_kpi.text(x, y - 0.034, value, fontsize=11, fontweight="bold", color="#17202a", va="top")
    ax_kpi.axhline(0.405, xmin=0.02, xmax=0.98, color="#d5d8dc", linewidth=0.8)
    ax_kpi.text(0.02, 0.36, "Statistical evidence", fontsize=9, fontweight="bold", va="top", color="#17202a")
    evidence = [
        f"FDR q-value: {factor_row['q_value']:.3f}",
        f"Deflated Sharpe: {factor_row['dsr']:.3f}",
        f"PBO: {diagnostics['pbo_detail']['pbo']:.3f}",
        f"Label-shuffle p: {factor_row['shuffle_p_value']:.3f}",
        f"Validation Sharpe 95% CI: [{bootstrap['validation']['sharpe_ci_low_annualized']:.2f}, {bootstrap['validation']['sharpe_ci_high_annualized']:.2f}]",
        f"Test Sharpe 95% CI: [{bootstrap['test']['sharpe_ci_low_annualized']:.2f}, {bootstrap['test']['sharpe_ci_high_annualized']:.2f}]",
    ]
    for idx, line in enumerate(evidence):
        ax_kpi.text(0.02, 0.315 - idx * 0.052, line, fontsize=7.2, color="#4d5656", va="top")

    # Drawdown
    ax_dd.fill_between(drawdown.index, drawdown.values, 0, color="#c0392b", alpha=0.32)
    ax_dd.plot(drawdown.index, drawdown.values, color="#c0392b", linewidth=0.9)
    shade_splits(ax_dd, config)
    trough = drawdown.idxmin()
    ax_dd.scatter([trough], [drawdown.min()], color="#7b241c", s=18, zorder=3)
    ax_dd.annotate(
        f"Max DD {drawdown.min():.1%}\n{trough.date()}",
        xy=(trough, drawdown.min()),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=7,
        color="#7b241c",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#e6b0aa", "alpha": 0.9},
    )
    ax_dd.set_title("Drawdown (net)", loc="left")
    ax_dd.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_dd.set_ylabel("Drawdown")
    style_axis(ax_dd)

    # Rolling Sharpe and volatility
    rolling_window = int(round(252 / horizon))
    rolling_sharpe = (
        full_returns.rolling(rolling_window).mean()
        / full_returns.rolling(rolling_window).std()
        * np.sqrt(PERIODS_PER_YEAR)
    )
    rolling_vol = full_returns.rolling(rolling_window).std() * np.sqrt(PERIODS_PER_YEAR)
    ax_roll.plot(rolling_sharpe.index, rolling_sharpe.values, color="#1f4e79", linewidth=1.7, label="Rolling 1Y Sharpe")
    ax_roll.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    shade_splits(ax_roll, config)
    ax_roll2 = ax_roll.twinx()
    ax_roll2.plot(rolling_vol.index, rolling_vol.values, color="#e67e22", linewidth=1.2, alpha=0.8, label="Rolling 1Y vol")
    ax_roll2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_roll2.set_ylabel("Annualized volatility", color="#e67e22")
    ax_roll2.tick_params(axis="y", colors="#e67e22")
    ax_roll.set_title("Rolling 1Y Sharpe and volatility", loc="left")
    ax_roll.set_ylabel("Sharpe")
    handles1, labels1 = ax_roll.get_legend_handles_labels()
    handles2, labels2 = ax_roll2.get_legend_handles_labels()
    ax_roll.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False)
    style_axis(ax_roll)

    # Monthly return heatmap
    monthly = realized.groupby([realized.index.year, realized.index.month]).apply(compound)
    monthly.index.names = ["year", "month"]
    monthly_pivot = monthly.unstack("month").reindex(columns=range(1, 13))
    years = sorted(monthly_pivot.index.astype(int).tolist())
    monthly_pivot = monthly_pivot.reindex(years)
    month_values = monthly_pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(month_values)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#f4f6f7")
    im = ax_month.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-0.25, vmax=0.25),
    )
    ax_month.set_xticks(range(len(years)))
    ax_month.set_xticklabels([str(y) for y in years])
    ax_month.set_yticks(range(12))
    ax_month.set_yticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax_month.set_title("Monthly net returns", loc="left")
    for row in range(month_values.shape[0]):
        for col in range(month_values.shape[1]):
            value = month_values[row, col]
            if np.isfinite(value):
                color = "white" if abs(value) >= 0.12 else "#222222"
                ax_month.text(col, row, f"{value:.1%}", ha="center", va="center", fontsize=6.2, color=color)
    ax_month.tick_params(length=0)
    for spine in ax_month.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_month, fraction=0.035, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    cbar.ax.tick_params(labelsize=6)

    # Calendar-year returns
    annual = realized.groupby(realized.index.year).apply(compound)
    year_values = annual.values.astype(float)
    colors = ["#2ecc71" if value >= 0 else "#e74c3c" for value in year_values]
    bars = ax_year.bar(annual.index.astype(str), year_values * 100, color=colors, alpha=0.85)
    for bar, value in zip(bars, year_values, strict=True):
        va = "bottom" if value >= 0 else "top"
        offset = 0.5 if value >= 0 else -0.5
        ax_year.text(
            bar.get_x() + bar.get_width() / 2,
            value * 100 + offset,
            f"{value:.1%}",
            ha="center",
            va=va,
            fontsize=7,
        )
    ax_year.axhline(0, color="#999999", linewidth=0.8)
    ax_year.set_title("Calendar-year net return (2019 and 2026 partial)", loc="left")
    ax_year.set_ylabel("Return")
    ax_year.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    style_axis(ax_year)

    # IC time series
    ic = full.ic_series.dropna()
    ic_colors = np.where(ic.values >= 0, "#2ecc71", "#e74c3c")
    ax_ic.bar(ic.index, ic.values, width=4.0, color=ic_colors, alpha=0.38, linewidth=0)
    ic_roll = ic.rolling(26).mean()
    ax_ic.plot(ic_roll.index, ic_roll.values, color="#1f4e79", linewidth=1.6, label="26-period rolling mean")
    ax_ic.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax_ic.axhline(ic.mean(), color="#17202a", linewidth=1.0, linestyle="--", label=f"Mean IC {ic.mean():.3f}")
    shade_splits(ax_ic, config)
    ax_ic.set_title("Rank IC (5-day forward return)", loc="left")
    ax_ic.set_ylabel("Spearman IC")
    ax_ic.legend(loc="upper left", frameon=False)
    style_axis(ax_ic)

    # IC histogram
    ic_values = ic.to_numpy(dtype=float)
    ax_ic_hist.hist(ic_values, bins=30, density=True, color="#5dade2", alpha=0.62, edgecolor="white")
    x_grid = np.linspace(ic_values.min(), ic_values.max(), 300)
    ax_ic_hist.plot(
        x_grid,
        norm.pdf(x_grid, loc=ic_values.mean(), scale=ic_values.std(ddof=1)),
        color="#1f4e79",
        linewidth=1.3,
    )
    ax_ic_hist.axvline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax_ic_hist.axvline(ic_values.mean(), color="#c0392b", linewidth=1.2, linestyle="--")
    ax_ic_hist.set_title("IC distribution", loc="left")
    ax_ic_hist.set_xlabel("IC")
    ax_ic_hist.set_ylabel("Density")
    ax_ic_hist.text(
        0.02,
        0.97,
        f"Mean {ic_values.mean():.3f}\nICIR {full.metrics['icir']:.2f}\nHit {full.metrics['ic_hit_rate']:.1%}",
        transform=ax_ic_hist.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d5d8dc", "alpha": 0.9},
    )
    style_axis(ax_ic_hist)

    # IC decay
    horizons = [1, 3, 5, 10, 20]
    decay_colors = {"train": "#7f8c8d", "validation": "#2980b9", "test": "#d35400"}
    for split_name in ("train", "validation", "test"):
        values = [ic_decay[split_name][h] for h in horizons]
        ax_ic_decay.plot(
            horizons,
            values,
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            color=decay_colors[split_name],
            label=split_name.capitalize(),
        )
    ax_ic_decay.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax_ic_decay.set_xticks(horizons)
    ax_ic_decay.set_title("IC decay by horizon", loc="left")
    ax_ic_decay.set_xlabel("Forward horizon (trading days)")
    ax_ic_decay.set_ylabel("Mean IC")
    ax_ic_decay.legend(frameon=False, fontsize=6.5)
    style_axis(ax_ic_decay)

    # Quintile returns
    q_val = np.asarray(quintile["validation"]["mean_forward_return_by_quintile_5d"], dtype=float)
    q_test = np.asarray(quintile["test"]["mean_forward_return_by_quintile_5d"], dtype=float)
    x_pos = np.arange(5)
    ax_quint.bar(x_pos - 0.19, q_val * 100, width=0.36, color="#2980b9", alpha=0.78, label="Validation")
    ax_quint.bar(x_pos + 0.19, q_test * 100, width=0.36, color="#d35400", alpha=0.78, label="Test")
    ax_quint.set_xticks(x_pos)
    ax_quint.set_xticklabels([f"Q{i+1}" for i in range(5)])
    ax_quint.axhline(0, color="#999999", linewidth=0.8)
    ax_quint.set_title("Quintile mean 5-day return", loc="left")
    ax_quint.set_ylabel("Return")
    ax_quint.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}%"))
    ax_quint.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax_quint.text(
        0.02,
        0.03,
        f"Q5-Q1: Val {quintile['validation']['q5_minus_q1_mean_5d']:.2%} (t={quintile['validation']['q5_minus_q1_tstat']:.2f})\n"
        f"Test {quintile['test']['q5_minus_q1_mean_5d']:.2%} (t={quintile['test']['q5_minus_q1_tstat']:.2f})",
        transform=ax_quint.transAxes,
        fontsize=6.4,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#d5d8dc", "alpha": 0.9},
    )
    style_axis(ax_quint)

    # Turnover and cost
    ax_turn.bar(full.turnover.index, full.turnover.values * 100, width=4.0, color="#85c1e9", alpha=0.35, linewidth=0)
    ax_turn.plot(
        full.turnover.index,
        full.turnover.rolling(26).mean().values * 100,
        color="#1f618d",
        linewidth=1.4,
        label="26-period average",
    )
    ax_turn.set_title("Turnover and cumulative cost", loc="left")
    ax_turn.set_ylabel("Turnover per 5 days")
    ax_turn.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_turn2 = ax_turn.twinx()
    cumulative_cost = (full.turnover * config.backtest.cost_bps_per_side / 10000.0).cumsum() * 100
    ax_turn2.plot(cumulative_cost.index, cumulative_cost.values, color="#c0392b", linewidth=1.3, label="Cumulative cost")
    ax_turn2.set_ylabel("Cumulative cost", color="#c0392b")
    ax_turn2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax_turn2.tick_params(axis="y", colors="#c0392b")
    handles1, labels1 = ax_turn.get_legend_handles_labels()
    handles2, labels2 = ax_turn2.get_legend_handles_labels()
    ax_turn.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False, fontsize=6.5)
    ax_turn.text(
        0.02,
        0.03,
        f"Ann. turnover {full.metrics['turnover_annual']:.1f}x\nCost drag {full.metrics['cost_drag_annual']:.2%}/yr",
        transform=ax_turn.transAxes,
        fontsize=6.5,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#d5d8dc", "alpha": 0.9},
    )
    style_axis(ax_turn)

    # Return distribution
    ret_values = full_returns.dropna().to_numpy(dtype=float) * 100
    ax_dist.hist(ret_values, bins=36, density=True, color="#58d68d", alpha=0.55, edgecolor="white")
    x_grid = np.linspace(ret_values.min(), ret_values.max(), 300)
    ax_dist.plot(
        x_grid,
        norm.pdf(x_grid, loc=ret_values.mean(), scale=ret_values.std(ddof=1)),
        color="#1e8449",
        linewidth=1.3,
    )
    ax_dist.axvline(ret_values.mean(), color="#17202a", linewidth=1.0, linestyle="--")
    ax_dist.axvline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax_dist.set_title("5-day net return distribution", loc="left")
    ax_dist.set_xlabel("Return")
    ax_dist.set_ylabel("Density")
    ax_dist.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax_dist.text(
        0.02,
        0.97,
        f"Skew {e['skew']:.2f}\nExcess kurt {e['excess_kurtosis']:.2f}\nVaR95 {e['var95_5d']:.2%}\nCVaR95 {e['cvar95_5d']:.2%}",
        transform=ax_dist.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d5d8dc", "alpha": 0.9},
    )
    style_axis(ax_dist)

    # Long/short leg contributions
    weights = full.weights
    aligned_fwd = fwd_ret.reindex(weights.index)
    long_contrib = (weights.clip(lower=0) * aligned_fwd).sum(axis=1)
    short_contrib = (weights.clip(upper=0) * aligned_fwd).sum(axis=1)
    gross_contrib = long_contrib + short_contrib
    ax_legs.plot(long_contrib.index, long_contrib.cumsum().values * 100, color="#1e8449", linewidth=1.5, label="Long leg")
    ax_legs.plot(short_contrib.index, short_contrib.cumsum().values * 100, color="#c0392b", linewidth=1.5, label="Short leg")
    ax_legs.plot(gross_contrib.index, gross_contrib.cumsum().values * 100, color="#17202a", linewidth=1.6, linestyle="--", label="Gross total")
    ax_legs.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    shade_splits(ax_legs, config)
    ax_legs.set_title("Cumulative long/short contribution (gross)", loc="left")
    ax_legs.set_ylabel("Cumulative contribution")
    ax_legs.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_legs.legend(loc="upper left", frameon=False, fontsize=6.5)
    ax_legs.text(
        0.02,
        0.03,
        f"Avg 5d: long {long_contrib.mean():.2%}, short {short_contrib.mean():.2%}",
        transform=ax_legs.transAxes,
        fontsize=6.5,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#d5d8dc", "alpha": 0.9},
    )
    style_axis(ax_legs)

    # Exposure and breadth
    net_exposure = weights.sum(axis=1)
    gross_exposure = weights.abs().sum(axis=1)
    active_names = (weights.abs() > 1e-12).sum(axis=1)
    ax_expo.plot(net_exposure.index, net_exposure.values, color="#8e44ad", linewidth=1.3, label="Net exposure")
    ax_expo.plot(gross_exposure.index, gross_exposure.values, color="#2c3e50", linewidth=1.2, label="Gross exposure")
    ax_expo.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    shade_splits(ax_expo, config)
    ax_expo.set_title("Exposure and breadth", loc="left")
    ax_expo.set_ylabel("Exposure (x NAV)")
    ax_expo2 = ax_expo.twinx()
    ax_expo2.plot(active_names.index, active_names.values, color="#16a085", linewidth=1.0, alpha=0.75, label="Active names")
    ax_expo2.set_ylabel("Active names", color="#16a085")
    ax_expo2.tick_params(axis="y", colors="#16a085")
    handles1, labels1 = ax_expo.get_legend_handles_labels()
    handles2, labels2 = ax_expo2.get_legend_handles_labels()
    ax_expo.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False, fontsize=6.2)
    ax_expo.text(
        0.02,
        0.03,
        f"Avg net {net_exposure.mean():.3f}x, gross {gross_exposure.mean():.2f}x",
        transform=ax_expo.transAxes,
        fontsize=6.3,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#d5d8dc", "alpha": 0.9},
    )
    style_axis(ax_expo)

    # Rolling market beta
    market_daily = fields["market_ret"].iloc[:, 0].astype(float)
    market_5d = (1.0 + market_daily).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0
    market_aligned = market_5d.reindex(full_returns.index)
    rolling_beta = full_returns.rolling(rolling_window).cov(market_aligned) / market_aligned.rolling(rolling_window).var()
    ax_beta.plot(rolling_beta.index, rolling_beta.values, color="#2c3e50", linewidth=1.5, label="Rolling 1Y beta")
    ax_beta.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax_beta.axhline(1, color="#e67e22", linewidth=0.8, linestyle="--", alpha=0.7)
    shade_splits(ax_beta, config)
    ax_beta.set_title("Rolling 1Y market beta", loc="left")
    ax_beta.set_ylabel("Beta vs CSI 300")
    ax_beta.text(
        0.02,
        0.03,
        f"Average beta {rolling_beta.mean():.2f}",
        transform=ax_beta.transAxes,
        fontsize=6.5,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#d5d8dc", "alpha": 0.9},
    )
    style_axis(ax_beta)

    # Latest holdings
    last_date = weights.index[-1]
    last_weights = weights.loc[last_date]
    long_hold = last_weights[last_weights > 0].nlargest(8)
    short_hold = last_weights[last_weights < 0].nsmallest(8)
    holdings = pd.concat([long_hold, short_hold]).sort_values()
    holding_colors = ["#c0392b" if value < 0 else "#1e8449" for value in holdings.values]
    bars = ax_hold.barh(range(len(holdings)), holdings.values * 100, color=holding_colors, alpha=0.82)
    ax_hold.set_yticks(range(len(holdings)))
    ax_hold.set_yticklabels(holdings.index, fontsize=6.3)
    ax_hold.axvline(0, color="#999999", linewidth=0.8)
    ax_hold.set_title(f"Latest portfolio weights — {last_date.date()}", loc="left")
    ax_hold.set_xlabel("Weight (% NAV)")
    ax_hold.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))
    for bar, value in zip(bars, holdings.values, strict=True):
        offset = 0.02 if value >= 0 else -0.02
        ax_hold.text(
            value * 100 + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2%}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=5.8,
        )
    style_axis(ax_hold)

    # Metrics table
    ax_metrics.axis("off")
    ax_metrics.set_title("Performance metrics", loc="left", fontsize=11, fontweight="bold", pad=8)
    metric_specs = [
        ("Net ann. return", lambda r, x: format_pct(r["net_ann_return"])),
        ("Gross ann. return", lambda r, x: format_pct(r["gross_ann_return"])),
        ("Ann. volatility", lambda r, x: format_pct(r["net_ann_vol"])),
        ("Net Sharpe", lambda r, x: format_num(r["net_sharpe"], 2)),
        ("Gross Sharpe", lambda r, x: format_num(r["gross_sharpe"], 2)),
        ("Sortino", lambda r, x: format_num(x.get("sortino", np.nan), 2)),
        ("Max drawdown", lambda r, x: format_pct(r["max_drawdown"])),
        ("Calmar", lambda r, x: format_num(r["net_calmar"], 2)),
        ("Hit rate", lambda r, x: format_pct(r["hit_rate"])),
        ("Ann. turnover", lambda r, x: f"{r['turnover_annual']:.1f}x"),
        ("Cost drag / yr", lambda r, x: format_pct(r["cost_drag_annual"], 2)),
        ("Mean Rank IC", lambda r, x: format_num(r["mean_ic"], 3)),
        ("ICIR", lambda r, x: format_num(r["icir"], 2)),
        ("IC t-stat", lambda r, x: format_num(r["ic_tstat"], 2)),
        ("IC hit rate", lambda r, x: format_pct(r["ic_hit_rate"])),
        ("Avg. names", lambda r, x: f"{r['avg_names']:.0f}"),
        ("Coverage", lambda r, x: format_pct(r["coverage"])),
        ("Profit factor", lambda r, x: format_num(x.get("profit_factor", np.nan), 2)),
        ("VaR 95 (5d)", lambda r, x: format_pct(x.get("var95_5d", np.nan), 2)),
        ("CVaR 95 (5d)", lambda r, x: format_pct(x.get("cvar95_5d", np.nan), 2)),
        ("Skew", lambda r, x: format_num(x.get("skew", np.nan), 2)),
        ("Excess kurtosis", lambda r, x: format_num(x.get("excess_kurtosis", np.nan), 2)),
    ]
    split_names = ["full", "train", "validation", "test"]
    table_rows = []
    for label, formatter in metric_specs:
        table_rows.append(
            [label]
            + [formatter(runs[name].metrics, extras[name]) for name in split_names]
        )
    table = ax_metrics.table(
        cellText=table_rows,
        colLabels=["Metric", "Full", "Train", "Validation", "Test"],
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#e5e8e8")
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 0:
            cell.set_facecolor("#eef3f7")
            cell.set_text_props(fontweight="bold", color="#17202a")
        elif row % 2 == 0:
            cell.set_facecolor("#fbfcfc")

    # Yearly stats heatmap/table
    ax_year_heat.axis("off")
    ax_year_heat.set_title("Year-by-year statistics", loc="left", fontsize=11, fontweight="bold", pad=8)
    stat_rows = [
        ("Return", "return", "pct", False),
        ("Ann. vol", "ann_vol", "pct", True),
        ("Sharpe", "sharpe", "num", False),
        ("Max DD", "max_drawdown", "pct", True),
        ("Mean IC", "mean_ic", "num", False),
        ("Hit rate", "hit_rate", "pct", False),
        ("Ann. turnover", "ann_turnover", "x", True),
    ]
    years_list = sorted(yearly)
    cell_text = []
    cell_colors = []
    base_cmap = plt.get_cmap("RdYlGn")
    for label, key, fmt, lower_is_better in stat_rows:
        values = np.array([yearly[year][key] for year in years_list], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size > 1 and finite.max() > finite.min():
            scores = (values - finite.min()) / (finite.max() - finite.min())
        else:
            scores = np.full_like(values, 0.5)
        if lower_is_better:
            scores = 1.0 - scores
        row_text = [label]
        row_colors = ["#eef3f7"]
        for value, score in zip(values, scores, strict=True):
            if not np.isfinite(value):
                row_text.append("—")
            elif fmt == "pct":
                row_text.append(f"{value:.1%}")
            elif fmt == "x":
                row_text.append(f"{value:.1f}x")
            else:
                row_text.append(f"{value:.2f}")
            color = base_cmap(float(np.clip(score, 0, 1))) if np.isfinite(score) else "#f4f6f7"
            row_colors.append(color)
        cell_text.append(row_text)
        cell_colors.append(row_colors)
    year_table = ax_year_heat.table(
        cellText=cell_text,
        colLabels=["Metric"] + [str(year) for year in years_list],
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 0.90],
    )
    year_table.auto_set_font_size(False)
    year_table.set_fontsize(7.6)
    for (row, col), cell in year_table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 0:
            cell.set_text_props(fontweight="bold", color="#17202a")
    ax_year_heat.text(
        0.0,
        -0.03,
        "2019 starts in July and 2026 ends in September; annual stats are partial for those years.",
        transform=ax_year_heat.transAxes,
        fontsize=6.2,
        color="#7f8c8d",
        va="top",
    )

    fig.text(
        0.01,
        0.002,
        "Source: FactorFoundry processed A-share panel; metrics use 5-day non-overlapping periods and 10 bps per-side cost. Past performance is not a live track record.",
        fontsize=7,
        color="#7f8c8d",
    )
    fig.subplots_adjust(left=0.03, right=0.985, top=0.985, bottom=0.03)

    figures_dir = config.paths.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / "illiquidity_20_tearsheet.png"
    pdf_path = figures_dir / "illiquidity_20_tearsheet.pdf"
    fig.savefig(png_path, dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    payload = {
        "factor": {
            "factor_id": FACTOR_ID,
            "name": spec.name,
            "family": spec.family,
            "category": spec.category,
            "formula": spec.formula,
            "hypothesis": spec.hypothesis,
            "direction": spec.direction,
            "window": spec.window,
        },
        "periods_per_year": PERIODS_PER_YEAR,
        "ranges": {name: {"start": start, "end": end} for name, (start, end) in ranges.items()},
        "metrics": {name: runs[name].metrics for name in split_names},
        "extra_metrics": extras,
        "yearly": yearly,
        "ic_decay": ic_decay,
        "quintile_spread": quintile,
        "bootstrap_sharpe_95ci": bootstrap,
        "statistical_evidence": {
            "fdr_q_value": float(factor_row["q_value"]),
            "deflated_sharpe": float(factor_row["dsr"]),
            "pbo": float(diagnostics["pbo_detail"]["pbo"]),
            "label_shuffle_p_value": float(factor_row["shuffle_p_value"]),
        },
        "latest_holdings": {
            "date": str(last_date.date()),
            "long": {str(k): float(v) for k, v in long_hold.items()},
            "short": {str(k): float(v) for k, v in short_hold.items()},
        },
        "outputs": {"png": str(png_path), "pdf": str(pdf_path)},
    }
    metrics_path = config.paths.output_dir / "illiquidity_20_tearsheet_metrics.json"
    metrics_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=lambda obj: float(obj) if isinstance(obj, np.floating) else int(obj) if isinstance(obj, np.integer) else str(obj),
        ),
        encoding="utf-8",
    )
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
