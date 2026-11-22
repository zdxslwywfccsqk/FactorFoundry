from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from .config import BacktestConfig
from .factors import FactorSpec, compute_factor


def bh_fdr(pvalues: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    qvalues = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return qvalues
    p = values[finite]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    qvalues[finite] = out
    return qvalues


def _period_sharpe(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std <= 0:
        return 0.0
    return float(np.mean(returns) / std)


def deflated_sharpe(
    returns: pd.Series,
    n_trials: int,
    trial_sharpes: list[float] | np.ndarray,
) -> dict[str, float]:
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) < 3 or n_trials < 1:
        return {"sharpe_period": np.nan, "sr0": np.nan, "dsr": np.nan}
    sr = float(np.mean(values) / np.std(values, ddof=1)) if np.std(values, ddof=1) > 0 else np.nan
    trial_sr = np.asarray(trial_sharpes, dtype=float)
    trial_sr = trial_sr[np.isfinite(trial_sr)]
    variance = float(np.var(trial_sr, ddof=1)) if len(trial_sr) > 1 else 0.0
    sigma = math.sqrt(max(variance, 0.0))
    gamma = 0.5772156649015329
    n = max(1, int(n_trials))
    expected_max = (1.0 - gamma) * norm.ppf(1.0 - 1.0 / n) + gamma * norm.ppf(
        1.0 - 1.0 / (n * math.e)
    )
    sr0 = sigma * expected_max
    if not np.isfinite(sr):
        return {"sharpe_period": np.nan, "sr0": float(sr0), "dsr": np.nan}
    skewness = float(skew(values, bias=False)) if len(values) > 2 else 0.0
    kurt = float(kurtosis(values, fisher=False, bias=False)) if len(values) > 3 else 3.0
    denominator = 1.0 - skewness * sr + (kurt - 1.0) / 4.0 * sr**2
    if denominator <= 0 or not np.isfinite(denominator):
        return {"sharpe_period": sr, "sr0": float(sr0), "dsr": np.nan}
    z = (sr - sr0) * math.sqrt(len(values) - 1) / math.sqrt(denominator)
    return {
        "sharpe_period": sr,
        "sr0": float(sr0),
        "dsr": float(norm.cdf(z)),
    }


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame,
    n_splits: int = 16,
    n_trials: int = 300,
    seed: int = 42,
) -> dict[str, float | int]:
    if returns_matrix.shape[0] < n_splits * 2 or returns_matrix.shape[1] < 2:
        return {"pbo": np.nan, "n_trials": 0, "median_oos_rank": np.nan}
    matrix = returns_matrix.dropna(how="all").fillna(0.0)
    block_ids = np.array_split(np.arange(len(matrix)), n_splits)
    rng = np.random.default_rng(seed)
    values = matrix.to_numpy(dtype=float)
    below_median = 0
    ranks: list[float] = []
    half = n_splits // 2
    for _ in range(n_trials):
        chosen = rng.choice(n_splits, size=half, replace=False)
        is_idx = np.concatenate([block_ids[i] for i in chosen])
        oos_idx = np.concatenate([block_ids[i] for i in range(n_splits) if i not in chosen])
        is_sr = np.array([_period_sharpe(values[is_idx, j]) for j in range(values.shape[1])])
        oos_sr = np.array([_period_sharpe(values[oos_idx, j]) for j in range(values.shape[1])])
        best = int(np.argmax(is_sr))
        rank = float(np.mean(oos_sr < oos_sr[best]))
        ranks.append(rank)
        if rank <= 0.5:
            below_median += 1
    return {
        "pbo": float(below_median / n_trials),
        "n_trials": int(n_trials),
        "median_oos_rank": float(np.median(ranks)) if ranks else np.nan,
    }


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) < 10:
        return {"sharpe_ci_low": np.nan, "sharpe_ci_high": np.nan}
    rng = np.random.default_rng(seed)
    sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        sharpes.append(_period_sharpe(sample))
    sharpes = np.asarray(sharpes)
    return {
        "sharpe_ci_low": float(np.quantile(sharpes, 0.025)),
        "sharpe_ci_high": float(np.quantile(sharpes, 0.975)),
    }


def audit_no_lookahead(
    spec: FactorSpec,
    fields: dict[str, pd.DataFrame],
    cutoff: str | pd.Timestamp,
    seed: int = 42,
) -> dict[str, float | bool | int]:
    cutoff = pd.Timestamp(cutoff)
    original = compute_factor(spec, fields)
    rng = np.random.default_rng(seed)
    perturbed_fields: dict[str, pd.DataFrame] = {}
    for name, frame in fields.items():
        perturbed = frame.copy()
        future_mask = perturbed.index > cutoff
        if future_mask.any():
            noise = rng.normal(loc=0.0, scale=1.0, size=(int(future_mask.sum()), frame.shape[1]))
            perturbed.loc[future_mask] = perturbed.loc[future_mask].to_numpy(dtype=float) + noise
        perturbed_fields[name] = perturbed
    perturbed = compute_factor(spec, perturbed_fields)
    if cutoff not in original.index or cutoff not in perturbed.index:
        return {"passed": False, "max_abs_diff": np.nan, "n_compared": 0}
    left = original.loc[cutoff]
    right = perturbed.loc[cutoff]
    both = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if both.empty:
        return {"passed": False, "max_abs_diff": np.nan, "n_compared": 0}
    diff = (both["left"] - both["right"]).abs()
    max_diff = float(diff.max())
    return {
        "passed": bool(max_diff < 1e-10),
        "max_abs_diff": max_diff,
        "n_compared": int(len(both)),
    }


def label_shuffle_test(
    spec: FactorSpec,
    fields: dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    config: BacktestConfig,
    n_shuffles: int = 30,
    seed: int = 42,
) -> dict[str, float | bool | int]:
    factor = compute_factor(spec, fields)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    dates = factor.index[(factor.index >= start) & (factor.index <= end)][:: config.horizon]
    rng = np.random.default_rng(seed)
    original_ics: list[float] = []
    shuffled_means: list[float] = []
    aligned: list[tuple[np.ndarray, np.ndarray]] = []
    for dt in dates:
        pair = pd.concat(
            [factor.loc[dt].rename("factor"), fwd_ret.loc[dt].rename("forward")], axis=1
        ).dropna()
        if len(pair) < config.min_names or pair["factor"].nunique() < 2:
            continue
        original_ics.append(float(pair["factor"].corr(pair["forward"], method="spearman")))
        aligned.append((pair["factor"].to_numpy(), pair["forward"].to_numpy()))
    if not aligned:
        return {"original_mean_ic": np.nan, "shuffled_mean_ic": np.nan, "p_value": np.nan, "passed": False, "n_shuffles": 0}
    for _ in range(n_shuffles):
        ics = []
        for x, y in aligned:
            y_shuffled = rng.permutation(y)
            ics.append(float(pd.Series(x).corr(pd.Series(y_shuffled), method="spearman")))
        shuffled_means.append(float(np.nanmean(ics)))
    original_mean = float(np.nanmean(original_ics))
    shuffled_mean = float(np.nanmean(shuffled_means))
    p_value = float((1 + np.sum(np.asarray(shuffled_means) >= original_mean)) / (n_shuffles + 1))
    return {
        "original_mean_ic": original_mean,
        "shuffled_mean_ic": shuffled_mean,
        "p_value": p_value,
        "passed": bool(original_mean > shuffled_mean and p_value < 0.10),
        "n_shuffles": int(n_shuffles),
    }
