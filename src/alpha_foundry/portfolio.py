from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .dsl import cs_zscore


def signal_from_factor(factor: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally standardize a factor and neutralize missing values."""
    return cs_zscore(factor).fillna(0.0)


def combine_signals(
    signals: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    if not signals:
        raise ValueError("At least one signal is required")
    if weights is None:
        weights = {key: 1.0 / len(signals) for key in signals}
    total = 0.0
    result: pd.DataFrame | None = None
    for key, signal in signals.items():
        weight = float(weights.get(key, 0.0))
        if weight == 0.0:
            continue
        total += weight
        contribution = signal * weight
        result = contribution if result is None else result + contribution
    if result is None:
        raise ValueError("All signal weights were zero")
    if not np.isclose(total, 1.0):
        result = result / total
    return result


def select_low_correlation(
    candidates: list,
    returns_fn: Callable[[object], pd.Series],
    max_corr: float,
    max_factors: int,
) -> list:
    selected: list = []
    for candidate in candidates:
        if len(selected) >= max_factors:
            break
        candidate_returns = returns_fn(candidate)
        if not selected:
            selected.append(candidate)
            continue
        corrs = []
        for other in selected:
            pair = pd.concat(
                [candidate_returns.rename("a"), returns_fn(other).rename("b")], axis=1
            ).dropna()
            corrs.append(abs(float(pair["a"].corr(pair["b"]))) if len(pair) > 5 else 0.0)
        if max(corrs) <= max_corr:
            selected.append(candidate)
    return selected
