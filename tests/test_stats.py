from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_foundry.stats import bh_fdr, deflated_sharpe, probability_of_backtest_overfitting


def test_bh_fdr_simple() -> None:
    q = bh_fdr(np.array([0.01, 0.02, 0.03, 0.5]))
    assert np.allclose(q, np.array([0.04, 0.04, 0.04, 0.5]))


def test_deflated_sharpe_is_probability() -> None:
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.001, 0.01, size=200))
    sharpes = rng.normal(0.02, 0.05, size=80)
    result = deflated_sharpe(returns, n_trials=80, trial_sharpes=sharpes)
    assert 0.0 <= result["dsr"] <= 1.0


def test_pbo_is_bounded() -> None:
    rng = np.random.default_rng(7)
    matrix = pd.DataFrame(rng.normal(size=(160, 12)))
    result = probability_of_backtest_overfitting(matrix, n_splits=16, n_trials=50, seed=7)
    assert 0.0 <= result["pbo"] <= 1.0
