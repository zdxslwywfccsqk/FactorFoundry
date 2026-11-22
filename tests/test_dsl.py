from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_foundry.dsl import cs_rank, cs_zscore, neutralize_cs, safe_div


def test_cs_rank_and_zscore() -> None:
    frame = pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], columns=["a", "b", "c"])
    ranked = cs_rank(frame)
    assert np.allclose(ranked.sum(axis=1).to_numpy(), 0.0)
    z = cs_zscore(frame)
    assert np.allclose(z.mean(axis=1).to_numpy(), 0.0, atol=1e-12)


def test_neutralize_removes_linear_exposure() -> None:
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 3.0, 4.0, 5.0]})
    control = x * 2.0 + 1.0
    residual = neutralize_cs(x, [control], min_names=2)
    assert residual.abs().max().max() < 1e-10


def test_safe_div_handles_zero() -> None:
    left = pd.DataFrame([[1.0, 2.0]])
    right = pd.DataFrame([[0.0, 2.0]])
    result = safe_div(left, right)
    assert pd.isna(result.iloc[0, 0])
    assert result.iloc[0, 1] == 1.0
