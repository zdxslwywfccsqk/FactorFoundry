from __future__ import annotations

import numpy as np
import pandas as pd


def safe_div(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    return numerator.div(denominator.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )


def cs_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank mapped to a symmetric [-0.5, 0.5] interval."""
    count = frame.notna().sum(axis=1)
    rank = frame.rank(axis=1, method="average")
    centered = rank.sub(1.0).div(count.sub(1).replace(0, np.nan), axis=0).sub(0.5)
    return centered.where(count.gt(1), 0.0)


def cs_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0).replace(
        [np.inf, -np.inf], np.nan
    )


def cs_demean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sub(frame.mean(axis=1), axis=0)


def winsorize_cs(
    frame: pd.DataFrame, lower: float = 0.01, upper: float = 0.99
) -> pd.DataFrame:
    lo = frame.quantile(lower, axis=1)
    hi = frame.quantile(upper, axis=1)
    return frame.clip(lower=lo, upper=hi, axis=0)


def ts_mean(frame: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.DataFrame:
    return frame.rolling(window, min_periods=min_periods or window).mean()


def ts_std(frame: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.DataFrame:
    return frame.rolling(window, min_periods=min_periods or window).std()


def ts_sum(frame: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.DataFrame:
    return frame.rolling(window, min_periods=min_periods or window).sum()


def ts_max(frame: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.DataFrame:
    return frame.rolling(window, min_periods=min_periods or window).max()


def ts_min(frame: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.DataFrame:
    return frame.rolling(window, min_periods=min_periods or window).min()


def ts_skew(frame: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.DataFrame:
    return frame.rolling(window, min_periods=min_periods or window).skew()


def ts_corr(
    left: pd.DataFrame,
    right: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    return left.rolling(window, min_periods=min_periods or window).corr(right)


def delay(frame: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return frame.shift(periods)


def pct_change(frame: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return frame.pct_change(periods, fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    )


def rolling_beta(
    asset_returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    mp = min_periods or window
    cov = asset_returns.rolling(window, min_periods=mp).cov(market_returns)
    var = market_returns.rolling(window, min_periods=mp).var()
    return cov.div(var.replace(0, np.nan), axis=0)


def neutralize_cs(
    factor: pd.DataFrame,
    controls: list[pd.DataFrame],
    min_names: int = 10,
) -> pd.DataFrame:
    """Residualize a factor on cross-sectional controls date by date."""
    out = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
    aligned_controls = [control.reindex_like(factor) for control in controls]
    for dt in factor.index:
        y = factor.loc[dt].to_numpy(dtype=float)
        xs = [control.loc[dt].to_numpy(dtype=float) for control in aligned_controls]
        if not xs:
            out.loc[dt] = y
            continue
        mask = np.isfinite(y)
        for x in xs:
            mask &= np.isfinite(x)
        if mask.sum() < min_names:
            continue
        design = np.column_stack([np.ones(mask.sum())] + [x[mask] for x in xs])
        try:
            beta, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
        except np.linalg.LinAlgError:
            continue
        residual = y[mask] - design @ beta
        out.loc[dt, mask] = residual
    return out


def scale_positive(frame: pd.DataFrame) -> pd.DataFrame:
    """Scale a positive-valued frame by its cross-sectional median."""
    median = frame.median(axis=1).replace(0, np.nan)
    return frame.div(median, axis=0).replace([np.inf, -np.inf], np.nan)
