from __future__ import annotations

import numpy as np
import pandas as pd


def make_fields(n_days: int = 320, n_symbols: int = 40, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    symbols = [f"s{i:03d}" for i in range(n_symbols)]
    returns = rng.normal(0.0003, 0.018, size=(n_days, n_symbols))
    close = pd.DataFrame(10.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=symbols)
    open_ = close * (1.0 + rng.normal(0.0, 0.004, size=close.shape))
    high = pd.DataFrame(
        np.maximum(open_.to_numpy(), close.to_numpy())
        * (1.0 + np.abs(rng.normal(0.0, 0.004, size=close.shape))),
        index=dates,
        columns=symbols,
    )
    low = pd.DataFrame(
        np.minimum(open_.to_numpy(), close.to_numpy())
        * (1.0 - np.abs(rng.normal(0.0, 0.004, size=close.shape))),
        index=dates,
        columns=symbols,
    )
    volume = pd.DataFrame(
        rng.lognormal(15.0, 0.5, size=close.shape), index=dates, columns=symbols
    )
    ret = close.pct_change(fill_method=None)
    dollar_volume = close * volume
    market_ret = ret.mean(axis=1)
    market_frame = pd.DataFrame(
        np.repeat(market_ret.to_numpy()[:, None], n_symbols, axis=1),
        index=dates,
        columns=symbols,
    )
    beta = ret.rolling(60, min_periods=30).cov(market_ret).div(
        market_ret.rolling(60, min_periods=30).var(), axis=0
    )
    idio = ret.sub(beta.mul(market_ret, axis=0))
    fields = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "ret": ret,
        "dollar_volume": dollar_volume,
        "market_ret": market_frame,
        "adv20": dollar_volume.rolling(20, min_periods=10).mean(),
        "adv60": dollar_volume.rolling(60, min_periods=20).mean(),
        "log_adv20": np.log(dollar_volume.rolling(20, min_periods=10).mean().clip(lower=1.0)),
        "beta_60": beta,
        "beta_120": ret.rolling(120, min_periods=60).cov(market_ret).div(
            market_ret.rolling(120, min_periods=60).var(), axis=0
        ),
        "idio_ret": idio,
        "idio_vol_20": idio.rolling(20, min_periods=10).std(),
        "tradable": pd.DataFrame(1.0, index=dates, columns=symbols),
    }
    for window in (3, 5, 10, 20, 60, 120):
        fields[f"ret_{window}"] = close.pct_change(window, fill_method=None)
    return fields
