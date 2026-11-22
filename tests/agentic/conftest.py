from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_foundry.config import (
    AppConfig,
    BacktestConfig,
    CriticConfig,
    DataConfig,
    PathConfig,
    PortfolioConfig,
    SplitConfig,
)


def make_fields(
    n_days: int = 320,
    n_symbols: int = 40,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    symbols = [f"s{i:03d}" for i in range(n_symbols)]
    returns = rng.normal(0.0003, 0.018, size=(n_days, n_symbols))
    close = pd.DataFrame(
        10.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=symbols
    )
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
            ret.rolling(120, min_periods=60).var(), axis=0
        ),
        "idio_ret": idio,
        "idio_vol_20": idio.rolling(20, min_periods=10).std(),
        "tradable": pd.DataFrame(1.0, index=dates, columns=symbols),
    }
    for window in (3, 5, 10, 20, 60, 120):
        fields[f"ret_{window}"] = close.pct_change(window, fill_method=None)
    return fields


def make_app_config(root: Path) -> AppConfig:
    raw = {
        "project": {"name": "FactorFoundry-test", "seed": 7},
        "paths": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "experiments_dir": "experiments",
            "output_dir": "output",
            "runs_dir": "runs",
        },
        "data": {
            "start_date": "2020-01-01",
            "end_date": "2021-03-31",
            "universe_size": 40,
            "index_symbol": "sh000300",
            "selection_end_date": "2020-06-30",
            "min_history_days": 60,
            "min_avg_dollar_volume": 1000.0,
            "min_price": 0.1,
            "max_missing_ratio": 0.9,
        },
        "splits": {
            "train_start": "2020-02-01",
            "train_end": "2020-06-30",
            "validation_start": "2020-07-01",
            "validation_end": "2020-10-31",
            "test_start": "2020-11-01",
            "test_end": "2021-03-31",
        },
        "backtest": {"horizon": 5, "portfolio_mode": "rank", "quantile": 0.2, "cost_bps_per_side": 10.0, "min_names": 10},
        "critic": {
            "min_coverage": 0.5,
            "min_validation_ic": 0.0,
            "min_validation_icir": 0.0,
            "min_validation_net_sharpe": -10.0,
            "max_turnover_annual": 1000.0,
            "max_fdr_q": 1.0,
            "max_pairwise_corr": 0.8,
            "min_subperiod_sign_ratio": 0.0,
        },
        "portfolio": {"max_factors": 3, "max_pairwise_corr": 0.95, "weighting": "equal"},
        "llm": {"enabled": False, "provider": "openai", "model": "unused"},
        "agentic": {"max_audits": 2, "factor_limit": 5},
    }
    paths = PathConfig(
        root=root,
        raw_dir=root / "data/raw",
        processed_dir=root / "data/processed",
        experiments_dir=root / "experiments",
        output_dir=root / "output",
        runs_dir=root / "runs",
    )
    return AppConfig(
        root=root,
        project_name="FactorFoundry-test",
        seed=7,
        paths=paths,
        data=DataConfig(**raw["data"]),
        splits=SplitConfig(**raw["splits"]),
        backtest=BacktestConfig(**raw["backtest"]),
        critic=CriticConfig(**raw["critic"]),
        portfolio=PortfolioConfig(**raw["portfolio"]),
        llm=raw["llm"],
        raw=raw,
    )


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    config = make_app_config(tmp_path)
    config.ensure_dirs()
    return config
