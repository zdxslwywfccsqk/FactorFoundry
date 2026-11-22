from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AppConfig
from .dsl import pct_change, rolling_beta

LOGGER = logging.getLogger(__name__)

PRICE_FIELDS = ("open", "high", "low", "close", "volume", "amount")


def _pivot_field(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    wide = prices.pivot_table(
        index="date", columns="symbol", values=field, aggfunc="last"
    )
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    return wide


def _wide_market(market: pd.Series, columns: pd.Index) -> pd.DataFrame:
    values = np.repeat(market.to_numpy()[:, None], len(columns), axis=1)
    return pd.DataFrame(values, index=market.index, columns=columns)


def select_universe(
    fields: dict[str, pd.DataFrame], config: AppConfig
) -> tuple[list[str], dict]:
    close = fields["close"]
    dollar_volume = fields["dollar_volume"]
    selection_end = pd.Timestamp(config.data.selection_end_date)
    selection_slice = slice(None, selection_end)

    history = close.loc[selection_slice].notna().sum()
    avg_dollar = dollar_volume.loc[selection_slice].mean()
    median_price = close.loc[selection_slice].median()
    missing_ratio = close.loc[selection_slice].isna().mean()

    mask = (
        (history >= config.data.min_history_days)
        & (avg_dollar >= config.data.min_avg_dollar_volume)
        & (median_price >= config.data.min_price)
        & (missing_ratio <= config.data.max_missing_ratio)
    )
    selected = mask[mask].index.tolist()

    if len(selected) < 30:
        LOGGER.warning("Only %d stocks passed strict filters; relaxing filters", len(selected))
        relaxed = history[history >= 60].index.intersection(avg_dollar.dropna().index)
        selected = avg_dollar.loc[relaxed].sort_values(ascending=False).head(120).index.tolist()

    metadata = {
        "selection_end_date": config.data.selection_end_date,
        "requested_universe_size": config.data.universe_size,
        "selected_universe_size": len(selected),
        "strict_filter_count": int(mask.sum()),
        "median_history_days": float(history.median()),
        "median_avg_dollar_volume": float(avg_dollar.median()),
        "missing_ratio_window": "selection_end_date",
        "pit_universe": False,
        "delisting_control": False,
    }
    return selected, metadata


def build_fields(
    prices: pd.DataFrame,
    index_frame: pd.DataFrame,
    config: AppConfig,
) -> tuple[dict[str, pd.DataFrame], dict]:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.drop_duplicates(subset=["date", "symbol"]).sort_values(
        ["date", "symbol"]
    )

    fields: dict[str, pd.DataFrame] = {}
    for field in PRICE_FIELDS:
        if field in prices.columns:
            fields[field] = _pivot_field(prices, field)

    dates = fields["close"].index
    symbols = fields["close"].columns
    if "amount" in fields and fields["amount"].notna().any().any():
        fields["dollar_volume"] = fields["amount"].replace([np.inf, -np.inf], np.nan)
    else:
        # Tencent's daily volume is reported in lots (100 shares); convert to shares.
        fields["dollar_volume"] = fields["close"].mul(fields["volume"]).mul(100.0).replace(
            [np.inf, -np.inf], np.nan
        )
    fields["ret"] = pct_change(fields["close"])

    index_frame = index_frame.copy()
    index_frame["date"] = pd.to_datetime(index_frame["date"])
    index_close = (
        index_frame.drop_duplicates(subset=["date"])
        .set_index("date")["close"]
        .astype(float)
        .sort_index()
        .reindex(dates)
        .ffill()
    )
    market_ret = pct_change(index_close.to_frame("market"))["market"]
    fields["market_ret"] = _wide_market(market_ret, symbols)

    fields["adv20"] = fields["dollar_volume"].rolling(20, min_periods=10).mean()
    fields["adv60"] = fields["dollar_volume"].rolling(60, min_periods=20).mean()
    fields["log_adv20"] = np.log(fields["adv20"].clip(lower=1.0))

    for window in (60, 120):
        fields[f"beta_{window}"] = rolling_beta(
            fields["ret"], market_ret, window=window, min_periods=max(30, window // 2)
        )
    idio = fields["ret"].sub(
        fields["beta_60"].mul(market_ret, axis=0)
    )
    fields["idio_ret"] = idio
    fields["idio_vol_20"] = idio.rolling(20, min_periods=10).std()

    for window in (3, 5, 10, 20, 60, 120):
        fields[f"ret_{window}"] = pct_change(fields["close"], periods=window)

    limit = pd.Series(0.095, index=symbols, dtype=float)
    growth_or_star = symbols.str.startswith(("sz300", "sh688", "sz301"))
    limit.loc[growth_or_star] = 0.195
    fields["tradable"] = (
        fields["ret"].abs().le(limit, axis=1)
        & fields["volume"].gt(0)
        & fields["close"].notna()
    ).astype(float)

    selected, metadata = select_universe(fields, config)
    for key, frame in list(fields.items()):
        fields[key] = frame.reindex(columns=selected)

    metadata.update(
        {
            "start_date": str(dates.min().date()),
            "end_date": str(dates.max().date()),
            "n_dates": int(len(dates)),
            "n_raw_symbols": int(len(symbols)),
            "n_selected_symbols": int(len(selected)),
            "fields": sorted(fields.keys()),
        }
    )
    return fields, metadata


def save_fields(fields: dict[str, pd.DataFrame], metadata: dict, output_dir: Path) -> None:
    fields_dir = output_dir / "fields"
    fields_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in fields.items():
        frame.to_parquet(fields_dir / f"{name}.parquet")
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)


def load_fields(output_dir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    fields_dir = output_dir / "fields"
    if not fields_dir.exists():
        raise FileNotFoundError(f"Missing processed fields: {fields_dir}")
    fields = {
        path.stem: pd.read_parquet(path)
        for path in sorted(fields_dir.glob("*.parquet"))
    }
    metadata_path = output_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    return fields, metadata


def file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def data_fingerprint(config: AppConfig) -> str:
    parts = []
    for path in (
        config.paths.raw_dir / "universe.csv",
        config.paths.raw_dir / "prices" / "prices.parquet",
        config.paths.raw_dir / "prices" / "index.parquet",
    ):
        if path.exists():
            parts.append(f"{path.name}:{file_sha256(path)}")
    payload = "|".join(parts + [json.dumps(config.raw, sort_keys=True)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
