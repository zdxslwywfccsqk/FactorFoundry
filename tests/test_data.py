from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_foundry.config import AppConfig
from alpha_foundry.data import select_universe


def test_universe_missingness_is_point_in_time(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "project": {"name": "test", "seed": 7},
                "paths": {
                    "raw_dir": "data/raw",
                    "processed_dir": "data/processed",
                    "experiments_dir": "experiments",
                    "output_dir": "output",
                    "runs_dir": "runs",
                },
                "data": {
                    "start_date": "2020-01-01",
                    "end_date": "2021-12-31",
                    "universe_size": 40,
                    "index_symbol": "sh000300",
                    "selection_end_date": "2020-06-30",
                    "min_history_days": 50,
                    "min_avg_dollar_volume": 1000.0,
                    "min_price": 0.1,
                    "max_missing_ratio": 0.10,
                },
                "splits": {
                    "train_start": "2020-01-01",
                    "train_end": "2020-06-30",
                    "validation_start": "2020-07-01",
                    "validation_end": "2020-12-31",
                    "test_start": "2021-01-01",
                    "test_end": "2021-12-31",
                },
                "backtest": {"horizon": 5, "min_names": 10},
                "critic": {},
                "portfolio": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = AppConfig.load(config_path)
    dates = pd.bdate_range("2020-01-01", "2021-12-31")
    symbols = [f"s{index:03d}" for index in range(35)]
    close = pd.DataFrame(10.0, index=dates, columns=symbols)
    close.loc[close.index > pd.Timestamp("2020-06-30"), "s001"] = np.nan
    fields = {
        "close": close,
        "dollar_volume": pd.DataFrame(1_000_000.0, index=dates, columns=symbols),
    }

    selected, metadata = select_universe(fields, config)

    assert "s001" in selected
    assert metadata["missing_ratio_window"] == "selection_end_date"
