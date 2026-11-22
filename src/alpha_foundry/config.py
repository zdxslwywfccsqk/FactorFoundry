from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PathConfig:
    root: Path
    raw_dir: Path
    processed_dir: Path
    experiments_dir: Path
    output_dir: Path
    runs_dir: Path = Path("runs")


@dataclass
class DataConfig:
    start_date: str
    end_date: str
    universe_size: int
    index_symbol: str
    selection_end_date: str
    min_history_days: int
    min_avg_dollar_volume: float
    min_price: float
    max_missing_ratio: float


@dataclass
class SplitConfig:
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


@dataclass
class BacktestConfig:
    horizon: int = 5
    portfolio_mode: str = "rank"
    quantile: float = 0.10
    execution_lag: int = 1
    cost_bps_per_side: float = 10.0
    commission_bps: float = 2.5
    stamp_duty_bps: float = 5.0
    spread_bps: float = 2.0
    impact_bps: float = 2.0
    borrow_bps_annual: float = 0.0
    min_names: int = 20


@dataclass
class CriticConfig:
    min_coverage: float = 0.65
    min_validation_ic: float = 0.005
    min_validation_icir: float = 0.10
    min_validation_net_sharpe: float = 0.10
    max_turnover_annual: float = 35.0
    max_fdr_q: float = 0.20
    max_pairwise_corr: float = 0.80
    min_subperiod_sign_ratio: float = 0.66


@dataclass
class PortfolioConfig:
    max_factors: int = 5
    max_pairwise_corr: float = 0.65
    weighting: str = "equal"


@dataclass
class AppConfig:
    root: Path
    project_name: str
    seed: int
    paths: PathConfig
    data: DataConfig
    splits: SplitConfig
    backtest: BacktestConfig
    critic: CriticConfig
    portfolio: PortfolioConfig
    llm: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> AppConfig:
        path = Path(path).resolve()
        root = path.parent
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        p = raw.get("paths", {})
        paths = PathConfig(
            root=root,
            raw_dir=root / p.get("raw_dir", "data/raw"),
            processed_dir=root / p.get("processed_dir", "data/processed"),
            experiments_dir=root / p.get("experiments_dir", "experiments"),
            output_dir=root / p.get("output_dir", "output"),
            runs_dir=root / p.get("runs_dir", "runs"),
        )
        return cls(
            root=root,
            project_name=raw.get("project", {}).get("name", "FactorFoundry"),
            seed=int(raw.get("project", {}).get("seed", 42)),
            paths=paths,
            data=DataConfig(**raw["data"]),
            splits=SplitConfig(**raw["splits"]),
            backtest=BacktestConfig(**raw["backtest"]),
            critic=CriticConfig(**raw["critic"]),
            portfolio=PortfolioConfig(**raw["portfolio"]),
            llm=raw.get("llm", {}),
            raw=raw,
        )

    def ensure_dirs(self) -> None:
        for path in (
            self.paths.raw_dir,
            self.paths.raw_dir / "prices",
            self.paths.processed_dir,
            self.paths.processed_dir / "fields",
            self.paths.experiments_dir,
            self.paths.output_dir,
            self.paths.output_dir / "figures",
            self.paths.runs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
