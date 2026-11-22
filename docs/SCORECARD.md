# FactorFoundry Alpha Evidence Scorecard

**Version:** `1.0.0`  
**Scope:** one reproducible research run, not a live trading system.  
**Command:** `factorfoundry-agent score <run_id> --format markdown`

> 中文摘要：这不是一个按 Sharpe 排名的排行榜。只有通过全部硬门槛的运行，才允许被称为
> `Alpha Candidate`；否则即使回测收益很高，也只能称为 benchmark 或 exploratory result。

## Why a scorecard

Factor research has a failure mode that is easy to hide: every failed trial disappears,
the surviving curve is reported, and a single Sharpe number becomes the conclusion.
FactorFoundry instead asks a harder question:

> Would this result still look credible after execution delay, realistic costs, all
> trials, leakage audits, and implementation constraints are accounted for?

The scorecard has two independent parts:

1. **Hard gates** determine whether a run may use the `Alpha Candidate` label.
2. **A 100-point evidence score** measures how strong and complete the evidence is.

A high score cannot compensate for a failed hard gate. A failed gate does not erase a
useful benchmark; it only forbids an alpha or deployment claim.

## Hard gates

All gates are evaluated fail-closed. Missing evidence is a failure, not an unknown that
can be ignored.

| Gate | Requirement | Why it blocks an alpha claim |
|---|---|---|
| `G1` Reproducible run contract | `run_config.json` snapshot; non-empty run id, config hash, data fingerprint; completed status; at least one artifact | A result that cannot be reconstructed is not evidence |
| `G2` Leakage control | label-boundary guard, full-prefix no-lookahead audit, label-shuffle on the same universe, PIT universe, and passed audits for selected factors | Returns from a later split or future data invalidate the estimate |
| `G3` Execution causality | at least one bar between signal timestamp and execution; execution price explicitly recorded | A close-price signal cannot be filled at the same close |
| `G4` Cost integrity | L1 turnover cost model; initial position and rebalance both charged; at least four cost components | Under-counted costs manufacture alpha |
| `G5` Statistical survival | all candidates counted; selected-factor FDR `q <= 0.10`; DSR `>= 0.90`; PBO `<= 0.20`; holdout bootstrap Sharpe lower bound `> 0` | The result must survive selection and multiple testing |
| `G6` Implementability | long-only result, or constrained short with a positive borrow cost; capacity estimate present | A research long-short with unlimited borrow is not an A-share strategy |
| `G7` Sample sufficiency | at least 24 non-overlapping validation periods and 24 non-overlapping holdout periods | Short samples produce unstable rankings and confidence intervals |

### Certification labels

| Label | Condition |
|---|---|
| `Paper Alpha` | all gates pass and score `>= 90` |
| `Alpha Candidate` | all gates pass and score `>= 80` |
| `Research Candidate` | all gates pass but score `< 80` |
| `Benchmark only` | at least one hard gate fails |

`Paper Alpha` means the research evidence is strong enough for a paper-trading and
capacity study. It does not mean live-trading approval.

## 100-point evidence score

The score is diagnostic. It is intentionally not dominated by net Sharpe.

| Dimension | Points | What earns credit |
|---|---:|---|
| Statistical evidence | 35 | FDR, DSR, PBO, bootstrap interval, IC/ICIR/subperiod stability |
| Execution and cost | 25 | execution lag, complete cost model, turnover, capacity, implementability |
| Data integrity | 20 | PIT universe, delisting control, T+1/limit/ST/suspension controls, PIT missingness |
| Robustness | 15 | subperiod stability, cost sensitivity, parameter perturbation, regime stability, pristine holdout |
| Reproducibility and governance | 5 | config snapshot, code revision, artifact lineage, retained failures, holdout and audit governance |

### Statistical evidence: 35 points

| Item | Points | Rule |
|---|---:|---|
| FDR | 8 | selected-factor maximum `q <= 0.05` = 8; `<= 0.10` = 6; `<= 0.20` = 4; `<= 0.30` = 2 |
| DSR | 8 | selected-factor minimum `DSR >= 0.95` = 8; `>= 0.90` = 6; `>= 0.75` = 4; `>= 0.50` = 2 |
| PBO | 7 | `<= 0.10` = 7; `<= 0.20` = 5; `<= 0.35` = 3; `<= 0.50` = 1 |
| Bootstrap Sharpe interval | 5 | holdout lower bound `> 0` = 5; validation lower bound `> 0` = 4; upper bound `> 0` = 2; interval present = 1 |
| IC stability | 7 | validation `IC > 0` = 2; `ICIR >= 0.30` = 2; subperiod sign ratio `>= 0.66` = 2; IC hit rate `>= 0.50` = 1 |

### Execution and cost: 25 points

| Item | Points | Rule |
|---|---:|---|
| Execution lag | 7 | at least one bar = 7; same-close or missing = 0 |
| Cost completeness | 7 | turnover-L1 model, initial and rebalance charged, at least four components = 7; at least two components = 5; only a per-side bps number = 2 |
| Turnover and capacity | 5 | annual turnover `<= 12x` = 3; `<= 20x` = 2; `<= 35x` = 1; capacity estimate = 2 |
| Implementability | 6 | long-only = 6; constrained short with positive borrow cost = 6; unconstrained research long-short = 0 |

### Data integrity: 20 points

| Item | Points | Rule |
|---|---:|---|
| PIT universe | 6 | explicit point-in-time construction = 6 |
| Survivorship and delisting | 5 | explicit delisting-aware universe = 5 |
| Trading constraints | 5 | T+1 = 2; limit, ST, and suspension controls = 1 each; a tradable mask alone earns at most 1 |
| PIT missingness | 4 | missingness computed only through the universe-selection cutoff = 4 |

### Robustness: 15 points

| Item | Points | Rule |
|---|---:|---|
| Subperiod stability | 4 | validation and holdout subperiod IC sign ratio each contribute 2 |
| Cost sensitivity | 4 | at least three cost levels reported |
| Parameter perturbation | 3 | at least three parameter variants reported |
| Regime stability | 2 | at least two market regimes reported |
| Pristine holdout | 2 | explicit evidence that the test period was not previously observed |

### Reproducibility and governance: 5 points

| Item | Points | Rule |
|---|---:|---|
| Config snapshot | 1 | `run_config.json` exists in the run directory |
| Code revision | 1 | non-empty git commit in the manifest |
| Artifact lineage | 1 | artifacts exist and carry content hashes |
| Retained failures | 1 | rejected trials remain in the evidence bundle |
| Holdout and audit governance | 1 | holdout release record and at least one audit report |

## Grades

| Grade | Score | Meaning |
|---|---:|---|
| `S` | 90-100 | research-grade evidence; still requires paper trading and capacity validation |
| `A` | 80-89 | strong conditional evidence |
| `B` | 65-79 | useful benchmark, not a certified alpha |
| `C` | 50-64 | exploratory; data, execution, or statistical gaps are material |
| `D` | `< 50` | incomplete or non-credible as alpha evidence |

If any hard gate fails, the maximum grade is `B`, even if the raw score would otherwise
be `A` or `S`.

## Evidence contract

Future runs should write `scorecard_evidence` into `summary.json`, `run_config.json`,
or a standalone `scorecard_evidence.json`. The deterministic tools, not an LLM, must
produce these fields.

```json
{
  "leakage": {
    "label_boundary_guard": true,
    "no_lookahead_full_prefix": true,
    "label_shuffle_same_universe": true,
    "pit_universe": true
  },
  "data": {
    "delisting_control": true,
    "t_plus_one": true,
    "limit_control": true,
    "st_control": true,
    "suspension_control": true,
    "selection_missing_window": true
  },
  "execution": {
    "execution_lag_bars": 1,
    "trade_price": "next_close",
    "implementability": "long_only",
    "borrow_cost_bps": 0.0,
    "capacity_estimate": true
  },
  "cost": {
    "mode": "turnover_l1",
    "initial_position_charged": true,
    "rebalance_charged": true,
    "components": ["commission", "stamp_duty", "spread", "impact"]
  },
  "statistics": {
    "holdout_pristine": true
  },
  "robustness": {
    "cost_sensitivity": [{"bps": 5}, {"bps": 10}, {"bps": 20}],
    "parameter_perturbation": [{"window": 10}, {"window": 20}, {"window": 40}],
    "regime_results": [{"regime": "low_vol"}, {"regime": "high_vol"}]
  }
}
```

An evidence flag is not a substitute for the corresponding deterministic audit. The
scorecard accepts the flag only because the runtime is expected to generate it from a
tool result that is linked by artifact id and content hash.

## Reporting rule

Every public benchmark must state:

- run id, config hash, data fingerprint, and git commit;
- score, grade, certification, and failed gates;
- validation and holdout net Sharpe with bootstrap intervals;
- FDR, DSR, PBO, turnover, and cost model;
- all known data and implementation limitations.

Do not report a headline Sharpe without the scorecard verdict next to it.

## Reference run status

`run_20261122_112513_8641a7` scores **46.0 / 100**, grade **D**, certification
**Benchmark only**. It passes `G1`, `G3`, `G4`, `G6`, and `G7`, but fails:

- `G2` because the public universe is not fully point-in-time and delisting-aware;
- `G5` because PBO is `0.5600`, DSR is `0.2548`, FDR q is `0.3303`, and the holdout
  bootstrap Sharpe interval includes zero.
