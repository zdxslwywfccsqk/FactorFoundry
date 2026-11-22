# Benchmark protocol

Every benchmark is also graded by [`SCORECARD.md`](SCORECARD.md). The scorecard is
independent of the headline Sharpe: a run that fails execution, cost, leakage, or
statistical-survival gates is labeled `Benchmark only`, regardless of its return.

## Splits

The current reference configuration uses:

- train: 2019-07-01 to 2021-12-31,
- validation: 2022-01-01 to 2024-12-31,
- test: 2025-01-01 to 2026-09-30.

Selection happens on train and validation only. The test split is locked until a
frozen portfolio is released by the orchestrator.

## Evaluation

- Signal at close `t`; execution at close `t+1`; forward close-to-close return from
  `t+1` to `t+1+horizon`.
- Five-day horizon and five-day rebalance in the reference run.
- Rank-weighted long-short portfolio by default, with an equal-weight quantile
  option and a separate long-only implementation check.
- Full traded-notional L1 turnover, including initial deployment and liquidation.
- Cost components: commission `2.5 bps`, stamp duty `5.0 bps` on sells, spread
  `2.0 bps`, and impact `2.0 bps`; borrow cost is explicit and separate.
- Trades are filtered through the available `tradable` mask.
- Labels whose last observation would cross a split end are excluded.

## Statistical evidence

Every benchmark report should include:

- mean rank IC, ICIR, and IC t-statistic;
- validation and test net Sharpe, annualized return, volatility, drawdown, and turnover;
- Benjamini-Hochberg FDR across all trials;
- Deflated Sharpe Ratio;
- Probability of Backtest Overfitting;
- bootstrap confidence interval for Sharpe;
- future-data perturbation and label-shuffle audits.

Failed trials count. Do not report a deflated statistic using only surviving
factors.

## Reference result and caveats

The v0.3 deterministic runtime reports test net Sharpe **0.7567** and annualized
net return **20.46%**. No factor passed every critic gate, so the portfolio is an
explicit validation-ranked fallback rather than an accepted alpha. The run has
PBO `0.5600`, DSR `0.2548`, FDR q `0.3303`, and a holdout bootstrap Sharpe
interval of `[-0.0898, 0.3177]`.

Under scorecard version `1.0.0`, the v0.3 reference run is **not an Alpha Candidate**.
It passes reproducibility, execution, cost, implementability, and sample-size gates,
but fails leakage control because the universe is not fully PIT/delisting-aware, and
fails statistical survival. The exact run is documented in `REFERENCE_RUN.md`; the
remaining work is tracked in [`BACKTEST_AUDIT.md`](BACKTEST_AUDIT.md).

## Required ablations

- deterministic template library vs. LLM hypothesis generation;
- single agent vs. multi-agent;
- with vs. without critic;
- with vs. without experiment memory;
- with vs. without leakage audits;
- long-only implementation vs. research long-short.

## Manuscript/release table

At minimum, report the benchmark, number of trials, selection rule, cost model,
turnover, drawdown, confidence interval, FDR, DSR, PBO, and all known data
limitations. The holdout number should be paired with the date it was first
released and whether it had been observed during development.
