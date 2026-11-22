# Benchmark protocol

## Splits

The current reference configuration uses:

- train: 2019-07-01 to 2021-12-31,
- validation: 2022-01-01 to 2024-12-31,
- test: 2025-01-01 to 2026-09-30.

Selection happens on train and validation only. The test split is locked until a
frozen portfolio is released by the orchestrator.

## Evaluation

- Signal at close `t`; forward close-to-close return over the horizon; implicit
  close execution.
- Five-day horizon and five-day rebalance in the reference run.
- Rank-weighted long-short portfolio by default, with an equal-weight quantile
  option.
- 10 bps per side in the reference run, followed by cost-sensitivity stress tests.
- Trades are filtered through the available `tradable` mask.

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

The v0.2 deterministic runtime reproduced the original project result: test net
Sharpe **0.7500** and annualized net return **18.96%** for an `illiquidity_20`
portfolio. The exact run is documented in `REFERENCE_RUN.md`. Those numbers are a
starting benchmark, not a claim of robust alpha. The same run showed PBO near
0.51, DSR near 0.13, FDR q near 0.19, and a wide bootstrap Sharpe interval. The
release process should re-run on the corrected PIT universe and cost model before
using a headline number.

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
