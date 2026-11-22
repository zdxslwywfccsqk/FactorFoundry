# Reference run: v0.3 hardened benchmark

Run id: `run_20261122_112513_8641a7`

## Manifest

| Field | Value |
|---|---|
| Git commit | `91c7634f79315c08e78485e3e3519b1399b62c2e` |
| Config hash | `a00e0b5dfd8b48929018583f3f605e148ebcb181507b1c5d54d8dc9995a60b7d` |
| Data fingerprint | `eec4dd96ed622bfa` |
| Status | `completed` |
| Artifacts | `375` |
| Holdout released | `yes` |

Command:

```bash
PYTHONPATH=src python3 -m alpha_foundry.agentic.cli run \
  --config config.json \
  --mode deterministic \
  --factor-limit 100 \
  --max-audits 12 \
  --release-holdout
```

## Scorecard verdict

```text
46.0 / 100 · Grade D · Alpha Candidate: NO
Certification: Benchmark only
```

Failed hard gates:

- `G2 Leakage control`: the universe is not fully point-in-time or delisting-aware.
- `G5 Statistical survival`: PBO, DSR, FDR q, and the holdout bootstrap interval do not meet
  the certification thresholds.

Passed hard gates include reproducibility, execution lag, cost integrity, long-only
implementation and capacity evidence, and sample sufficiency. See
[`SCORECARD.md`](SCORECARD.md).

## Funnel

| Stage | Count |
|---|---:|
| hypothesis cards | 73 |
| implemented factors | 73 |
| critic candidates | 73 |
| critic accepted | 0 |
| portfolio factors | 1 |
| total artifacts | 375 |

No factor passed every deterministic critic check. The public portfolio is therefore an
**exploratory fallback** selected on validation performance, not an accepted alpha. The
fallback factor is `illiquidity_20`.

## Portfolio metrics

| Metric | Validation | Test / holdout |
|---|---:|---:|
| Net Sharpe | 0.7587 | 0.7567 |
| Net annualized return | 12.84% | 20.46% |
| Net annualized volatility | 18.09% | 30.73% |
| Max drawdown | -24.40% | -36.00% |
| Annualized turnover | 13.38x | 20.37x |
| Annualized cost drag | 1.20% | 1.83% |
| Bootstrap 95% Sharpe interval | [-0.0578, 0.2742] | [-0.0898, 0.3177] |

The holdout Sharpe is similar in level to the v0.2.1 result, but it is not statistically
distinguishable from zero at the 95% level.

## Execution and cost protocol

- Signal at close `t`; execution at close `t+1`; holding period of five trading days.
- Returns whose label would extend beyond a split end are excluded.
- Turnover is full traded notional: `Σabs(w_t - w_{t-1})`.
- Initial deployment is charged. The initial long-short turnover is `2.0` and the initial
  all-in cost is approximately `17.93 bps`.
- Cost components are commission `2.5 bps`, stamp duty `5.0 bps` on sells, spread `2.0 bps`,
  and impact `2.0 bps`; borrow cost is zero for the long-only implementation.

## Statistical and leakage evidence

| Item | Value |
|---|---:|
| PBO | 0.5600 |
| FDR q for fallback `illiquidity_20` | 0.3303 |
| DSR for fallback `illiquidity_20` | 0.2548 |
| No-lookahead audit scope | full prefix |
| No-lookahead compared values | 210,352 |
| No-lookahead max absolute difference | 0.0 |
| Label-shuffle p-value | 0.0323 |
| Label-shuffle periods | 144 |
| Label-shuffle same universe | yes |

The leakage audits themselves pass. The statistical survival evidence does not. A passed
audit means the implementation does not obviously use future data; it does not turn a high
PBO or a low DSR into alpha.

## Implementation check

A separate long-only version of the same signal is reported to test whether the research
portfolio has an executable direction:

| Metric | Validation | Holdout |
|---|---:|---:|
| Long-only net Sharpe | 0.6136 | 2.9689 |
| Long-only net annualized return | 13.70% | 88.44% |
| Long-only annualized turnover | 4.30x | 7.48x |

The holdout long-only return is dominated by raw market beta and the market path, not by a
market-neutral alpha. It should not be quoted without a beta benchmark and an explicit
market-exposure decomposition.

The 10%-ADV capacity estimate is a median AUM of approximately `297.5 million CNY`, with a
p10 estimate of approximately `200.1 million CNY`. This is a screening estimate, not an
execution-capacity proof.

## Universe and remaining limitations

- The universe is selected from 198 publicly available symbols; 149 pass the current filter.
- Missingness is now computed only through `selection_end_date`.
- The universe is still not a full point-in-time market universe and has no delisting return
  model, so `G2` fails.
- ST status, exact limit-up/limit-down direction, T+1 settlement, IPO restrictions, borrow
  recall, and intraday impact remain simplified or unmodelled.
- The test window may have been observed during earlier project development.
- The portfolio is a fallback, not a factor that passed every acceptance gate.

## Comparison with v0.2.1

The v0.2.1 benchmark reported net Sharpe `0.7500` and annualized net return `18.96%` using
same-close execution, half-L1 turnover costs, a 155-symbol full-sample missingness filter,
and no separate long-only result. The v0.3 result is a cleaner benchmark but still fails the
Alpha Evidence Scorecard. The two headline numbers are not directly comparable.

## Interpretation

This run is useful because it is reproducible, auditable, and honest about failure. It is
not an investment recommendation and must not be presented as a certified or deployable
alpha.
