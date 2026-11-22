# Reference run: v0.2 deterministic runtime

Run id: `run_20261119_144245_6b2581`

Command:

```bash
PYTHONPATH=src python3 -m alpha_foundry.agentic.cli run \
  --config config.json \
  --mode deterministic \
  --factor-limit 100 \
  --max-audits 12 \
  --release-holdout
```

## Funnel

| Stage | Count |
|---|---:|
| hypothesis cards | 73 |
| implemented factors | 73 |
| critic candidates | 73 |
| critic accepted | 2 |
| portfolio factors | 1 |
| total artifacts | 374 |

Accepted factors were `corrclosevol_20` and `illiquidity_20`. The portfolio
selected `illiquidity_20` because adding the second factor did not improve
validation performance enough under the correlation constraint.

## Portfolio metrics

| Metric | Validation | Test / holdout |
|---|---:|---:|
| Net Sharpe | 0.6181 | 0.7500 |
| Net annualized return | 10.61% | 18.96% |
| Max drawdown | -25.02% | -36.52% |
| Annualized turnover | 6.71x | 10.14x |

## Statistical evidence

| Item | Value |
|---|---:|
| PBO | 0.5133 |
| FDR q for `illiquidity_20` | 0.1865 |
| DSR for `illiquidity_20` | 0.1263 |
| FDR q for `corrclosevol_20` | 0.0110 |
| DSR for `corrclosevol_20` | 0.0278 |

## Interpretation

The headline test numbers match the original project claim: net Sharpe `0.75`
and annualized net return `18.96%`. The new runtime reproduces them with an
explicit artifact chain and a recorded holdout release.

This is still not strong evidence of a durable, implementable alpha:

- PBO is high;
- DSR is low;
- the test window may have been observed in earlier project iterations;
- the long-short backtest does not fully model A-share borrowing constraints;
- the universe and data source retain documented point-in-time and survivorship risks;
- costs are constant rather than capacity-aware.

Treat this run as a reproducible benchmark and a starting point for the next
research protocol, not as an investment recommendation.
