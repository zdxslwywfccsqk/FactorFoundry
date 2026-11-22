# Backtest framework audit

**Audit date:** 2026-11-22  
**Scope:** deterministic research backtest in `src/alpha_foundry/backtest.py`, the
statistical audits in `src/alpha_foundry/stats.py`, and the agentic acceptance path in
`src/alpha_foundry/agentic/session.py`.  
**Original audited run:** `run_20261119_144245_6b2581` (net Sharpe `0.7500`, annualized
net return `18.96%`).  
**Remediated run:** `run_20261122_112513_8641a7` (net Sharpe `0.7567`, annualized net
return `20.46%`, scorecard `46/100`, grade `D`, certification `Benchmark only`).

## Verdict

The v0.2.1 framework was **not fully correct**, and its result must not be described as
robust or directly tradable alpha. The v0.3.0 remediation fixes the engine-level defects
below, but the current result still fails the scorecard's leakage and statistical-survival
gates. It remains a reproducible research benchmark, not a certified alpha.

The high-level structure was and remains sound:

- factors are computed from historical rolling data;
- cross-sectional rank IC and rank-weighted portfolios are implemented clearly;
- gross and net returns are separated;
- artifacts preserve producer, lineage, split, hash, and status;
- the holdout is protected by a runtime capability guard;
- the critic can veto a factor but cannot accept one that failed the deterministic gates.

Those properties are valuable. They are not enough to validate the final performance
number. The v0.3.0 status of each issue is recorded here so the audit does not become
stale evidence.

## v0.3.0 remediation status

| Finding | Status | Evidence |
|---|---|---|
| Same-close signal and execution | **Fixed** | one-bar execution lag; close `t+1` entry |
| Labels crossing split boundaries | **Fixed** | eligible signal dates require label end inside the split |
| Half-L1 and undercounted initial cost | **Fixed** | full L1 traded notional, initial deployment charged, cost components separated |
| PBO and DSR absent from certification | **Partially fixed** | scorecard blocks certification; critic still records DSR without making it an acceptance gate |
| Research long-short only | **Partially fixed** | separate long-only implementation and capacity estimate reported; short-leg borrow/recall still unmodelled |
| FDR tests IC, not net return | **Open** | IC significance and net-return bootstrap are reported separately |
| Single-cutoff no-lookahead audit | **Fixed** | full-prefix comparison, 210,352 values checked in the reference run |
| Label-shuffle universe mismatch | **Fixed** | same `tradable` mask, execution lag, and split boundary |
| Full-sample missingness filter | **Fixed** | missingness computed only through `selection_end_date` |
| Non-PIT and survivorship-biased universe | **Open** | public universe has `pit_universe=false` and no delisting returns |
| Validation-set portfolio selection | **Open** | fallback portfolio is explicitly ranked on validation; nested validation remains future work |
| Stale weights after skipped periods | **Fixed** | skipped periods liquidate and charge turnover |
| Bootstrap interval not surfaced | **Fixed** | validation and holdout bootstrap intervals are stored and reported |
| Ambiguous return timeline | **Fixed in protocol** | signal, execution, holding, turnover, and label-end timestamps documented |

The remediation status explains why the new run has a higher score than the original
audit but still fails certification: better execution and cost accounting do not create
statistical significance.

## Blocking findings

### P0 (fixed in v0.3.0): same-close signal and execution

`forward_returns` computes `close[t + H] / close[t] - 1` in
`src/alpha_foundry/backtest.py:21`. The agentic protocol records the execution assumption
as "signal at close t; implicit close execution" in
`src/alpha_foundry/agentic/session.py:415`.

A factor that uses the close at `t` cannot be filled at the same close. At minimum the
backtest needs an execution lag of one bar and an explicit execution price. This is an
optimistic bias, not a presentation issue. The reported Sharpe and annual return must be
recomputed after the fix.

### P0 (fixed in v0.3.0): labels can cross split boundaries

`evaluate_factor` filters signal dates with `start <= t <= end` in
`src/alpha_foundry/backtest.py:57-61`, but the label loaded at `t` may extend into the
next split. For example, the last training signal can use returns from the validation
period, and the last validation signal can use returns from the test period.

The signal-date filter must also require the label's last observation to be on or before
the split end. Without that condition, validation and holdout performance are not clean
out-of-sample estimates.

### P0 (fixed in v0.3.0): cost accounting understates trading costs

Turnover is defined as half the L1 weight change in
`src/alpha_foundry/backtest.py:110-114`, then multiplied by `cost_bps_per_side` in
`src/alpha_foundry/backtest.py:152`.

For a gross-2 long-short portfolio, initial deployment has L1 weight change `2.0`. With
10 bps per unit of traded notional, the initial cost should be approximately 20 bps. The
current formula charges approximately 10 bps. A full reversal has L1 change `4.0`; the
current formula charges approximately 20 bps instead of approximately 40 bps.

The cost model must define `cost_bps` as cost per unit of traded notional and use
`sum(abs(w_t - w_{t-1})) * bps / 10000`, including the initial position. This fix alone
will reduce the reference net Sharpe and net return.

### P0 (partially fixed): PBO and DSR do not affect acceptance

The critic computes DSR in `src/alpha_foundry/agentic/session.py:611-614` and stores PBO
in the review, but the acceptance expression in
`src/alpha_foundry/agentic/session.py:639` does not include either statistic. The current
reference run has PBO `0.5133` and DSR `0.1263` for the selected factor, yet the factor is
accepted.

That is not a coding accident to hide with documentation. Either DSR and PBO become hard
acceptance gates, or the project must state that they are diagnostic and that the run is
not a statistically certified alpha. The scorecard takes the second route while blocking
the `Alpha Candidate` label until the evidence improves.

### P0 (partially fixed): research long-short is not an executable A-share strategy

The default portfolio is a gross-2 rank long-short portfolio. The framework does not model
borrow availability, borrow fees, recall risk, short-sale restrictions, or a long-only
implementation. The reference run's own limitations acknowledge this.

An A-share alpha claim needs at least one of:

- a long-only implementation with an investable benchmark;
- a constrained short implementation with explicit borrow assumptions and costs.

## High-priority findings

### P1 (open): FDR tests IC, not net portfolio return

`_pvalues` in `src/alpha_foundry/agentic/session.py:546-559` tests the mean IC t-statistic.
This is a valid test of cross-sectional predictive association, but it is not a test of
net portfolio profitability. The project should report both:

- IC significance after FDR;
- net-return significance from a bootstrap or block bootstrap, together with DSR and PBO.

### P1 (fixed in v0.3.0): no-lookahead audit checks only one cutoff row

`audit_no_lookahead` in `src/alpha_foundry/stats.py:130-161` compares only the factor row
at the cutoff date. A factor that leaks future data on another date can still pass.

The audit should compare the entire prefix `original.loc[:cutoff]` against the perturbed
prefix. The scorecard's `no_lookahead_full_prefix` evidence flag is intentionally blocked
until that comparison exists.

### P1 (fixed in v0.3.0): label-shuffle audit does not use the main backtest universe

`label_shuffle_test` in `src/alpha_foundry/stats.py:164-206` does not apply the same
`tradable` mask, execution lag, or split-boundary label filter as the main evaluation.
The audit therefore tests a different sample from the one used to produce the result.

Both paths must share one universe/date/label function. Otherwise a passed shuffle test
does not validate the reported portfolio.

### P1 (partially fixed): universe selection still uses full-sample missingness

`select_universe` computes `missing_ratio = close.isna().mean()` over the full sample in
`src/alpha_foundry/data.py:44`, while history, dollar volume, and price are computed only
through `selection_end_date`. This is a known point-in-time leakage path. It should use
`close.loc[selection_slice].isna().mean()`.

The public-data universe also remains subject to survivorship and delisting bias. A
price-only public feed is not sufficient for a production PIT universe.

### P1 (open): portfolio construction is selected on validation performance

`construct_portfolio` ranks candidates by validation net Sharpe and greedily adds factors
only when validation Sharpe improves by more than `0.01` in
`src/alpha_foundry/agentic/session.py:784-838`. This is a second selection layer on the
same validation data.

Without nested validation or a separate selection holdout, the reported validation
portfolio Sharpe is optimistically biased. The scorecard's `holdout_pristine` and
multiple-testing requirements are designed to expose this gap.

## Medium-priority findings

### P2 (fixed in v0.3.0): skipped periods leave stale previous weights

When `len(available) < min_names`, `evaluate_factor` uses `continue` at
`src/alpha_foundry/backtest.py:82-83` without updating `previous_weights`. The next valid
period computes turnover against an old portfolio even though the intervening period was
not traded. The strategy should explicitly choose one rule: hold the old portfolio, or
liquidate and charge the liquidation cost. Silently skipping is not a complete rule.

### P2 (fixed in v0.3.0): bootstrap intervals are implemented but not integrated

`bootstrap_sharpe_ci` exists in `src/alpha_foundry/stats.py:110-127`, but the agentic
portfolio and holdout reports do not include its output. The benchmark protocol promises
the interval without publishing it. The bootstrap result should be stored in the
backtest artifact and surfaced in the human-readable report.

### P2 (fixed in protocol): long-short accounting and return timing need a single contract

The current code computes weights from signal-time availability, applies the return to
those weights, and records the signal date as the return date. After adding execution
lag, the framework needs one explicit timeline:

- signal timestamp;
- execution timestamp and price;
- holding-period start and end;
- turnover and cost timestamp;
- label end timestamp used for split filtering.

Without a single timeline, different parts of the system can silently disagree about
which period a return belongs to.

## What is already sound

- The core factor library is deterministic and uses rolling historical windows.
- Rank IC uses Spearman correlation and the rank portfolio construction is transparent.
- Gross and net return series are separated, and annualization is based on the horizon.
- The Benjamini-Hochberg implementation in `bh_fdr` is correct.
- PBO uses block splits and random in-sample/out-of-sample partitions; it is not the
  canonical full CSCV implementation, but its direction is reasonable.
- Holdout access is guarded and audited.
- Rejected hypotheses, implementations, and critic decisions remain in the artifact store.
- The language-model critic cannot accept a factor that failed the deterministic checks.

## What the reference number means

The remediated `0.7567` net Sharpe and `20.46%` annualized net return are a reproducible
benchmark under one-bar execution, component costs, split-boundary label guards, and a
149-symbol universe. They are **not** evidence of a certified alpha: no factor passed
every critic gate, PBO is `0.5600`, DSR is `0.2548`, FDR q is `0.3303`, and the holdout
bootstrap interval includes zero. The portfolio is an explicit fallback.

## Required next actions

1. Construct a full point-in-time universe with delisting returns and ST/limit state.
2. Add exact T+1 settlement and directional limit-up/limit-down execution rules.
3. Make DSR, PBO, and net-return bootstrap evidence deterministic acceptance gates, not
   only scorecard gates.
4. Add nested validation or a separate selection split for portfolio construction.
5. Model borrow availability, recall risk, and financing costs for any short leg.
6. Replace the ADV capacity screen with a participation-rate impact model.
7. Test the long-only implementation against a beta-matched benchmark before quoting it.
8. Publish the generated scorecard alongside every benchmark table.
