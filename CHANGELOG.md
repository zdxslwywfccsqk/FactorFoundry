# Changelog

All notable changes to FactorFoundry are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-11-22

Evidence-hardening release.

- Added a one-bar execution lag: signals observed at close `t` are executed at close
  `t+1`, and labels end at `t+1+horizon`.
- Enforced split-boundary label guards so no training or validation label can consume
  returns from the following split.
- Replaced half-L1 turnover with full traded-notional L1 turnover, including initial
  deployment and liquidation after skipped periods.
- Split costs into commission, sell-side stamp duty, spread, market impact, and optional
  borrow cost, with initial and rebalance charges recorded separately.
- Added a separate long-only implementation check, long/short exposure diagnostics, a
  bootstrap Sharpe interval, and a 10%-ADV capacity estimate.
- Extended future-data perturbation to compare the full available prefix and aligned the
  label-shuffle test to the same tradable universe, execution lag, and split boundary.
- Changed universe missingness filtering to use only the pre-`selection_end_date` window.
- Added the Alpha Evidence Scorecard: seven fail-closed gates, a 100-point evidence score,
  grade caps, and a separate `Alpha Candidate` certification.
- Added `factorfoundry-agent score <run_id>` and `make score RUN=<run_id>` with Markdown and
  JSON output, plus run-local config snapshots for reproducible rescoring.
- Published the v0.3 reference run `run_20261122_112513_8641a7`: net Sharpe `0.7567`,
  annualized net return `20.46%`, score `46/100`, grade `D`, certification `Benchmark only`.
- Added `docs/BACKTEST_AUDIT.md` and `docs/SCORECARD.md` documenting the engine-level
  defects, remediation status, remaining leakage limitations, and residual statistical risk.

## [0.2.1] - 2026-11-22

Open-source presentation and community launch refresh.

- Rebuilt the README as a product-quality research showcase with visual identity.
- Added a reproducible brand asset renderer, README hero, and social preview.
- Added a Chinese project introduction and launch kit.
- Added a public roadmap and clearer contribution paths.
- Sharpened the positioning around auditable alpha discovery and negative evidence.

## [0.2.0] - 2026-11-22

Initial open-source release candidate.

- Added the Mini-Agent-backed role runtime and deterministic reference policy.
- Added typed artifact blackboard with hashes, lineage, and split labels.
- Added append-only SQLite/JSONL event ledger and per-run manifests.
- Added train/validation/test split guard with explicit holdout release.
- Added cost-aware backtests, FDR, DSR, PBO, no-lookahead, and label-shuffle checks.
- Added low-correlation portfolio construction and evidence reports.
- Added the `run_20261119_144245_6b2581` reference benchmark and caveats.
- Added CI, contribution, security, citation, and release documentation.

## [0.1.0] - 2026-10-19

- Initial fixed-sequence research pipeline and factor-template prototype.
