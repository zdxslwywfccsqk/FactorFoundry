# Roadmap

FactorFoundry is being built as public infrastructure for auditable alpha
research. The roadmap prioritizes evidence quality over feature count.

## Now — v0.2.x

- [x] deterministic reference runtime with no model API dependency;
- [x] Mini-Agent role adapter with scoped tools;
- [x] typed artifact blackboard, event ledger, and run manifest;
- [x] train/validation/test capability guard;
- [x] cost-aware backtest, FDR, DSR, PBO, and leakage audits;
- [x] reproducible A-share reference benchmark with explicit caveats;
- [ ] publish the launch kit, social preview, and first community review;
- [ ] add a one-command synthetic-data demo for contributors without market data.

## v0.3 — Evidence hardening

- [ ] point-in-time universe construction;
- [ ] delisting-aware and survivorship-controlled evaluation;
- [ ] long-only and short-constrained implementation modes;
- [ ] capacity, market-impact, and borrow-cost models;
- [ ] cost-sensitivity and parameter-perturbation reports;
- [ ] benchmark runs exportable without private market data.

## v0.4 — Agent research quality

- [ ] controlled comparison: template hypotheses vs. LLM hypotheses;
- [ ] single-agent vs. multi-agent ablations with fixed budget;
- [ ] experiment memory that retrieves rejected hypotheses and failure reasons;
- [ ] structured debate or feedback between Critic and Hypothesis roles;
- [ ] provider-neutral model adapter and cost accounting.

## v0.5 — Community benchmark

- [ ] leaderboard with fixed data fingerprints and evaluation protocol;
- [ ] external benchmark submission template;
- [ ] reference results across factor families and holding horizons;
- [ ] review checklist for claims that accompany a public run.

## v1.0 — Reproducibility standard

- [ ] stable artifact and event schemas;
- [ ] independent reproduction guide and archived reference environment;
- [ ] cross-market adapters with market-specific execution assumptions;
- [ ] release process for benchmark data, code, and evidence bundles.

## How to influence the roadmap

Open a research proposal issue for a new hypothesis, a benchmark submission, or a
falsification test. Contributions that make a claim easier to reproduce or easier
to disprove are prioritized over features that only make a result look better.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contribution workflow.
