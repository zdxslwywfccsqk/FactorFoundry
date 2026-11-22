# FactorFoundry

**Auditable multi-agent alpha factor research for A-share equities.**

FactorFoundry turns economically motivated hypotheses into reproducible,
leakage-audited factor experiments. The design principle is deliberate:

> Language-model agents propose and critique. Deterministic tools compute, audit,
> and decide whether a factor survives.

This repository is a research framework, not investment advice and not a live
trading system.

## Status

`v0.2` is the first agentic-runtime release candidate. It includes:

- a deterministic offline research path that always works;
- an optional Mini-Agent role runtime with scoped tools;
- an append-only artifact blackboard and event ledger;
- a train/validation/test split guard with explicit holdout release;
- cost-aware backtests, FDR/DSR/PBO, leakage audit, and label-shuffle tests;
- a greedy low-correlation portfolio builder;
- a run report with a complete evidence chain.

The legacy fixed-sequence pipeline remains available through
`alpha_foundry.pipeline` for comparison, but new work should use
`alpha_foundry.agentic`.

## Architecture

```text
ResearchOrchestrator
  ├── deterministic reference policy
  └── Mini-Agent role runner
        ├── Hypothesis Agent
        ├── Feasibility Agent
        ├── Factor Engineer
        ├── Evaluator (deterministic service)
        ├── Critic Agent
        └── Portfolio Agent
                │
                ▼
        Artifact Blackboard
        SQLite + JSON + JSONL
                │
                ▼
        Deterministic Tool Layer
        data · DSL · backtest · statistics · text split guard
```

Agents do not pass mutable Python objects or chat messages into the numerical
core. They submit typed artifacts. Every artifact records its producer, parents,
content hash, schema version, split, and payload. The orchestrator enforces legal
state transitions and the split guard.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
[`docs/AGENT_PROTOCOL.md`](docs/AGENT_PROTOCOL.md).

## Quickstart: deterministic mode

Required: Python 3.10+, and cached or downloadable A-share data.

```bash
git clone https://github.com/zdxslwywfccsqk/factorfoundry.git
cd factorfoundry
python3 -m pip install -e .

PYTHONPATH=src python3 -m alpha_foundry.agentic.cli run \
  --config config.json \
  --mode deterministic \
  --factor-limit 12 \
  --max-audits 4
```

This runs the full artifact protocol without any model API call. It is the
reference path for CI and offline reproduction.

To explicitly evaluate the locked holdout after portfolio construction:

```bash
PYTHONPATH=src python3 -m alpha_foundry.agentic.cli run \
  --config config.json \
  --mode deterministic \
  --factor-limit 12 \
  --max-audits 4 \
  --release-holdout
```

The holdout release is recorded in `holdout_unlock.json` and the event ledger.

## Quickstart: Mini-Agent mode

Install the optional runtime:

```bash
python3 -m pip install -e '.[mini-agent]'
```

For local development against a checkout of Mini-Agent:

```bash
export MINI_AGENT_ROOT=/path/to/Mini-Agent
```

Configure an API key through an environment variable rather than committing it:

```bash
export FACTORFOUNDRY_API_KEY='...'
```

Enable the LLM in `config.json`:

```json
{
  "llm": {
    "enabled": true,
    "provider": "openai",
    "model": "gpt-4.1-mini",
    "api_base": "https://api.openai.com/v1",
    "api_key_env": "FACTORFOUNDRY_API_KEY"
  }
}
```

Then run:

```bash
PYTHONPATH=src python3 -m alpha_foundry.agentic.cli run \
  --config config.json \
  --mode mini-agent \
  --factor-limit 8 \
  --max-audits 4
```

If the model runtime or credentials are unavailable, the orchestrator can fall
back to the deterministic path when `agentic.fallback_on_agent_failure` is true.

## Run outputs

Each run creates a self-contained directory:

```text
runs/<run_id>/
  manifest.json          # run contract, config hash, data fingerprint, status
  report.md              # human-readable evidence report
  summary.json           # machine-readable summary
  events.jsonl           # append-only event stream
  ledger.sqlite          # artifacts + events
  artifacts/             # one JSON file per artifact
  agent_logs/            # Mini-Agent transcripts and logs, when enabled
  holdout_unlock.json    # only if the holdout was explicitly released
```

Useful commands:

```bash
PYTHONPATH=src python3 -m alpha_foundry.agentic.cli list --config config.json
PYTHONPATH=src python3 -m alpha_foundry.agentic.cli inspect <run_id> --config config.json
```

## Reference benchmark

The v0.2 deterministic runtime reproduces the original project result: a test net
Sharpe of **0.7500** and annualized net return of **18.96%** for an
`illiquidity_20` portfolio. The full run, command, and caveats are in
[`docs/REFERENCE_RUN.md`](docs/REFERENCE_RUN.md). It is a reference benchmark,
not a claim of robust alpha. The same run showed:

- PBO around `0.51`;
- Deflated Sharpe around `0.13`;
- FDR q-value around `0.19`;
- a wide bootstrap Sharpe interval;
- a long-short portfolio whose short leg is difficult to implement in A-shares.

The release protocol therefore treats the headline number as conditional
evidence. See [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md).

## Validation protocol

- Signal at close `t`; forward close-to-close return; implicit close execution.
- Five-day horizon and five-day rebalance in the reference configuration.
- Rank-weighted long-short by default; quantile mode is available.
- Constant per-side cost in the reference run, with sensitivity tests required
  before any external claim.
- Train and validation are available to research agents. Test is locked.
- Acceptance requires deterministic statistical gates plus future-data
  perturbation and label-shuffle audits.
- An LLM critic may add vetoes but cannot accept a factor that failed the gates.

## Repository layout

```text
src/alpha_foundry/
  data.py                 # market-data panel construction
  dsl.py                  # cross-sectional and time-series operators
  factors.py              # economic template library
  backtest.py             # cost-aware evaluation
  stats.py                # FDR, DSR, PBO, leakage audits
  portfolio.py            # signal combination
  agentic/
    schemas.py            # artifact and role contracts
    artifacts.py          # append-only artifact store
    ledger.py             # event ledger
    splits.py             # holdout guard
    session.py            # deterministic research services
    tools.py              # scoped Mini-Agent tools
    roles.py              # role prompts and tool boundaries
    runtime.py            # Mini-Agent adapter
    orchestrator.py       # state machine
    report.py             # evidence report
    cli.py                # command-line entry point
  pipeline.py             # legacy fixed-sequence pipeline
tests/                    # offline and adapter tests
docs/                     # architecture, protocol, benchmark, release checklist
```

## Tests

```bash
PYTHONPATH=src python3 -m pytest -q
```

If a local Mini-Agent checkout is available:

```bash
PYTHONPATH=src:${MINI_AGENT_ROOT} \
  python3 -m pytest -q tests/agentic/test_mini_agent_adapter.py
```

## Scope and non-goals

- No market data is redistributed.
- No live trading or broker integration.
- No arbitrary agent-authored Python execution in the default runtime.
- No claim that a single historical run proves durable alpha.
- No free-for-all multi-agent chat; roles are bounded by tools and artifacts.

## Open-source notes

- Code license: MIT.
- Mini-Agent license and attribution: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- Contribution rules: [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Security model: [`SECURITY.md`](SECURITY.md).
- Release checklist: [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md).

If you use this project academically, see [`CITATION.cff`](CITATION.cff).
Release history is in [`CHANGELOG.md`](CHANGELOG.md).
