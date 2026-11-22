# Contributing

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Optional Mini-Agent integration:

```bash
python -m pip install -e '.[mini-agent]'
```

## Before opening a pull request

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m alpha_foundry.agentic.cli run \
  --config config.json --factor-limit 4 --max-audits 1
```

Please keep the following invariants:

1. **No test-set access in research agents.** Add a regression test for any new tool.
2. **No silent metric mutation.** Metrics must come from deterministic tool output.
3. **No untracked experiments.** Every accepted or rejected factor needs an artifact.
4. **No credentials or market data in commits.** Use environment variables and local data directories.
5. **No arbitrary agent-authored shell execution** in the default distribution.

## Adding a factor template

Add the economic mechanism to `src/alpha_foundry/factors.py`, extend the formula
and compute branch, and cover it with:

- a unit test for finite coverage,
- a no-lookahead audit,
- a smoke backtest on synthetic data.

The agent interface should expose templates only. Do not give agents direct
access to the raw Python evaluator.

## Adding an agent role

Update `roles.py`, the role allowlist in `tools.py`, and `docs/AGENT_PROTOCOL.md`.
Every role must have a bounded tool set, a maximum step count, and an artifact
contract.
