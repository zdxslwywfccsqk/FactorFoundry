# Third-party notices

FactorFoundry is released under the MIT License. It builds on open-source
software and data adapters whose own licenses continue to apply.

## Mini-Agent

The optional agent runtime integrates with **Mini-Agent** from MiniMax-AI:

- Repository: <https://github.com/MiniMax-AI/Mini-Agent>
- License: MIT
- Usage: single-agent execution loop, LLM client abstraction, and tool protocol

FactorFoundry does not vendor the Mini-Agent source tree. The optional dependency
is pinned separately so its license and notices remain visible.

## Python packages

The project uses NumPy, pandas, SciPy, PyArrow, Matplotlib, Pydantic, and
pytest. Each package retains its own license and copyright.

## Market data

Market data is not distributed with this repository. Data adapters may query
public services such as Baostock, Sina, or Tencent quote endpoints. Users are
responsible for complying with those services' terms, redistribution limits,
and rate limits.

## Research disclaimer

This repository is a research and engineering project. It is not investment
advice, an offer, or a solicitation. Backtests are conditional on data quality,
execution assumptions, and cost models that are necessarily incomplete.
