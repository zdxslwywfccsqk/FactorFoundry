# Security policy

## Scope

FactorFoundry processes market data and optionally calls language-model APIs.
The most relevant risks are:

- prompt injection through retrieved text or tool output,
- credential leakage through logs or artifacts,
- arbitrary code execution by an agent,
- accidental holdout access and research leakage,
- denial of service through unbounded agent loops or API spend.

## Current controls

- The default runtime is deterministic and makes no external model calls.
- Research agents receive scoped tools, not shell or arbitrary Python execution.
- The test split is blocked by a run-level split guard and explicit release record.
- Tool arguments, results, errors, and timing are appended to an event ledger.
- API keys are read from environment variables and are never written to artifacts.
- Agent steps and token limits are bounded by role configuration.

## Reporting a vulnerability

Please do not publish credentials, private market data, or exploitable prompts in
an issue. Open a minimal private report to the maintainers with:

- affected commit or run id,
- reproduction steps,
- whether the holdout or an API key was exposed,
- suggested mitigation if known.

## Deployment note

For multi-user or hosted use, run generated code only inside a container or
microVM with no network, read-only data mounts, CPU/memory/time limits, and a
separate holdout evaluation service. The local process-level guard is not a
security boundary against a malicious operator.
