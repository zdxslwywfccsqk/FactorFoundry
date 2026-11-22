# Agent protocol

## Roles

| Role | Responsibility | May write | Must not do |
|---|---|---|---|
| Director | Inspect state and identify the next bottleneck | notes, research narration | accept factors, unlock holdout |
| Hypothesis | Turn economic mechanisms into hypothesis cards | `hypothesis_card` | run backtests, invent data |
| Feasibility | Check fields and cross-sectional coverage | `feasibility_report` | evaluate returns |
| Engineer | Implement feasible hypotheses with the DSL | `factor_implementation` | change evaluation protocol |
| Evaluator | Run train/validation tools | `backtest_report` | access test |
| Critic | Inspect evidence and add evidence-based vetoes | `critic_review` | accept a factor that failed deterministic gates |
| Portfolio | Combine accepted factors | `portfolio_decision` | use test performance for selection |
| Librarian | Summarize artifacts and open questions | notes | change research decisions |

## Task lifecycle

```text
PENDING → RUNNING → COMPLETED
                  ↘ FAILED
                  ↘ BLOCKED
```

Every task records its role, objective, structured inputs, allowed tools, step
limit, result status, token count, created artifacts, and a transcript.

## State machine

```text
hypothesis → feasibility → engineering → evaluation → critic → portfolio
                                                               ↘ holdout (explicit release)
```

The orchestrator enforces legal transitions. A role cannot skip the critic or
construct a portfolio from rejected factors.

## Critic authority

Deterministic gates are authoritative. An LLM critic may:

- add a veto to an accepted factor,
- request a narrower interpretation,
- document unresolved risks.

An LLM critic may not accept a factor that failed FDR, coverage, train/validation,
turnover, or leakage checks. This asymmetry prevents an eloquent review from
overriding statistical evidence.

## Prompt-injection posture

Retrieved text and tool output are treated as untrusted data. Agents have no
shell, no arbitrary Python tool, and no ability to mint API credentials. Any
future web or document retrieval tool should be data-only and should not be able
to alter the system prompt or control plane.
