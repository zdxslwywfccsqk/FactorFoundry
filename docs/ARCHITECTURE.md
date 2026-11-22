# Architecture

## Design principle

Language models are good at proposing mechanisms, selecting implementations, and
writing criticism. They are poor places to store numerical truth. FactorFoundry
therefore separates:

- **generation and judgment** — agent roles;
- **computation and acceptance** — deterministic tools and statistical gates;
- **coordination** — a state machine over a shared artifact blackboard;
- **evidence** — append-only artifacts, events, and manifests.

## Runtime layers

```text
CLI / orchestration
    ResearchOrchestrator
        ├── deterministic reference policy
        └── Mini-Agent role runner
                    │
                    ▼
              ResearchSession
        ├── artifact store
        ├── event ledger
        ├── split guard
        ├── factor cache
        ├── backtest cache
        └── critic / portfolio services
                    │
                    ▼
             deterministic core
        data → DSL → backtest → statistics → portfolio → report
```

## Artifact blackboard

Agents do not pass mutable Python objects to each other. They submit artifacts:

```text
run_plan
  └── hypothesis_card
        └── feasibility_report
              └── factor_implementation
                    ├── backtest_report (train)
                    ├── backtest_report (validation)
                    └── audit_report
                          └── critic_review
                                └── portfolio_decision
                                      └── holdout_report
```

Every artifact has a run id, producer, parents, split label, schema version,
content hash, and JSON payload. Revisions append new artifacts instead of
mutating old evidence.

## Split guard

`train` and `validation` are available to research tools. `test` is denied until
`SplitGuard.release_holdout` writes a run-level unlock record. The release is
logged and cannot be repeated without an explicit force flag. In production, the
same interface should be backed by a separate evaluation service and filesystem
permissions.

## Mini-Agent integration

Each LLM role is a separate Mini-Agent instance with:

- a role-specific system prompt,
- a scoped tool allowlist,
- a maximum step count,
- a maximum token budget,
- a transcript artifact.

Mini-Agent supplies the single-agent loop. FactorFoundry supplies the multi-agent
state machine, artifact protocol, and deterministic guardrails. We do not model
multi-agent collaboration as an unbounded chat room.

## Why the deterministic path exists

The deterministic path is the reference implementation. It makes CI and offline
reproduction possible, and it is the fallback when model credentials are absent.
The Mini-Agent path must produce the same artifact kinds and pass the same gates.
