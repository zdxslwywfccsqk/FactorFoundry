# Open-source release checklist

## Reproducibility

- [x] Commit contains no market data or generated run artifacts.
- [x] `pyproject.toml` declares runtime and optional dependencies.
- [ ] Release notes record the resolved environment and reference config hash.
- [ ] A fresh environment can run the deterministic smoke test offline.
- [x] Every benchmark number has a run id, config hash, data fingerprint, and git commit.
- [x] The test split is marked as locked unless the release artifact includes an explicit unlock record.
- [x] Failed trials and rejected factors are included in the evidence bundle.

## Research quality

- [x] FDR, DSR, PBO, bootstrap intervals, and turnover are reported.
- [x] The run has a generated scorecard with a published score, grade, and certification.
- [x] Long-only and long-short results are separated.
- [x] A capacity screening estimate is included.
- [ ] Cost-sensitivity and parameter-perturbation stress tests are included.
- [x] Survivorship, PIT, ST, suspension, limit-up/down, and delisting risks are documented.
- [x] The report states that the historical test set may have been observed during development.

## Security and compliance

- [x] No API keys, private endpoints, or personal data are in the repository.
- [x] `SECURITY.md` and `THIRD_PARTY_NOTICES.md` are present.
- [x] Agent tools have no default shell or arbitrary code execution.
- [x] Data-source terms and redistribution limits are documented.
- [x] The investment-advice disclaimer is visible in the README.

## Community

- [x] `LICENSE`, `CONTRIBUTING.md`, and issue templates are present.
- [ ] Quickstart works on a clean machine.
- [x] Architecture, agent protocol, and benchmark protocol docs are current.
- [ ] A minimal example run is committed as expected-output text, not binary market data.
- [x] README hero, Chinese README, and GitHub social preview assets are present.
- [ ] GitHub About text and suggested topics are populated from the launch kit.
