# Open-source release checklist

## Reproducibility

- [ ] Commit contains no market data or generated run artifacts.
- [ ] `pyproject.toml` declares runtime and optional dependencies.
- [ ] Release notes record the resolved environment and reference config hash.
- [ ] A fresh environment can run the deterministic smoke test offline.
- [ ] Every benchmark number has a run id, config hash, data fingerprint, and git commit.
- [ ] The test split is marked as locked unless the release artifact includes an explicit unlock record.
- [ ] Failed trials and rejected factors are included in the evidence bundle.

## Research quality

- [ ] FDR, DSR, PBO, bootstrap intervals, and turnover are reported.
- [ ] Long-only and long-short results are separated.
- [ ] Cost sensitivity and capacity stress tests are included.
- [ ] Survivorship, PIT, ST, suspension, limit-up/down, and delisting risks are documented.
- [ ] The report states that the historical test set may have been observed during development.

## Security and compliance

- [ ] No API keys, private endpoints, or personal data are in the repository.
- [ ] `SECURITY.md` and `THIRD_PARTY_NOTICES.md` are present.
- [ ] Agent tools have no default shell or arbitrary code execution.
- [ ] Data-source terms and redistribution limits are documented.
- [ ] The investment-advice disclaimer is visible in the README.

## Community

- [ ] `LICENSE`, `CONTRIBUTING.md`, and issue templates are present.
- [ ] Quickstart works on a clean machine.
- [ ] Architecture, agent protocol, and benchmark protocol docs are current.
- [ ] A minimal example run is committed as expected-output text, not binary market data.
- [ ] README hero, Chinese README, and GitHub social preview render correctly.
- [ ] GitHub About text and suggested topics are populated from the launch kit.
