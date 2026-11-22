from __future__ import annotations

import numpy as np
from conftest import make_fields

from alpha_foundry.factors import compute_factor, generate_factor_specs
from alpha_foundry.stats import audit_no_lookahead


def test_factor_ids_are_unique_and_large_enough() -> None:
    specs = generate_factor_specs()
    ids = [spec.spec_id for spec in specs]
    assert len(ids) == len(set(ids))
    assert len(ids) >= 60


def test_factor_computation_and_no_lookahead() -> None:
    fields = make_fields()
    specs = {spec.spec_id: spec for spec in generate_factor_specs()}
    for factor_id in ("rev_5", "mom_20", "lowvol_20", "rangevol_20", "clv_5"):
        factor = compute_factor(specs[factor_id], fields)
        assert factor.shape == fields["close"].shape
        assert np.isfinite(factor.to_numpy()).mean() > 0.4
        audit = audit_no_lookahead(
            specs[factor_id], fields, cutoff=fields["close"].index[-80]
        )
        assert audit["passed"], (factor_id, audit)
