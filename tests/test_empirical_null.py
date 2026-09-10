from __future__ import annotations

import numpy as np

from perturbo._statistics import (
    empirical_pvals_from_null,
    empirical_pvals_from_tnull_fixed0,
    z_to_two_sided_pvalues,
)


def test_empirical_pvals_from_null_matches_bias_corrected_tail() -> None:
    null_z = np.array([0.1, -0.2, 0.4, -0.5], dtype=float)
    real_z = np.array([0.3, -0.6], dtype=float)

    pvals = empirical_pvals_from_null(null_z, real_z, two_sided=True, bias_correction=True)

    assert np.allclose(pvals, np.array([3 / 5, 1 / 5], dtype=float))


def test_empirical_pvals_from_tnull_fixed0_returns_params() -> None:
    rng = np.random.default_rng(0)
    null_z = rng.standard_t(df=5, size=200)
    real_z = np.array([0.0, 1.0, 2.0], dtype=float)

    pvals, params = empirical_pvals_from_tnull_fixed0(null_z, real_z, return_params=True)

    assert pvals.shape == (3,)
    assert 0.0 <= pvals.min() <= pvals.max() <= 1.0
    assert params["n_null"] == 200
    assert params["df"] > 0
    assert params["scale"] > 0


def test_z_to_two_sided_pvalues_is_symmetric() -> None:
    z = np.array([-2.0, 0.0, 2.0], dtype=float)
    pvals = z_to_two_sided_pvalues(z)
    assert np.isclose(pvals[0], pvals[2])
    assert np.isclose(pvals[1], 1.0)
