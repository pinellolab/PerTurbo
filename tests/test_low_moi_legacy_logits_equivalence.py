"""The shared-slope selection model reaches the low-MOI saddlepoint in two forms.

``precompute_low_moi_permutations`` fits screen-wide covariate slopes plus a
per-target intercept, then stores the result as coefficients in a compact basis
(``Q @ coef`` recovers a target's logits) *and*, since this change, as the shared
logit vector plus the intercepts (``eta_shared + delta``). The kernel rebuilds
logits from the compact form with a (cells, basis) product for every promoted
block; a batch covariate makes that basis hundreds of columns wide and on a
genome-wide screen the rebuild dominated the CRT. The legacy form is one
broadcast add.

These tests pin what the switch is allowed to change: nothing that a caller
reads. Flags must be identical; p-values may differ only by the float32 rounding
the compact form carries in its stored coefficients and basis, which is measured
here rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from perturbo._internal.saddlepoint import fit_low_moi_propensity_saddlepoint


def _wide_basis_problem(seed: int = 5, *, levels: int = 40, num_targets: int = 6, num_genes: int = 9):
    """A control-anchored screen whose selection basis carries a batch one-hot."""

    rng = np.random.default_rng(seed)
    num_controls = 500
    own_counts = rng.integers(10, 45, size=num_targets)
    codes = np.concatenate([np.full(num_controls, -1), np.repeat(np.arange(num_targets), own_counts)])
    num_cells = codes.size
    control = codes < 0
    contribution = rng.normal(size=(num_cells, num_genes))
    for target in range(num_targets):
        contribution[codes == target, target % num_genes] += 1.1
    batch = rng.integers(0, levels, size=num_cells)
    onehot = np.zeros((num_cells, levels))
    onehot[np.arange(num_cells), batch] = 1.0
    # intercept, one continuous covariate, and the batch indicators: the basis
    # the production fit hands the kernel has exactly this shape.
    basis = np.column_stack([np.ones(num_cells), rng.normal(size=num_cells), onehot])
    beta = np.concatenate([[np.log(own_counts.mean() / num_controls)], [0.3], rng.normal(scale=0.4, size=levels)])
    eta_shared = basis @ beta
    delta = rng.normal(scale=0.5, size=num_targets)
    # ``basis[:, 0]`` is the all-ones column, so the intercept direction is e_0
    # and the compact coefficients are ``beta + delta * e_0`` exactly.
    coef = np.tile(beta, (num_targets, 1))
    coef[:, 0] += delta
    common = dict(
        contribution=contribution,
        target_codes=codes,
        control_mask=control,
        num_targets=num_targets,
        # promote every pair: the promoted-block path is the one being replaced
        screen_p_value=1.0,
        gene_block_size=16,
    )
    return common, basis, coef, eta_shared, delta


def _finite_pairs(a, b):
    finite = np.isfinite(a) & np.isfinite(b)
    assert finite.any()
    return finite


def _rel_gap(a, b):
    finite = _finite_pairs(a.log_p_value, b.log_p_value)
    return np.abs(a.log_p_value[finite] - b.log_p_value[finite]) / np.maximum(np.abs(b.log_p_value[finite]), 1e-300)


def test_legacy_and_compact_forms_agree_within_the_solvers_own_noise_in_float64():
    """Given the same logits, switching representation is inside the kernel's noise.

    The two forms hand the kernel logits that agree to ~1e-15, and the null
    moments come back agreeing to ~1e-14. ``log_p`` does not: the saddlepoint
    root is found by a fixed-iteration safeguarded Newton, and on a minority of
    pairs it amplifies summation-order noise in its inputs to a few 1e-7
    relative. Measured here on this fixture: perturbing the *compact* form's own
    coefficients by 1e-15 moves its log p by up to ~4e-7, more than the
    compact-versus-legacy gap (~2e-7), on a different set of pairs. So the
    invariant is not bit-equality but that the representation switch never
    exceeds what the solver already does to itself.
    """

    common, basis, coef, eta_shared, delta = _wide_basis_problem()
    compact = fit_low_moi_propensity_saddlepoint(propensity_coefficients=coef, propensity_basis=basis, **common)
    legacy = fit_low_moi_propensity_saddlepoint(shared_logits=eta_shared, intercepts=delta, **common)

    assert compact.valid.any()
    np.testing.assert_array_equal(compact.valid, legacy.valid)
    np.testing.assert_array_equal(compact.used_fallback, legacy.used_fallback)
    for field in ("null_mean", "null_variance", "observed_sum"):
        left, right = np.asarray(getattr(compact, field)), np.asarray(getattr(legacy, field))
        mask = np.isfinite(left) & np.isfinite(right)
        np.testing.assert_allclose(left[mask], right[mask], rtol=1e-12, atol=1e-12)

    # the kernel's own conditioning floor, measured rather than assumed
    rng = np.random.default_rng(0)
    jittered = fit_low_moi_propensity_saddlepoint(
        propensity_coefficients=coef * (1.0 + 1e-15 * rng.standard_normal(coef.shape)),
        propensity_basis=basis,
        **common,
    )
    self_noise = float(_rel_gap(jittered, compact).max())
    switch = float(_rel_gap(legacy, compact).max())
    assert self_noise > 1e-9, "the fixture must exercise the solver's sensitive pairs"
    assert switch < 1e-6, switch
    assert switch <= 2.0 * self_noise, (switch, self_noise)


def test_float32_compact_storage_is_the_only_difference_from_the_legacy_form():
    """Production stores the compact form in float32; this measures what that costs.

    The legacy form keeps ``eta_shared`` and ``delta`` in float64, so switching
    to it is a precision *gain*. The tolerance here is the float32 rounding of a
    logit of order one propagated through the tail, and is the bound a caller
    comparing old and new tables should expect.
    """

    common, basis, coef, eta_shared, delta = _wide_basis_problem(seed=11)
    basis32 = basis.astype(np.float32)
    coef32 = coef.astype(np.float32)
    rebuilt = basis32.astype(np.float64) @ coef32.astype(np.float64).T          # (cells, targets)
    exact = eta_shared[:, None] + delta[None, :]
    logit_gap = float(np.max(np.abs(rebuilt - exact)))
    assert 0.0 < logit_gap < 2e-5, logit_gap  # float32 rounding, not a bug

    compact = fit_low_moi_propensity_saddlepoint(propensity_coefficients=coef32, propensity_basis=basis32, **common)
    legacy = fit_low_moi_propensity_saddlepoint(shared_logits=eta_shared, intercepts=delta, **common)

    np.testing.assert_array_equal(compact.valid, legacy.valid)
    np.testing.assert_array_equal(compact.used_fallback, legacy.used_fallback)
    finite = _finite_pairs(compact.log_p_value, legacy.log_p_value)
    gap = np.max(np.abs(compact.log_p_value[finite] - legacy.log_p_value[finite]))
    assert gap < 1e-4, gap
    # ...and materially below any decision anyone makes on a p-value.
    assert gap < 1e-3 * max(1.0, np.max(np.abs(legacy.log_p_value[finite])))


def test_targets_without_a_pool_stay_missing_under_the_legacy_form():
    common, basis, coef, eta_shared, delta = _wide_basis_problem(seed=3)
    delta = delta.copy()
    delta[2] = np.nan  # how precompute marks an untestable target
    fit = fit_low_moi_propensity_saddlepoint(shared_logits=eta_shared, intercepts=delta, **common)
    assert not fit.valid[2].any()
    assert np.isnan(fit.log_p_value[2]).all()
    others = [t for t in range(common["num_targets"]) if t != 2]
    assert fit.valid[others].any()
