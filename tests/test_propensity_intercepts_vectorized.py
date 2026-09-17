"""The batched device bisection reproduces the scalar per-target reference."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from perturbo._internal.score_resampling import (
    _bisect_propensity_intercepts,
    _padded_own_rows,
    _solve_propensity_intercept,
)


def _screen(seed: int, *, num_controls: int = 700, num_targets: int = 9, levels: int = 30):
    rng = np.random.default_rng(seed)
    own_counts = rng.integers(0, 60, size=num_targets)   # a zero-cell target is untestable
    codes = np.concatenate([np.full(num_controls, -1), np.repeat(np.arange(num_targets), own_counts)])
    control = codes < 0
    n = codes.size
    batch = rng.integers(0, levels, size=n)
    basis = np.column_stack([np.ones(n), rng.normal(size=n), np.eye(levels)[batch]])
    beta = np.concatenate([[-2.0], [0.4], rng.normal(scale=0.6, size=levels)])
    eta = basis @ beta
    return codes, control, eta


def _reference(codes, control, eta, num_targets):
    out = np.full(num_targets, np.nan)
    for t in range(num_targets):
        own = codes == t
        pool = control | own
        selected = int(np.count_nonzero(own & pool))
        if selected == 0 or selected == int(np.count_nonzero(pool)):
            continue
        out[t] = _solve_propensity_intercept(eta[pool], selected)
    return out


def _vectorized(codes, control, eta, num_targets):
    own_rows = np.flatnonzero((codes >= 0) & ~control)
    own_codes = codes[own_rows].astype(np.int64)
    table, mask = _padded_own_rows(own_rows, own_codes, num_targets)
    selected = np.bincount(codes[codes >= 0], minlength=num_targets).astype(np.float64)
    pool_size = float(np.count_nonzero(control)) + np.bincount(own_codes, minlength=num_targets)
    testable = (selected > 0) & (selected < pool_size)
    out = np.full(num_targets, np.nan)
    block = np.flatnonzero(testable)
    out[block] = np.asarray(
        _bisect_propensity_intercepts(
            jnp.asarray(eta[control]), jnp.asarray(eta[table[block]]), jnp.asarray(mask[block]),
            jnp.asarray(selected[block]),
        )
    )
    return out


def test_batched_bisection_matches_the_scalar_reference_per_target():
    for seed in (1, 2, 3):
        codes, control, eta = _screen(seed)
        ref = _reference(codes, control, eta, 9)
        vec = _vectorized(codes, control, eta, 9)
        np.testing.assert_array_equal(np.isnan(ref), np.isnan(vec))
        finite = ~np.isnan(ref)
        assert finite.sum() >= 6
        np.testing.assert_allclose(vec[finite], ref[finite], rtol=0.0, atol=1e-9)


def test_intercepts_reproduce_the_observed_count_on_each_pool():
    codes, control, eta = _screen(7)
    vec = _vectorized(codes, control, eta, 9)
    for t in np.flatnonzero(~np.isnan(vec)):
        pool = control | (codes == t)
        fitted = np.sum(1.0 / (1.0 + np.exp(-(eta[pool] + vec[t]))))
        assert abs(fitted - np.count_nonzero(codes == t)) < 1e-6


def test_padded_table_covers_every_own_cell_exactly_once():
    codes, control, eta = _screen(4)
    own_rows = np.flatnonzero((codes >= 0) & ~control)
    table, mask = _padded_own_rows(own_rows, codes[own_rows].astype(np.int64), 9)
    assert mask.sum() == own_rows.size
    assert np.array_equal(np.sort(table[mask]), np.sort(own_rows))
    for t in range(9):
        assert np.array_equal(np.sort(table[t, mask[t]]), np.flatnonzero(codes == t))
