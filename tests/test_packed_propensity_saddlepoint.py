"""The packed (element, gene) propensity saddlepoint reproduces the per-pair one.

Packing candidates from many elements into one block changes how the work is
scheduled, not what is computed: every column still reduces over its own pool
with its own logits. These tests pin that down against the per-column kernel,
which is itself validated against exact enumeration in test_propensity_tails.

Tolerances are 1e-6 relative in log p rather than machine precision. The
arithmetic per pair is the same, but the pool rows are summed in a different
order and the logits come out of a different BLAS shape, and the bracketed
Newton solve with a fixed iteration count amplifies those last-bit differences
- most visibly near the null mean, where the Lugannani-Rice correction is a
difference of two large reciprocals. Measured: 1e-7 relative at p ~ 0.5,
1e-8 in the tails. The same sensitivity exists between two block sizes of one
implementation, so it is a property of the solver, not of the packing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from perturbo._internal.saddlepoint import (
    fit_high_moi_propensity_saddlepoint,
    fit_low_moi_propensity_saddlepoint,
    propensity_saddlepoint_log_two_sided,
)


def test_per_pair_logits_match_the_shared_logit_kernel_column_by_column():
    rng = np.random.default_rng(0)
    pool, columns = 300, 5
    contribution = jnp.asarray(rng.normal(size=(pool, columns)))
    logits = jnp.asarray(rng.normal(loc=-3.0, size=(pool, columns)))
    observed = jnp.asarray(rng.normal(scale=4.0, size=columns))

    packed_log_p, packed_valid = propensity_saddlepoint_log_two_sided(observed, contribution, logits)
    for column in range(columns):
        log_p, valid = propensity_saddlepoint_log_two_sided(
            observed[column : column + 1],
            contribution[:, column : column + 1],
            logits[:, column],
        )
        assert bool(valid[0]) == bool(packed_valid[column])
        np.testing.assert_allclose(np.asarray(log_p)[0], np.asarray(packed_log_p)[column], rtol=1e-6, atol=1e-7)


def _low_moi_problem(seed: int = 3):
    rng = np.random.default_rng(seed)
    num_controls, num_targets, num_genes = 400, 6, 9
    own_counts = rng.integers(8, 40, size=num_targets)
    codes = np.concatenate([np.full(num_controls, -1), np.repeat(np.arange(num_targets), own_counts)])
    num_cells = codes.size
    control = codes < 0
    contribution = rng.normal(size=(num_cells, num_genes))
    # Give a few pairs a real shift so the screen promotes something.
    for target in range(num_targets):
        contribution[codes == target, target % num_genes] += 1.2
    shared = rng.normal(scale=0.5, size=num_cells)
    intercepts = np.log(own_counts / num_controls) + rng.normal(scale=0.1, size=num_targets)
    return contribution, codes, control, shared, intercepts, num_targets


def test_low_moi_packed_fit_matches_per_target_pools():
    contribution, codes, control, shared, intercepts, num_targets = _low_moi_problem()
    fit = fit_low_moi_propensity_saddlepoint(
        contribution=contribution,
        target_codes=codes,
        control_mask=control,
        shared_logits=shared,
        intercepts=intercepts,
        num_targets=num_targets,
        screen_p_value=0.5,
        gene_block_size=7,  # forces several blocks and a padded final one
    )
    evaluated = fit.valid & ~fit.used_fallback
    assert evaluated.sum() > 5, "the test needs promoted pairs to compare"
    assert (~evaluated & fit.valid).sum() > 0, "and some the screen kept"

    for target in range(num_targets):
        rows = np.flatnonzero(control | (codes == target))
        cells = np.flatnonzero(codes == target)
        pool = jnp.asarray(contribution[rows])
        logits = jnp.asarray(shared[rows] + intercepts[target])
        selection = np.asarray(jax.nn.sigmoid(logits))
        bernoulli = selection * (1.0 - selection)
        observed = contribution[cells].sum(axis=0)
        np.testing.assert_allclose(fit.observed_sum[target], observed, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fit.null_mean[target], selection @ contribution[rows], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            fit.null_variance[target], bernoulli @ np.square(contribution[rows]), rtol=1e-10, atol=1e-12
        )
        for gene in np.flatnonzero(evaluated[target]):
            log_p, valid = propensity_saddlepoint_log_two_sided(
                jnp.asarray(observed[gene : gene + 1]), pool[:, gene : gene + 1], logits
            )
            assert bool(valid[0]) == bool(fit.valid[target, gene])
            np.testing.assert_allclose(fit.log_p_value[target, gene], np.asarray(log_p)[0], rtol=1e-6, atol=1e-7)


def test_low_moi_fit_is_invariant_to_the_block_size():
    contribution, codes, control, shared, intercepts, num_targets = _low_moi_problem(seed=11)
    common = dict(
        contribution=contribution,
        target_codes=codes,
        control_mask=control,
        shared_logits=shared,
        intercepts=intercepts,
        num_targets=num_targets,
        screen_p_value=0.3,
    )
    one = fit_low_moi_propensity_saddlepoint(gene_block_size=1, **common)
    many = fit_low_moi_propensity_saddlepoint(gene_block_size=64, **common)
    np.testing.assert_array_equal(one.valid, many.valid)
    np.testing.assert_array_equal(one.used_fallback, many.used_fallback)
    np.testing.assert_allclose(one.log_p_value, many.log_p_value, rtol=1e-6, atol=1e-7, equal_nan=True)


def test_low_moi_targets_without_a_pool_are_left_missing():
    contribution, codes, control, shared, intercepts, num_targets = _low_moi_problem(seed=5)
    intercepts = intercepts.copy()
    intercepts[2] = np.nan
    fit = fit_low_moi_propensity_saddlepoint(
        contribution=contribution,
        target_codes=codes,
        control_mask=control,
        shared_logits=shared,
        intercepts=intercepts,
        num_targets=num_targets,
    )
    assert not fit.valid[2].any()
    assert np.isnan(fit.p_value[2]).all() and np.isnan(fit.null_mean[2]).all()
    assert fit.valid[[0, 1, 3, 4, 5]].all()


def _high_moi_problem(seed: int = 7):
    rng = np.random.default_rng(seed)
    num_cells, num_genes, num_elements, q = 500, 6, 5, 2
    residual = rng.normal(size=(num_cells, num_genes))
    weight = rng.uniform(0.5, 2.0, size=(num_cells, num_genes))
    nuisance = np.column_stack([np.ones(num_cells), rng.normal(size=num_cells)])
    inverse = np.stack([np.linalg.inv(nuisance.T @ (nuisance * weight[:, [g]]) ) for g in range(num_genes)])
    score = nuisance.T @ residual
    membership = rng.random((num_cells, num_elements)) < 0.08
    cell_index, element_index = np.nonzero(membership)
    basis = np.column_stack([np.ones(num_cells), rng.normal(size=num_cells)])
    coefficients = np.column_stack([np.full(num_elements, -2.5), rng.normal(scale=0.3, size=num_elements)])
    return dict(
        score_residual=residual,
        observation_weight=weight,
        nuisance_design=nuisance,
        nuisance_information_inverse=inverse,
        nuisance_score=score,
        cell_index=cell_index,
        element_index=element_index,
        num_elements=num_elements,
        propensity_coefficients=coefficients,
        propensity_basis=basis,
        screen_p_value=1.0,
    )


def test_high_moi_packed_fit_is_invariant_to_the_block_size():
    problem = _high_moi_problem()
    one = fit_high_moi_propensity_saddlepoint(gene_block_size=1, **problem)
    many = fit_high_moi_propensity_saddlepoint(gene_block_size=64, **problem)
    assert (one.valid & ~one.used_fallback).all(), "screen_p_value=1 evaluates every pair"
    np.testing.assert_array_equal(one.valid, many.valid)
    np.testing.assert_allclose(one.log_p_value, many.log_p_value, rtol=1e-6, atol=1e-7)


def test_high_moi_packed_fit_matches_the_per_element_kernel():
    problem = _high_moi_problem(seed=9)
    fit = fit_high_moi_propensity_saddlepoint(gene_block_size=4, **problem)
    residual = problem["score_residual"]
    weight = problem["observation_weight"]
    nuisance = problem["nuisance_design"]
    direction = np.einsum("gqr,rg->gq", problem["nuisance_information_inverse"], problem["nuisance_score"])
    contribution = residual - weight * (nuisance @ direction.T)
    for element in range(problem["num_elements"]):
        cells = problem["cell_index"][problem["element_index"] == element]
        observed = contribution[cells].sum(axis=0)
        np.testing.assert_allclose(fit.observed_sum[element], observed, rtol=1e-10, atol=1e-10)
        logits = np.clip(problem["propensity_basis"] @ problem["propensity_coefficients"][element], -30.0, 30.0)
        log_p, valid = propensity_saddlepoint_log_two_sided(
            jnp.asarray(observed), jnp.asarray(contribution), jnp.asarray(logits)
        )
        np.testing.assert_array_equal(np.asarray(valid), fit.valid[element])
        np.testing.assert_allclose(fit.log_p_value[element], np.asarray(log_p), rtol=1e-6, atol=1e-7)
