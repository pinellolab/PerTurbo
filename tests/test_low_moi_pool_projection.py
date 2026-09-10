"""The low-MOI saddlepoint's pool projection.

Two properties. First, packed and corrected cumulants and tails must agree with
a brute-force per-target construction that refits the nuisance on the pool
(controls plus the target's own cells) and evaluates the exact CGF on the
corrected rows. Second, the reason it exists: with a control pool that is not
large next to a target, the uncorrected statistic is anti-conservative under
the sharp null and the corrected one is not.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from perturbo._internal.saddlepoint import (  # noqa: E402
    fit_low_moi_propensity_saddlepoint,
    propensity_saddlepoint_log_two_sided,
)


def _nb_residual_weight(counts, mu, theta):
    weight = mu * theta / (mu + theta)
    residual = theta * (counts - mu) / (theta + mu)
    return residual, weight


def _fit_null_intercepts(counts, offsets, design, theta, iterations=25):
    """Per-gene Fisher scoring for the nuisance coefficients on the given cells."""
    n_cells, n_genes = counts.shape
    q = design.shape[1]
    coef = np.zeros((q, n_genes))
    coef[0] = np.log(np.maximum(counts.mean(axis=0), 1e-3)) - offsets.mean()
    for _ in range(iterations):
        eta = offsets[:, None] + design @ coef
        mu = np.exp(eta)
        residual, weight = _nb_residual_weight(counts, mu, theta)
        for g in range(n_genes):
            info = design.T @ (weight[:, g, None] * design)
            score = design.T @ residual[:, g]
            coef[:, g] += np.linalg.solve(info + 1e-9 * np.eye(q), score)
    eta = offsets[:, None] + design @ coef
    mu = np.exp(eta)
    return coef, mu


def _simulate(rng, *, n_controls, n_targets, cells_per_target, n_genes, q):
    n_cells = n_controls + n_targets * cells_per_target
    theta = np.exp(rng.normal(1.0, 0.3, size=n_genes))
    beta0 = rng.normal(0.5, 0.8, size=n_genes)
    offsets = rng.normal(0.0, 0.3, size=n_cells)
    design = np.ones((n_cells, q))
    if q > 1:
        design[:, 1:] = rng.normal(0.0, 1.0, size=(n_cells, q - 1))
    coef = np.zeros((q, n_genes))
    coef[0] = beta0
    if q > 1:
        coef[1:] = rng.normal(0.0, 0.2, size=(q - 1, n_genes))
    mu = np.exp(offsets[:, None] + design @ coef)
    counts = rng.negative_binomial(theta, theta / (theta + mu)).astype(np.float64)
    codes = np.full(n_cells, -1, dtype=np.int64)
    codes[n_controls:] = np.repeat(np.arange(n_targets), cells_per_target)
    control = np.zeros(n_cells, dtype=bool)
    control[:n_controls] = True
    return counts, offsets, design, theta, codes, control


def _control_projected_contribution(counts, offsets, design, theta, control):
    coef, _ = _fit_null_intercepts(counts[control], offsets[control], design[control], theta)
    mu = np.exp(offsets[:, None] + design @ coef)
    residual, weight = _nb_residual_weight(counts, mu, theta)
    q = design.shape[1]
    n_genes = counts.shape[1]
    information = np.zeros((n_genes, q, q))
    nuisance_score = np.zeros((q, n_genes))
    for g in range(n_genes):
        information[g] = design[control].T @ (weight[control, g, None] * design[control])
        nuisance_score[:, g] = design[control].T @ residual[control, g]
    direction = np.stack([np.linalg.solve(information[g], nuisance_score[:, g]) for g in range(n_genes)], axis=1)
    contribution = residual - weight * (design @ direction)
    return contribution, weight, information


def _propensity(codes, control, n_targets):
    """Intercept-only selection model: shared logits zero, per-target intercept matching the count."""
    shared = np.zeros(codes.shape[0])
    intercepts = np.full(n_targets, np.nan)
    for t in range(n_targets):
        own = (codes == t) & ~control
        pool = control | own
        n_pool = int(pool.sum())
        n_own = int((codes == t).sum())
        intercepts[t] = np.log(n_own / (n_pool - n_own))
    return shared, intercepts


def _brute_force(contribution, weight, design, information, codes, control, shared, intercepts, n_targets, gene):
    """Per-target pool refit and exact CGF, one gene: returns (log p, observed, mean, variance)."""
    out = []
    q = design.shape[1]
    for t in range(n_targets):
        own = (codes == t) & ~control
        own_rows = np.flatnonzero(own)
        pool_rows = np.concatenate([np.flatnonzero(control), own_rows])
        score = design[own_rows].T @ contribution[own_rows, gene]
        info = information[gene] + design[own_rows].T @ (weight[own_rows, gene, None] * design[own_rows])
        e = np.linalg.solve(info, score)
        corrected = contribution[:, gene] - weight[:, gene] * (design @ e)
        observed = corrected[codes == t].sum()
        logits = shared[pool_rows] + intercepts[t]
        pi = 1.0 / (1.0 + np.exp(-logits))
        c = corrected[pool_rows]
        mean = np.sum(pi * c)
        variance = np.sum(pi * (1 - pi) * c**2)
        log_p, valid = propensity_saddlepoint_log_two_sided(
            jnp.asarray([observed]), jnp.asarray(c)[:, None], jnp.asarray(logits)[:, None]
        )
        out.append((float(log_p[0]), bool(valid[0]), observed, mean, variance))
    return out


@pytest.mark.parametrize("q", [1, 2])
def test_packed_projection_matches_per_target_pool_refit(q):
    rng = np.random.default_rng(3)
    n_targets, cells_per_target, n_controls, n_genes = 6, 30, 90, 5
    counts, offsets, design, theta, codes, control = _simulate(
        rng, n_controls=n_controls, n_targets=n_targets, cells_per_target=cells_per_target, n_genes=n_genes, q=q
    )
    contribution, weight, information = _control_projected_contribution(counts, offsets, design, theta, control)
    shared, intercepts = _propensity(codes, control, n_targets)
    fit = fit_low_moi_propensity_saddlepoint(
        contribution=contribution,
        target_codes=codes,
        control_mask=control,
        shared_logits=shared,
        intercepts=intercepts,
        num_targets=n_targets,
        screen_p_value=1.0,  # evaluate every pair exactly
        gene_block_size=8,
        weight=weight,
        nuisance_design=design,
        control_information=information,
    )
    for gene in range(n_genes):
        reference = _brute_force(
            contribution, weight, design, information, codes, control, shared, intercepts, n_targets, gene
        )
        for t, (log_p, valid, observed, mean, variance) in enumerate(reference):
            assert np.isclose(fit.observed_sum[t, gene], observed, rtol=1e-9, atol=1e-9)
            assert np.isclose(fit.null_mean[t, gene], mean, rtol=1e-9, atol=1e-9)
            assert np.isclose(fit.null_variance[t, gene], variance, rtol=1e-9, atol=1e-9)
            if valid:
                assert fit.valid[t, gene]
                assert np.isclose(fit.log_p_value[t, gene], log_p, rtol=1e-6, atol=1e-8)


def test_categorical_projection_matches_dense_one_hot():
    """A categorical batch nuisance must give the same answer as its one-hot dense design."""
    rng = np.random.default_rng(5)
    n_targets, cells_per_target, n_controls, n_genes, n_batches = 5, 24, 120, 4, 3
    n_cells = n_controls + n_targets * cells_per_target
    batch = rng.integers(0, n_batches, size=n_cells)
    design = np.eye(n_batches)[batch]
    theta = np.exp(rng.normal(1.0, 0.3, size=n_genes))
    coef = rng.normal(0.5, 0.6, size=(n_batches, n_genes))
    offsets = rng.normal(0.0, 0.3, size=n_cells)
    mu = np.exp(offsets[:, None] + design @ coef)
    counts = rng.negative_binomial(theta, theta / (theta + mu)).astype(np.float64)
    codes = np.full(n_cells, -1, dtype=np.int64)
    codes[n_controls:] = np.repeat(np.arange(n_targets), cells_per_target)
    control = np.zeros(n_cells, dtype=bool)
    control[:n_controls] = True
    contribution, weight, information = _control_projected_contribution(counts, offsets, design, theta, control)
    shared, intercepts = _propensity(codes, control, n_targets)
    common = dict(
        contribution=contribution,
        target_codes=codes,
        control_mask=control,
        shared_logits=shared,
        intercepts=intercepts,
        num_targets=n_targets,
        screen_p_value=1.0,
        gene_block_size=8,
        weight=weight,
    )
    dense = fit_low_moi_propensity_saddlepoint(**common, nuisance_design=design, control_information=information)
    diagonal = np.stack([information[g].diagonal() for g in range(n_genes)], axis=1)  # (batches, genes)
    categorical = fit_low_moi_propensity_saddlepoint(**common, batch_codes=batch, control_information=diagonal)
    np.testing.assert_allclose(categorical.observed_sum, dense.observed_sum, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(categorical.null_mean, dense.null_mean, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(categorical.null_variance, dense.null_variance, rtol=1e-9, atol=1e-9)
    # The dense path drops the cubic term of the corrected third cumulant (it is
    # third order in the shift and only steers the screen); the diagonal path keeps
    # it. Skewness therefore agrees to second order, the exact tails exactly.
    np.testing.assert_allclose(categorical.null_skewness, dense.null_skewness, rtol=1e-2, atol=1e-6)
    np.testing.assert_allclose(categorical.log_p_value, dense.log_p_value, rtol=1e-6, atol=1e-8)


def test_projection_is_identity_when_targets_are_controls():
    """A target whose cells are all controls has nothing out of sample; the statistic must not move."""
    rng = np.random.default_rng(11)
    n_targets, cells_per_target, n_controls, n_genes = 4, 20, 200, 3
    counts, offsets, design, theta, codes, control = _simulate(
        rng, n_controls=n_controls, n_targets=n_targets, cells_per_target=cells_per_target, n_genes=n_genes, q=2
    )
    control[:] = True  # every cell is a control, targets are labelled subsets of the controls
    contribution, weight, information = _control_projected_contribution(counts, offsets, design, theta, control)
    shared, intercepts = _propensity(codes, control, n_targets)
    common = dict(
        contribution=contribution, target_codes=codes, control_mask=control, shared_logits=shared,
        intercepts=intercepts, num_targets=n_targets, screen_p_value=1.0, gene_block_size=8,
    )
    legacy = fit_low_moi_propensity_saddlepoint(**common)
    corrected = fit_low_moi_propensity_saddlepoint(**common, weight=weight, nuisance_design=design, control_information=information)
    np.testing.assert_allclose(corrected.observed_sum, legacy.observed_sum, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(corrected.null_variance, legacy.null_variance, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(corrected.log_p_value, legacy.log_p_value, rtol=1e-10, atol=1e-10)


def test_projection_removes_small_pool_anticonservativeness():
    """Sharp null, 300 controls, 100-cell targets: legacy inflates, corrected does not."""
    rng = np.random.default_rng(2024)
    n_targets, cells_per_target, n_controls, n_genes = 40, 100, 300, 60
    counts, offsets, design, theta, codes, control = _simulate(
        rng, n_controls=n_controls, n_targets=n_targets, cells_per_target=cells_per_target, n_genes=n_genes, q=1
    )
    contribution, weight, information = _control_projected_contribution(counts, offsets, design, theta, control)
    shared, intercepts = _propensity(codes, control, n_targets)
    common = dict(
        contribution=contribution, target_codes=codes, control_mask=control, shared_logits=shared,
        intercepts=intercepts, num_targets=n_targets, screen_p_value=1.0, gene_block_size=64,
        # The inflation this test measures was quantified under the symmetric
        # two-sided convention; the equal-tail default removes part of it on
        # its own, so pin the convention to keep the check about the projection.
        two_sided="symmetric",
    )
    legacy = fit_low_moi_propensity_saddlepoint(**common)
    corrected = fit_low_moi_propensity_saddlepoint(**common, weight=weight, nuisance_design=design, control_information=information)
    legacy_rate = float(np.nanmean(legacy.p_value < 0.05))
    corrected_rate = float(np.nanmean(corrected.p_value < 0.05))
    # 2,400 null pairs: the binomial SE at 0.05 is 0.0045. W_own/W_c = 1/3 here,
    # so the legacy statistic is inflated by a variance ratio near (1 + 1/3).
    assert legacy_rate > 0.065, legacy_rate
    assert 0.035 < corrected_rate < 0.065, corrected_rate
