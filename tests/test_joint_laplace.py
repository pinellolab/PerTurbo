from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from perturbo.core import PerTurboData
from perturbo._internal.joint_laplace import (
    _gene_negative_log_posterior,
    _observed_nb_weights,
    fit_joint_nb_laplace,
    low_moi_marginal_variances,
    prepare_joint_nb_design,
    projected_diagonal_marginal_variances,
)


def test_prepare_joint_nb_design_pools_controls_and_applies_cell_mask() -> None:
    data = PerTurboData(
        counts=jnp.asarray(
            [
                [1, 0],
                [0, 2],
                [3, 1],
                [2, 4],
                [1, 2],
                [5, 0],
            ]
        ),
        pert_id=jnp.asarray([0, 1, 2, 3, 2, 3]),
        pert_names=["NTC_A", "NTC_B", "target_A", "target_B"],
        gene_names=["gene_1", "gene_2"],
        cell_mask=jnp.asarray([True, True, True, True, True, False]),
        size_factors=jnp.asarray([[0.0], [0.1], [-0.1], [0.2], [-0.2], [0.0]]),
        covariates=jnp.asarray([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]),
        covariate_names=["batch_B"],
    )

    design = prepare_joint_nb_design(
        data,
        control_perturbations=["NTC_A", "NTC_B"],
        dispersion=np.asarray([5.0, 10.0]),
    )

    assert design.target_names == ("target_A", "target_B")
    assert design.nuisance_names == ("intercept", "batch_B")
    assert design.gene_names == ("gene_1", "gene_2")
    assert design.counts.shape == (5, 2)
    np.testing.assert_array_equal(
        np.asarray(design.target_design),
        np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
    )
    np.testing.assert_array_equal(np.asarray(design.control_mask), [True, True, False, False, False])


def test_prepare_joint_nb_design_excludes_unassigned_matrix_rows_and_rejects_high_moi() -> None:
    base = PerTurboData(
        counts=jnp.ones((4, 1), dtype=jnp.int32),
        pert_id=jnp.asarray(
            [
                [1, 0],
                [0, 1],
                [0, 0],
                [0, 1],
            ]
        ),
        pert_names=["NTC", "target"],
        gene_names=["gene"],
        size_factors=jnp.zeros((4, 1)),
    )
    design = prepare_joint_nb_design(
        base,
        control_perturbations=["NTC"],
        dispersion=np.asarray([4.0]),
    )
    assert design.num_cells == 3

    high_moi = PerTurboData(
        counts=base.counts,
        pert_id=jnp.asarray(
            [
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 1],
            ]
        ),
        pert_names=base.pert_names,
        gene_names=base.gene_names,
        size_factors=base.size_factors,
    )
    with pytest.raises(ValueError, match="low-MOI"):
        prepare_joint_nb_design(
            high_moi,
            control_perturbations=["NTC"],
            dispersion=np.asarray([4.0]),
        )

    allowed = prepare_joint_nb_design(
        high_moi,
        control_perturbations=["NTC"],
        dispersion=np.asarray([4.0]),
        allow_high_moi=True,
    )
    np.testing.assert_array_equal(np.asarray(allowed.target_design[:, 0]), [0.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(np.asarray(allowed.control_mask), [True, False, False, False])


def test_low_moi_marginal_variances_match_dense_hessian_inverse() -> None:
    counts = jnp.asarray([0, 2, 1, 5, 3, 4, 1, 0], dtype=jnp.float32)
    target_design = jnp.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    nuisance_design = jnp.column_stack(
        [jnp.ones(counts.shape[0]), jnp.asarray([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0.5, -0.5])]
    )
    offset = jnp.asarray([0.1, -0.1, 0.0, 0.2, -0.2, 0.1, 0.0, -0.1])
    theta = 7.0
    parameters = jnp.asarray([0.4, -0.15, -0.5, 0.7])
    effect_prior_scale = 1.3
    nuisance_prior_scale = 8.0

    def objective(value: jnp.ndarray) -> jnp.ndarray:
        return _gene_negative_log_posterior(
            value,
            counts=counts,
            target_design=target_design,
            nuisance_design=nuisance_design,
            offset=offset,
            theta=jnp.asarray(theta),
            effect_prior_scale=effect_prior_scale,
            nuisance_prior_scale=nuisance_prior_scale,
            cell_chunk_size=None,
        )

    dense_hessian = np.asarray(jax.hessian(objective)(parameters), dtype=np.float64)
    dense_variances = np.diag(np.linalg.inv(dense_hessian))[2:]
    eta = np.asarray(
        offset
        + nuisance_design @ parameters[:2]
        + target_design @ parameters[2:]
    )
    structured_variances = low_moi_marginal_variances(
        counts=np.asarray(counts),
        eta=eta,
        theta=theta,
        target_design=np.asarray(target_design),
        nuisance_design=np.asarray(nuisance_design),
        effect_prior_scale=effect_prior_scale,
        nuisance_prior_scale=nuisance_prior_scale,
    )

    np.testing.assert_allclose(structured_variances, dense_variances, rtol=2e-5, atol=2e-6)


def test_projected_diagonal_variances_match_exact_random_high_moi_marginals() -> None:
    rng = np.random.default_rng(51)
    num_cells = 30_000
    num_effects = 6
    target_design = rng.binomial(1, 0.18, size=(num_cells, num_effects)).astype(float)
    technical = rng.normal(size=num_cells)
    nuisance_design = np.column_stack([np.ones(num_cells), technical])
    effects = np.linspace(-0.3, 0.3, num_effects)
    eta = 0.5 + 0.15 * technical + target_design @ effects
    theta = 12.0
    mean = np.exp(eta)
    counts = rng.negative_binomial(theta, theta / (theta + mean))
    effect_prior_scale = 4.0
    nuisance_prior_scale = 10.0

    approximation = projected_diagonal_marginal_variances(
        counts=counts,
        eta=eta,
        theta=theta,
        target_design=target_design,
        nuisance_design=nuisance_design,
        effect_prior_scale=effect_prior_scale,
        nuisance_prior_scale=nuisance_prior_scale,
        correlation_diagnostic_pairs=100,
    )

    weights = _observed_nb_weights(counts, eta, theta)
    nuisance_information = nuisance_design.T @ (weights[:, None] * nuisance_design)
    nuisance_information += np.eye(nuisance_design.shape[1]) / nuisance_prior_scale**2
    nuisance_cross = nuisance_design.T @ (weights[:, None] * target_design)
    effect_information = target_design.T @ (weights[:, None] * target_design)
    effect_information += np.eye(num_effects) / effect_prior_scale**2
    schur = effect_information - nuisance_cross.T @ np.linalg.solve(
        nuisance_information,
        nuisance_cross,
    )
    exact_variance = np.diag(np.linalg.inv(schur))

    assert approximation.num_correlation_pairs == 15
    assert approximation.max_abs_projected_correlation < 0.025
    np.testing.assert_allclose(approximation.variance, exact_variance, rtol=2e-3)


def test_projected_diagonal_diagnostic_detects_correlated_effect_columns() -> None:
    rng = np.random.default_rng(63)
    shared = rng.binomial(1, 0.3, size=500).astype(float)
    target_design = np.column_stack([shared, shared, rng.binomial(1, 0.3, size=500)])
    eta = np.full(500, 0.5)
    theta = 8.0
    mean = np.exp(eta)
    counts = rng.negative_binomial(theta, theta / (theta + mean))

    approximation = projected_diagonal_marginal_variances(
        counts=counts,
        eta=eta,
        theta=theta,
        target_design=target_design,
        nuisance_design=np.ones((500, 1)),
        effect_prior_scale=2.0,
        nuisance_prior_scale=None,
        correlation_diagnostic_pairs=10,
    )

    assert approximation.num_correlation_pairs == 3
    assert approximation.max_abs_projected_correlation > 0.999


def _simulated_low_moi_data(seed: int = 4) -> tuple[PerTurboData, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cells_per_group = 240
    labels = np.repeat(np.arange(3), cells_per_group)
    target_design = np.column_stack([labels == 1, labels == 2]).astype(float)
    covariate = rng.normal(size=labels.size)
    nuisance = np.column_stack([np.ones(labels.size), covariate])
    offset = rng.normal(scale=0.25, size=labels.size)
    nuisance_coefficients = np.asarray(
        [
            [0.8, 1.2],
            [0.25, -0.2],
        ]
    )
    true_effects = np.asarray(
        [
            [-0.65, 0.45],
            [0.55, -0.5],
        ]
    )
    theta = np.asarray([9.0, 14.0])
    eta = offset[:, None] + nuisance @ nuisance_coefficients + target_design @ true_effects
    mean = np.exp(eta)
    probability = theta[None, :] / (theta[None, :] + mean)
    counts = rng.negative_binomial(theta[None, :], probability)
    data = PerTurboData(
        counts=jnp.asarray(counts),
        pert_id=jnp.asarray(labels),
        pert_names=["NTC", "target_A", "target_B"],
        gene_names=["gene_1", "gene_2"],
        size_factors=jnp.asarray(offset[:, None]),
        covariates=jnp.asarray(covariate[:, None]),
        covariate_names=["technical"],
    )
    return data, theta, true_effects


def test_joint_nb_laplace_recovers_effects_and_is_cell_chunk_invariant() -> None:
    data, theta, true_effects = _simulated_low_moi_data()
    design = prepare_joint_nb_design(
        data,
        control_perturbations=["NTC"],
        dispersion=theta,
    )

    full = fit_joint_nb_laplace(
        design,
        effect_prior_scale=10.0,
        cell_chunk_size=None,
    )
    chunked = fit_joint_nb_laplace(
        design,
        effect_prior_scale=10.0,
        cell_chunk_size=73,
    )

    assert np.all(np.asarray(full.converged))
    assert np.all(np.asarray(full.posterior_scale) > 0)
    assert full.covariance_approximation == "exact_low_moi"
    assert np.all(np.isnan(np.asarray(full.max_abs_projected_effect_correlation)))
    np.testing.assert_allclose(np.asarray(full.posterior_mean), true_effects, atol=0.16)
    # Cell chunking changes the order the Hessian and gradient are accumulated in,
    # so the two agree only to float32. On jax 0.11 the largest gap is 7e-4 absolute
    # on effects of order 1, where jax 0.9 agreed to 2e-4. The property under test is
    # that chunking does not change the answer, not that the summation order is fixed.
    np.testing.assert_allclose(
        np.asarray(chunked.posterior_mean),
        np.asarray(full.posterior_mean),
        rtol=5e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(chunked.posterior_scale),
        np.asarray(full.posterior_scale),
        rtol=5e-3,
        atol=2e-3,
    )


def test_joint_nb_laplace_uses_projected_diagonal_covariance_for_high_moi() -> None:
    rng = np.random.default_rng(72)
    num_cells = 2_000
    target_design = rng.binomial(1, 0.22, size=(num_cells, 3)).astype(np.int8)
    ntc = rng.binomial(1, 0.25, size=num_cells).astype(np.int8)
    assignments = np.column_stack([ntc, target_design])
    unassigned = assignments.sum(axis=1) == 0
    assignments[unassigned, 0] = 1
    technical = rng.normal(size=num_cells)
    offset = rng.normal(scale=0.15, size=num_cells)
    true_effects = np.asarray([[-0.45], [0.35], [0.6]])
    eta = offset + 0.7 + 0.2 * technical + target_design @ true_effects[:, 0]
    theta = np.asarray([11.0])
    mean = np.exp(eta)
    counts = rng.negative_binomial(theta[0], theta[0] / (theta[0] + mean))[:, None]
    data = PerTurboData(
        counts=jnp.asarray(counts),
        pert_id=jnp.asarray(assignments),
        pert_names=["NTC", "target_A", "target_B", "target_C"],
        gene_names=["gene"],
        size_factors=jnp.asarray(offset[:, None]),
        covariates=jnp.asarray(technical[:, None]),
        covariate_names=["technical"],
    )
    design = prepare_joint_nb_design(
        data,
        control_perturbations=["NTC"],
        dispersion=theta,
        allow_high_moi=True,
        require_control_cells=False,
    )
    fit = fit_joint_nb_laplace(
        design,
        effect_prior_scale=10.0,
        covariance_approximation="auto",
        correlation_diagnostic_pairs=20,
    )

    assert fit.covariance_approximation == "projected_diagonal"
    assert bool(fit.converged[0])
    assert int(fit.num_projected_effect_correlation_pairs[0]) == 3
    assert float(fit.max_abs_projected_effect_correlation[0]) < 0.1
    np.testing.assert_allclose(np.asarray(fit.posterior_mean), true_effects, atol=0.13)
    assert np.all(np.asarray(fit.posterior_scale) > 0)
