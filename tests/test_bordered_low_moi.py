"""Original-coordinate parity for mixed-design low-MOI inference."""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from perturbo.core import PerTurboData
from perturbo.crt import fit_shared_propensity_coefficients
from perturbo._internal import bordered, bordered_scores, saddlepoint, score_resampling
from perturbo._internal.bordered_scores import segmented_information, segmented_transpose_dot
from perturbo._internal.joint_laplace import prepare_joint_nb_design


@contextmanager
def _x64():
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def _mixed_design(seed=123, continuous=2):
    rng = np.random.default_rng(seed)
    n, groups, genes = 480, 6, 3
    batch = np.tile(np.arange(groups), n // groups)
    covariates = np.column_stack([np.eye(groups)[batch, 1:], rng.normal(size=(n, continuous))])
    z = np.column_stack([np.ones(n), covariates]).astype(np.float32)
    beta = rng.normal(scale=0.12, size=(z.shape[1], genes))
    beta[0] = 1.0
    labels = np.repeat([0, 1, 2], [300, 90, 90]).astype(np.int32)
    offsets = rng.normal(scale=0.15, size=(n, 1))
    eta = z @ beta + offsets
    eta[labels == 1, 0] -= 0.7
    theta = np.full(genes, 8.0)
    mean = np.exp(eta)
    counts = rng.negative_binomial(theta, theta / (theta + mean))
    data = PerTurboData(
        counts=jnp.asarray(counts), pert_id=jnp.asarray(labels),
        pert_names=["NTC", "a", "b"], gene_names=["g1", "g2", "g3"],
        size_factors=jnp.asarray(offsets), covariates=jnp.asarray(covariates),
        covariate_names=[f"cov{k}" for k in range(covariates.shape[1])],
    )
    return prepare_joint_nb_design(data, control_perturbations=["NTC"], dispersion=theta)


@pytest.mark.parametrize("continuous", [2, 4])
def test_resampling_matches_dense_with_multiple_continuous_covariates(monkeypatch, continuous):
    design = _mixed_design(continuous=continuous)
    kwargs = dict(num_resamples=39, seed=73, nuisance_prior_scale=3.0,
                  curvature_jitter=1e-4, return_resampled_scores=True, maxiter=100)
    structured = score_resampling.run_low_moi_score_permutations(design, **kwargs)
    monkeypatch.setattr(score_resampling, "detect_bordered_design", lambda _: None)
    monkeypatch.setattr(bordered, "detect_bordered_design", lambda _: None)
    dense = score_resampling.run_low_moi_score_permutations(design, **kwargs)
    np.testing.assert_allclose(structured.observed_score, dense.observed_score, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(structured.resampled_scores, dense.resampled_scores, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(structured.p_value, dense.p_value)
    np.testing.assert_array_equal(structured.null_converged, dense.null_converged)


def test_segmented_pool_information_matches_dense():
    with _x64():
        model = _mixed_design()
        z = np.asarray(model.nuisance_design, dtype=np.float64)
        design = bordered.detect_bordered_design(z)
        rng = np.random.default_rng(44)
        weight = rng.uniform(0.2, 2.0, size=(z.shape[0], 3))
        values = rng.normal(size=weight.shape)
        targets = rng.integers(0, 4, size=z.shape[0], dtype=np.int32)
        info = segmented_information(design, jnp.asarray(weight), jnp.asarray(targets), num_targets=4)
        rhs = segmented_transpose_dot(design, jnp.asarray(values), jnp.asarray(targets), num_targets=4)
        actual = bordered.solve(design, info, rhs)
        expected = []
        for t in range(4):
            rows = targets == t
            matrix = np.einsum("nq,ng,nr->gqr", z[rows], weight[rows], z[rows])
            target_rhs = (z[rows].T @ values[rows]).T
            expected.append(np.linalg.solve(matrix, target_rhs[..., None])[..., 0])
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("screen", [0.05, 1.0])
def test_low_moi_spa_pool_projection_matches_dense(monkeypatch, screen):
    # Force multiple cell chunks, including a partial final chunk, to exercise
    # the bounded workspace independently of production-sized data.
    monkeypatch.setattr(bordered_scores, "_CUMULANT_WORKING_ELEMENTS", 100)
    bordered_scores._correct_bordered_cumulants.clear_cache()
    with _x64():
        model = _mixed_design()
        z = np.asarray(model.nuisance_design, dtype=np.float64)
        design = bordered.detect_bordered_design(z)
        controls = np.asarray(model.control_mask)
        targets = np.full(len(z), -1, dtype=np.int32)
        targets[300:390], targets[390:] = 0, 1
        rng = np.random.default_rng(301)
        weight = rng.uniform(0.2, 1.5, size=(len(z), 3))
        residual = rng.normal(size=weight.shape)
        residual[300:390, 0] += 0.4
        info = np.einsum("nq,ng,nr->gqr", z[controls], weight[controls], z[controls]) + np.eye(z.shape[1]) * 0.25
        direction = np.linalg.solve(info, (z[controls].T @ residual[controls]).T[..., None])[..., 0]
        contribution = residual - weight * (z @ direction.T)
        kwargs = dict(contribution=contribution, target_codes=targets, control_mask=controls,
                      shared_logits=np.zeros(len(z)), intercepts=np.full(2, np.log(90 / 300)),
                      num_targets=2, screen_p_value=screen, weight=weight, gene_block_size=2)
        structured = saddlepoint.fit_low_moi_propensity_saddlepoint(
            **kwargs, bordered_design=design,
            control_information=bordered.weighted_information(design.take(np.flatnonzero(controls)), weight[controls], ridge=0.25),
        )
        monkeypatch.setattr(saddlepoint, "detect_bordered_design", lambda _: None)
        dense = saddlepoint.fit_low_moi_propensity_saddlepoint(**kwargs, nuisance_design=z, control_information=info)
        for field in ("observed_sum", "null_mean", "null_variance", "null_skewness", "p_value", "log_p_value"):
            tolerance = 2e-8 if field in ("p_value", "log_p_value") else 2e-9
            np.testing.assert_allclose(getattr(structured, field), getattr(dense, field), atol=2e-10, rtol=tolerance)
        np.testing.assert_array_equal(structured.valid, dense.valid)
        np.testing.assert_array_equal(structured.used_fallback, dense.used_fallback)


def test_mixed_resampling_does_not_use_dense_target_kernel(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Mixed batch design reached dense target projection")
    monkeypatch.setattr(score_resampling, "prepare_control_only_target_scores", forbidden)
    monkeypatch.setattr(score_resampling, "batched_efficient_score_from_indices", forbidden)
    result = score_resampling.run_low_moi_score_permutations(_mixed_design(), num_resamples=9)
    assert np.isfinite(result.observed_score).all()


def test_complete_mixed_propensity_spa_matches_dense(monkeypatch):
    with _x64():
        design = _mixed_design()
        # Reuse the assignment model so the comparison isolates the complete
        # response-side fitting, score, pool projection and SPA pipeline.
        permutations = score_resampling.precompute_low_moi_permutations(
            design, num_resamples=9, resampling_mechanism="propensity", draw_resamples=False,
        )
        kwargs = dict(num_resamples=9, permutations=permutations, tail_approximation="saddlepoint",
                      saddlepoint_only=True, saddlepoint_screen_p_value=1.0, maxiter=100)
        structured = score_resampling.run_low_moi_score_permutations(design, **kwargs)
        monkeypatch.setattr(score_resampling, "detect_bordered_design", lambda _: None)
        monkeypatch.setattr(bordered, "detect_bordered_design", lambda _: None)
        monkeypatch.setattr(saddlepoint, "detect_bordered_design", lambda _: None)
        dense = score_resampling.run_low_moi_score_permutations(design, **kwargs)
        np.testing.assert_allclose(structured.observed_score, dense.observed_score, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(structured.parametric_log_p_value, dense.parametric_log_p_value, atol=3e-5, rtol=3e-5)
        np.testing.assert_array_equal(structured.parametric_fit_valid, dense.parametric_fit_valid)


def test_shared_selection_fit_preserves_original_coefficients(monkeypatch):
    from perturbo._internal.high_moi import resampling

    model = _mixed_design()
    z = np.asarray(model.nuisance_design)
    targeting = ~np.asarray(model.control_mask)
    with monkeypatch.context() as scoped:
        def forbidden(*args, **kwargs):
            raise AssertionError("Structured shared selection fit rebuilt a dense Gram projection")
        scoped.setattr(np.linalg, "lstsq", forbidden)
        structured = fit_shared_propensity_coefficients(z, targeting)
    monkeypatch.setattr(resampling, "prepare_bordered_propensity", lambda *args, **kwargs: None)
    dense = fit_shared_propensity_coefficients(z, targeting)
    assert structured.shape == (z.shape[1],)
    np.testing.assert_allclose(z @ structured, z @ dense, atol=2e-5, rtol=2e-5)
