"""Structured NB fitting preserves original coefficients, priors and diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from perturbo._internal import bordered
from perturbo._internal import jax_kernels as kernels
from perturbo._internal.score_resampling import newton_step_magnitude
from perturbo.crt import ControlNuisance, polish_baseline_to_null_mode


def _problem(*, missing_group: bool = False, missing_reference: bool = False, genes: int = 4):
    rng = np.random.default_rng(812)
    codes = np.tile(np.arange(4), 48)
    if missing_group:
        codes = codes[codes != 2]
    if missing_reference:
        codes = codes[codes != 3]
    # Interleave a continuous predictor with the reference-coded batch dummies.
    design = np.column_stack(
        [np.ones(codes.size), codes == 0, rng.normal(size=codes.size), codes == 1, codes == 2]
    ).astype(np.float64)
    beta = rng.normal(scale=0.15, size=(design.shape[1], genes))
    beta[0] += 0.8
    offsets = rng.normal(scale=0.1, size=(codes.size, 1))
    theta = rng.uniform(2.0, 8.0, size=genes)
    mean = np.exp(offsets + design @ beta)
    counts = rng.negative_binomial(theta, theta / (theta + mean)).astype(np.float32)
    return counts, design, offsets, theta, beta


def _arguments(problem, *, prior: float = 0.35):
    counts, design, offsets, theta, _ = problem
    return tuple(jnp.asarray(value, dtype=jnp.float32) for value in (
        counts, design, offsets, theta, prior, 0.002, 1e-6
    ))


@pytest.mark.parametrize("missing_group", [False, True])
@pytest.mark.parametrize("prior", [0.0, 0.35])
def test_nb_fit_and_residuals_match_dense_with_prior(missing_group, prior):
    args = _arguments(_problem(missing_group=missing_group), prior=prior)
    assert bordered.detect_bordered_design(args[1]) is not None
    expected = kernels._fisher_nb_null_kernel(*args, max_iterations=80)
    actual = kernels.fisher_nb_null(*args, max_iterations=80)
    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(value, reference, rtol=2e-5, atol=3e-6)
    residual, weight = kernels.nb_null_residual_and_weight(*args[:4], actual[0])
    np.testing.assert_allclose(residual, expected[1], rtol=2e-5, atol=3e-6)
    np.testing.assert_allclose(weight, expected[2], rtol=2e-5, atol=3e-6)


def test_laplace_covariance_and_final_step_match_dense():
    args = _arguments(_problem(missing_group=True))
    expected = kernels._fit_nb_null_laplace_kernel(*args, max_iterations=80)
    actual = kernels.fit_nb_null_laplace(*args, max_iterations=80)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5, atol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-5, atol=3e-6)
    np.testing.assert_allclose(actual[2], expected[2], rtol=2e-4, atol=2e-6)
    assert 0 < int(actual[3]) <= 80


@pytest.mark.parametrize("entry,kernel", [
    ("fisher_nb_null", "_fisher_nb_null_kernel"),
    ("fit_nb_null_laplace", "_fit_nb_null_laplace_kernel"),
    ("nb_null_residual_and_weight", "_nb_null_residual_and_weight_kernel"),
])
def test_host_entry_points_dispatch_before_compilation(monkeypatch, entry, kernel):
    args = _arguments(_problem())
    received = []
    sentinel = object()

    def capture(*values, **kwargs):
        received.append(values[1])
        return sentinel

    monkeypatch.setattr(kernels, kernel, capture)
    call_args = args if entry != "nb_null_residual_and_weight" else (*args[:4], jnp.zeros((5, 4)))
    assert getattr(kernels, entry)(*call_args) is sentinel
    assert isinstance(received[0], bordered.BorderedDesign)


def test_traced_design_retains_dense_fallback():
    args = _arguments(_problem())
    traced_fit = jax.jit(lambda design: kernels.fisher_nb_null(
        args[0], design, *args[2:], max_iterations=80
    ))
    actual = traced_fit(args[1])
    expected = kernels._fisher_nb_null_kernel(*args, max_iterations=80)
    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(value, reference, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("supplied_structure", [False, True])
@pytest.mark.parametrize("prior", [0.0, 0.35])
def test_missing_reference_keeps_dense_fit_with_original_regularization(supplied_structure, prior):
    args = list(_arguments(_problem(missing_reference=True), prior=prior))
    args[5] = jnp.asarray(1e-8, dtype=jnp.float32)
    expected = kernels._fisher_nb_null_kernel(*args, max_iterations=40)
    if supplied_structure:
        args[1] = bordered.detect_bordered_design(args[1])
    actual = kernels.fisher_nb_null(*args, max_iterations=40)
    for value, reference in zip(actual, expected):
        np.testing.assert_array_equal(value, reference)


def test_reference_dependency_requires_a_nonzero_constant_border():
    _, design, _, _, _ = _problem(missing_reference=True)
    structured = bordered.detect_bordered_design(design)
    assert bordered.has_reference_dependency(structured)
    np.testing.assert_array_equal(bordered.to_dense_numpy(structured), design)
    without_intercept = bordered.detect_bordered_design(design[:, 1:])
    assert not bordered.has_reference_dependency(without_intercept)


def test_polishing_reuses_detection_and_preserves_genes_without_counts(monkeypatch):
    counts, design, offsets, theta, beta = _problem()
    counts[:, -1] = 0
    nuisance = ControlNuisance(
        counts=counts, nuisance_design=design, coefficients=beta, offsets=offsets,
        dispersion=theta, nuisance_names=tuple(f"z{i}" for i in range(design.shape[1])),
        gene_names=tuple(f"g{i}" for i in range(counts.shape[1])),
    )
    detections = []
    original = bordered.detect_bordered_design

    def detect(value):
        detections.append(value.shape)
        return original(value)

    monkeypatch.setattr(bordered, "detect_bordered_design", detect)
    actual = polish_baseline_to_null_mode(nuisance, gene_block=2, curvature_jitter=0.002)
    args = _arguments((counts[:, :-1], design, offsets, theta[:-1], beta[:, :-1]), prior=0.0)
    expected = kernels._fisher_nb_null_kernel(*args, max_iterations=50)[0]
    np.testing.assert_allclose(actual.coefficients[:, :-1], expected, rtol=2e-5, atol=3e-6)
    np.testing.assert_array_equal(actual.coefficients[:, -1], beta[:, -1])
    assert detections == [design.shape]


@pytest.mark.parametrize("dense_ridge", [False, True])
@pytest.mark.parametrize("missing_reference", [False, True])
@pytest.mark.parametrize("supplied_structure", [False, True])
def test_float64_diagnostic_matches_dense_across_gene_blocks(
    monkeypatch, dense_ridge, missing_reference, supplied_structure
):
    counts, design, offsets, theta, beta = _problem(
        missing_group=True, missing_reference=missing_reference, genes=270
    )
    prior = 0.35
    ridge = np.diag(np.linspace(prior + 0.002, prior + 0.2, design.shape[1]))
    if dense_ridge:
        ridge[0, 2] = ridge[2, 0] = 0.02
    calls = []
    original = bordered.solve_numpy

    def capture(*args):
        calls.append(args[2].shape)
        return original(*args)

    monkeypatch.setattr(bordered, "solve_numpy", capture)
    input_design = bordered.detect_bordered_design(design) if supplied_structure else design
    actual = newton_step_magnitude(
        counts, input_design, offsets, theta, beta, prior_precision=prior, ridge=ridge
    )
    mean = np.exp(offsets + design @ beta)
    denominator = theta[None, :] + mean
    residual = theta[None, :] * (counts - mean) / denominator
    weight = theta[None, :] * mean / denominator
    gradient = design.T @ residual - prior * beta
    information = np.einsum("nq,ng,nr->gqr", design, weight, design) + ridge
    step = np.linalg.solve(information, gradient.T[..., None])[..., 0]
    expected = np.max(np.abs(step), axis=1)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    assert actual.dtype == np.float64
    assert len(calls) == (0 if dense_ridge or missing_reference else 2)
