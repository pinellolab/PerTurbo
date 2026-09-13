"""Structured propensity solves preserve the existing orthonormal IRLS law."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from perturbo._internal.bordered import detect_bordered_design
from perturbo._internal.high_moi import resampling


def _fixture():
    rng = np.random.default_rng(102)
    batch = np.tile(np.arange(5), 80)
    first, second = rng.normal(size=(2, batch.size)).astype(np.float32)
    # Original reference coding, with two continuous columns interleaved.
    design = np.column_stack(
        [batch == 1, first, np.ones(batch.size), batch == 2, second, batch == 3, batch == 4]
    ).astype(np.float32)
    beta = rng.normal(scale=0.3, size=(4, design.shape[1]))
    beta[:, 2] -= 0.7
    probability = 1 / (1 + np.exp(-(beta @ design.T)))
    indicators = (rng.random(probability.shape) < probability).astype(np.float32)
    bordered = detect_bordered_design(design)
    assert bordered is not None
    basis = resampling.propensity_basis(design)
    return design, indicators, bordered, basis, batch


@pytest.mark.parametrize("iterations,jitter", [(1, 0.2), (8, 1e-6), (25, 1e-6)])
def test_logits_and_probabilities_match_dense(iterations, jitter, monkeypatch):
    design, indicators, bordered, basis, _ = _fixture()
    expected, expected_basis = resampling.fit_propensity_coefficients(
        indicators, design, basis=basis, max_iterations=iterations, jitter=jitter
    )

    def forbid_dense(*args, **kwargs):
        raise AssertionError("The supported mixed design must use its structured solve.")

    monkeypatch.setattr(resampling, "_logistic_irls_orthonormal", forbid_dense)
    actual, actual_basis = resampling.fit_propensity_coefficients(
        indicators, design, basis=basis, bordered_design=bordered, max_iterations=iterations, jitter=jitter
    )
    np.testing.assert_array_equal(actual_basis, expected_basis)
    actual_logits = resampling.propensity_logits_from_coefficients(actual, actual_basis)
    expected_logits = resampling.propensity_logits_from_coefficients(expected, expected_basis)
    np.testing.assert_allclose(actual_logits, expected_logits, atol=8e-6, rtol=2e-5)
    np.testing.assert_allclose(jax.nn.sigmoid(actual_logits), jax.nn.sigmoid(expected_logits), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("jitter", [1e-6, 0.03])
def test_full_rank_masked_pools_retain_global_damping(jitter, monkeypatch):
    _, indicators, bordered, basis, batch = _fixture()
    mask = np.ones_like(indicators)
    mask[0, ::3] = 0
    mask[1, ::4] = 0
    mask[2, ::7] = 0
    expected = resampling.fit_masked_propensity_coefficients(indicators, mask, basis, jitter=jitter)
    context = resampling.prepare_bordered_propensity(bordered, basis)
    assert context is not None

    def forbid_repeated_preparation(*args, **kwargs):
        raise AssertionError("Prepared contexts must be reused across target batches.")

    def forbid_dense(*args, **kwargs):
        raise AssertionError("Full-rank masked designs must use the structured solve.")

    monkeypatch.setattr(resampling, "prepare_bordered_propensity", forbid_repeated_preparation)
    monkeypatch.setattr(resampling, "_logistic_irls_masked_orthonormal", forbid_dense)
    actual = resampling.fit_masked_propensity_coefficients(
        indicators, mask, basis, bordered_design=context, jitter=jitter
    )
    np.testing.assert_allclose(actual @ basis.T, expected @ basis.T, atol=3e-5, rtol=3e-5)


def test_masked_missing_reference_preserves_full_basis_damping():
    _, indicators, bordered, basis, batch = _fixture()
    mask = np.ones_like(indicators)
    mask[:, batch == 0] = 0
    # A large damping term makes a mistaken original-coordinate diagonal
    # penalty or masked Gram matrix detectable well above float32 rounding.
    expected = resampling.fit_masked_propensity_coefficients(indicators, mask, basis, jitter=0.03, max_iterations=2)
    context = resampling.prepare_bordered_propensity(bordered, basis)
    actual = resampling._logistic_irls_bordered(
        indicators, mask, context, np.float32(0.03), np.float32(30), max_iterations=2
    )
    np.testing.assert_allclose(actual @ basis.T, expected @ basis.T, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("missing_batch", [0, 1])
def test_masked_pools_missing_batches_keep_dense_coefficients(missing_batch, monkeypatch):
    _, indicators, bordered, basis, batch = _fixture()
    mask = np.ones_like(indicators)
    mask[0, batch == missing_batch] = 0
    expected = resampling.fit_masked_propensity_coefficients(indicators, mask, basis)

    def forbid_structured(*args, **kwargs):
        raise AssertionError("Locally unidentified coefficients must retain the dense solver.")

    monkeypatch.setattr(resampling, "_logistic_irls_bordered", forbid_structured)
    actual = resampling.fit_masked_propensity_coefficients(indicators, mask, basis, bordered_design=bordered)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("variant", ["missing_reference", "dependent_continuous", "rescaled_continuous"])
def test_unsupported_or_ill_conditioned_designs_keep_dense_fallback(variant, monkeypatch):
    design, indicators, _, _, batch = _fixture()
    if variant == "missing_reference":
        rows = batch != 0
        design, indicators = design[rows], indicators[:, rows]
    elif variant == "dependent_continuous":
        design[:, 4] = 2 * design[:, 1]
    else:
        design[:, 1] *= 10_000
    bordered = detect_bordered_design(design)
    assert bordered is not None
    basis = resampling.propensity_basis(design)
    assert resampling.prepare_bordered_propensity(bordered, basis) is None
    expected, _ = resampling.fit_propensity_coefficients(indicators, design, basis=basis)

    def forbid_structured(*args, **kwargs):
        raise AssertionError("The unsupported design must retain the dense solver.")

    monkeypatch.setattr(resampling, "_logistic_irls_bordered", forbid_structured)
    actual, _ = resampling.fit_propensity_coefficients(indicators, design, basis=basis, bordered_design=bordered)
    np.testing.assert_array_equal(actual, expected)
    mask = np.ones_like(indicators)
    expected_masked = resampling.fit_masked_propensity_coefficients(indicators, mask, basis)
    actual_masked = resampling.fit_masked_propensity_coefficients(indicators, mask, basis, bordered_design=bordered)
    np.testing.assert_array_equal(actual_masked, expected_masked)


def test_nonfinite_structured_result_retries_existing_dense_solver(monkeypatch):
    design, indicators, bordered, basis, _ = _fixture()
    expected, _ = resampling.fit_propensity_coefficients(indicators, design, basis=basis)
    monkeypatch.setattr(
        resampling, "_logistic_irls_bordered", lambda *args, **kwargs: np.full(expected.shape, np.nan, dtype=np.float32)
    )
    actual, _ = resampling.fit_propensity_coefficients(indicators, design, basis=basis, bordered_design=bordered)
    np.testing.assert_array_equal(actual, expected)


def test_rare_assignments_preserve_fitted_counts_and_high_moi_spa_tails(monkeypatch, record_property):
    """Check the assignment law and tail calculation at 0.2% prevalence.

    The residual matrix is fixed between routes. Its last column deliberately
    stresses both tails; this is a numerical regression, not calibration or
    biological validation of either estimator.
    """
    from perturbo._internal.saddlepoint import fit_high_moi_propensity_saddlepoint

    rng = np.random.default_rng(2064)
    num_cells, num_batches, num_elements, num_genes = 20_000, 32, 3, 3
    batch = np.arange(num_cells) % num_batches
    continuous = rng.normal(size=(num_cells, 2)).astype(np.float32)
    design = np.column_stack(
        [np.ones(num_cells), continuous[:, 0], np.eye(num_batches)[batch, 1:], continuous[:, 1]]
    ).astype(np.float32)
    beta = rng.normal(scale=0.2, size=(num_elements, design.shape[1]))
    beta[:, 0] = np.log(0.002 / 0.998)
    probability = 1 / (1 + np.exp(-(beta @ design.T)))
    indicators = (rng.random(probability.shape) < probability).astype(np.float32)
    indicators[0, batch == 0] = 0  # An event-free reference level is especially demanding.
    indicators[1, batch == 5] = 0
    selected_counts = indicators.sum(axis=1)
    assert np.all((selected_counts > 15) & (selected_counts < 80))
    bordered = detect_bordered_design(design)
    assert bordered is not None

    original_x64 = jax.config.read("jax_enable_x64")
    try:
        jax.config.update("jax_enable_x64", False)
        basis = resampling.propensity_basis(design)
        context = resampling.prepare_bordered_propensity(bordered, basis)
        assert context is not None
        expected, _ = resampling.fit_propensity_coefficients(indicators, design, basis=basis)

        def forbid_dense(*args, **kwargs):
            raise AssertionError("This regression must exercise the structured propensity solver.")

        with monkeypatch.context() as patch:
            patch.setattr(resampling, "_logistic_irls_orthonormal", forbid_dense)
            actual, _ = resampling.fit_propensity_coefficients(
                indicators, design, basis=basis, bordered_design=context
            )
        expected_probability = np.asarray(jax.nn.sigmoid(expected @ basis.T), dtype=np.float64)
        actual_probability = np.asarray(jax.nn.sigmoid(actual @ basis.T), dtype=np.float64)
        probability_error = np.max(np.abs(actual_probability - expected_probability))
        fitted_count_error = np.max(np.abs(actual_probability.sum(1) - expected_probability.sum(1)))
        record_property("max_probability_error", float(probability_error))
        record_property("max_fitted_count_difference", float(fitted_count_error))
        record_property("max_count_score_error", float(np.max(np.abs(actual_probability.sum(1) - selected_counts))))
        # At this prevalence, 1e-6 is 0.05% of the baseline probability.
        assert probability_error < 1e-6
        np.testing.assert_allclose(actual_probability.sum(1), expected_probability.sum(1), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(actual_probability.sum(1), selected_counts, rtol=1e-4, atol=1e-4)

        # Hold the null residual, weight, and nuisance correction fixed so only
        # propensity fitting can account for a tail difference.
        residual = rng.normal(size=(num_cells, num_genes))
        residual[:, 2] += 1.5 * indicators[0] - 1.5 * indicators[1]
        weight = rng.uniform(0.5, 1.5, size=residual.shape)
        information = np.einsum("nq,ng,nr->gqr", design.astype(float), weight, design.astype(float))
        direction = np.linalg.solve(information, (design.T @ residual).T[..., None])[..., 0].T
        cell_index, element_index = np.nonzero(indicators.T)
        common = dict(
            score_residual=residual, observation_weight=weight,
            nuisance_design=design, nuisance_direction=direction,
            cell_index=cell_index, element_index=element_index, num_elements=num_elements,
            propensity_basis=basis, screen_p_value=1.0, gene_block_size=3,
        )
        jax.config.update("jax_enable_x64", True)
        expected_tail = fit_high_moi_propensity_saddlepoint(propensity_coefficients=expected, **common)
        actual_tail = fit_high_moi_propensity_saddlepoint(propensity_coefficients=actual, **common)
        record_property("minimum_log_p_value", float(np.min(expected_tail.log_p_value)))
        record_property("max_log_p_error", float(np.max(np.abs(actual_tail.log_p_value - expected_tail.log_p_value))))
        assert expected_tail.valid.all() and actual_tail.valid.all()
        assert not expected_tail.used_fallback.any() and not actual_tail.used_fallback.any()
        assert np.min(expected_tail.log_p_value) < -10
        np.testing.assert_allclose(actual_tail.observed_sum, expected_tail.observed_sum, rtol=0, atol=0)
        # |delta log(p)| < 1e-3 bounds the relative p-value change to about 0.1%.
        np.testing.assert_allclose(actual_tail.log_p_value, expected_tail.log_p_value, rtol=0, atol=1e-3)
    finally:
        jax.config.update("jax_enable_x64", original_x64)
