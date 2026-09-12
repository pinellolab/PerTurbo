"""Rank-deficient propensity designs must not create synthetic predictors."""

from __future__ import annotations

import numpy as np

from perturbo._internal.high_moi.resampling import (
    fit_propensity_probabilities,
    propensity_basis,
)


def test_zero_and_duplicate_columns_leave_an_intercept_only_fit_unchanged() -> None:
    outcome = np.zeros((1, 100), dtype=np.float32)
    outcome[0, :20] = 1.0
    intercept = np.ones((100, 1), dtype=np.float32)

    expected = np.asarray(fit_propensity_probabilities(outcome, intercept))
    variants = (
        np.column_stack([intercept, np.zeros(100)]),
        np.column_stack([intercept, intercept]),
        np.column_stack([intercept, 1_000.0 * intercept]),
    )
    for design in variants:
        actual = np.asarray(fit_propensity_probabilities(outcome, design))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(actual, 0.2, rtol=1e-6, atol=1e-7)


def test_propensity_fit_is_invariant_to_units_and_row_permutation() -> None:
    rng = np.random.default_rng(9)
    covariate = np.linspace(-2.0, 2.0, 240, dtype=np.float32)
    probability = 1.0 / (1.0 + np.exp(-(-0.8 + 0.7 * covariate)))
    outcome = (rng.random(covariate.size) < probability).astype(np.float32)[None, :]
    design = np.column_stack([np.ones(covariate.size), covariate]).astype(np.float32)
    expected = np.asarray(fit_propensity_probabilities(outcome, design))

    redundant = np.column_stack(
        [np.ones(covariate.size), 1_000.0 * covariate, covariate, np.zeros(covariate.size)]
    ).astype(np.float32)
    rescaled = np.asarray(fit_propensity_probabilities(outcome, redundant))
    np.testing.assert_allclose(rescaled, expected, rtol=2e-5, atol=2e-6)

    order = rng.permutation(covariate.size)
    permuted = np.asarray(fit_propensity_probabilities(outcome[:, order], redundant[order]))
    np.testing.assert_allclose(permuted[:, np.argsort(order)], expected, rtol=2e-5, atol=2e-6)


def test_large_screens_keep_real_correlated_covariate_directions() -> None:
    """Float32 source precision must not turn the row count into a rank cutoff."""

    rng = np.random.default_rng(31)
    rows = 200_000
    first = rng.normal(size=rows).astype(np.float32)
    second = (first + 0.01 * rng.normal(size=rows)).astype(np.float32)
    design = np.column_stack([np.ones(rows, dtype=np.float32), first, second])

    basis = np.asarray(propensity_basis(design))
    assert basis.shape == (rows, 3)
