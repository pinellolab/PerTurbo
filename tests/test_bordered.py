"""Structured nuisance algebra must solve the original penalized equation."""

from __future__ import annotations

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from perturbo._internal import bordered as bordered_module
from perturbo._internal.bordered import (
    BorderedInfo,
    add_information,
    detect_bordered_design,
    matmul,
    matmul_numpy,
    solve,
    solve_numpy,
    transpose_dot,
    transpose_dot_numpy,
    weighted_information,
    weighted_information_numpy,
)


@contextmanager
def _x64(enabled=True):
    original = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", original)


def _problem(seed=2, n=180, genes=5, groups=4):
    rng = np.random.default_rng(seed)
    codes = np.arange(n) % (groups + 1)
    rng.shuffle(codes)
    indicators = (codes[:, None] == np.arange(groups)).astype(float)
    dense = np.column_stack([np.ones(n), rng.normal(size=(n, 2)), indicators])
    # Interleaving prevents a hidden assumption about nuisance column order.
    permutation = rng.permutation(dense.shape[1])
    dense = dense[:, permutation]
    weights = rng.uniform(0.1, 2.0, (n, genes))
    rhs = rng.normal(size=(genes, dense.shape[1]))
    return dense, weights, rhs


def _dense_information(dense, weights, ridge):
    q = dense.shape[1]
    ridge = np.asarray(ridge)
    penalty = ridge * np.eye(q) if ridge.ndim == 0 else ridge[..., :, None] * np.eye(q)
    return np.einsum("nd,...ng,ne->...gde", dense, weights, dense) + penalty


def _dense_solution(information, rhs):
    return np.linalg.solve(information, rhs[..., None])[..., 0]


def test_detector_preserves_reference_rows_intercept_and_original_columns():
    dense, _, _ = _problem()
    design = detect_bordered_design(dense)
    assert design is not None
    assert design.num_columns == dense.shape[1]
    assert design.num_border == 3
    assert design.num_groups == 4
    intercept = np.flatnonzero(np.all(dense == 1, axis=0))[0]
    assert intercept in design.border_indices
    reference_rows = dense[:, design.group_indices].sum(1) == 0
    np.testing.assert_array_equal(design.codes == design.num_groups, reference_rows)
    # Reconstruct by multiplication, with no full-coded reference coefficient.
    np.testing.assert_array_equal(matmul_numpy(design, np.eye(dense.shape[1])), dense)


@pytest.mark.parametrize("kind", ["overlap", "single_indicator", "continuous", "nonfinite"])
def test_detector_conservatively_declines_unsupported_designs(kind):
    dense, _, _ = _problem()
    if kind == "overlap":
        dense = np.column_stack([np.ones(4), [0, 1, 0, 1], [0, 0, 1, 1]])
    elif kind == "single_indicator":
        dense = np.column_stack([np.ones(4), [0, 1, 0, 1]])
    elif kind == "continuous":
        dense = dense[:, np.any((dense != 0) & (dense != 1), axis=0)]
    else:
        dense[0, 0] = np.nan
    assert detect_bordered_design(dense) is None


@pytest.mark.parametrize("ridge_kind", ["none", "scalar", "diagonal", "per_gene"])
def test_numpy_solve_matches_original_dense_information_and_penalty(ridge_kind):
    dense, weights, rhs = _problem()
    design = detect_bordered_design(dense)
    ridge = {
        "none": 0.0,
        "scalar": 0.7,
        "diagonal": np.linspace(0.02, 2, dense.shape[1]),
        "per_gene": np.linspace(0.02, 2, rhs.size).reshape(rhs.shape),
    }[ridge_kind]
    info = weighted_information_numpy(design, weights, ridge)
    actual = solve_numpy(design, info, rhs)
    expected = _dense_solution(_dense_information(dense, weights, ridge), rhs)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    assert actual.dtype == np.float64
    assert info.border.shape == (weights.shape[1], 3, 3)
    assert info.cross.shape == (weights.shape[1], 3, 4)
    assert info.diagonal.shape == (weights.shape[1], 4)


def test_jitted_solve_and_products_match_dense_with_two_continuous_covariates():
    dense, weights, rhs = _problem()
    ridge = np.linspace(0.1, 0.9, dense.shape[1])
    with _x64():
        design = detect_bordered_design(dense)

        @jax.jit
        def run(design, weights, rhs):
            info = weighted_information(design, weights, ridge)
            solution = solve(design, info, rhs)
            return solution, matmul(design, solution.T), transpose_dot(design, weights)

        solution, fitted, cross = run(design, weights, rhs)
        expected = _dense_solution(_dense_information(dense, weights, ridge), rhs)
        np.testing.assert_allclose(solution, expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(fitted, dense @ expected.T, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(cross, (dense.T @ weights).T, rtol=2e-12, atol=2e-12)


def test_numpy_float64_does_not_depend_on_jax_x64_setting():
    dense, weights, rhs = _problem()
    # A perturbation below float32 precision must survive host detection.
    continuous_column = np.flatnonzero(np.any((dense != 0) & (dense != 1), axis=0))[0]
    dense[:, continuous_column] += np.linspace(0, 1e-9, dense.shape[0])
    with _x64(False):
        design = detect_bordered_design(dense)
        assert design.border.dtype == np.float64
        result = solve_numpy(design, weighted_information_numpy(design, weights, 0.3), rhs)
    expected = _dense_solution(_dense_information(dense, weights, 0.3), rhs)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)


def test_jax_float32_solve_preserves_dtype_and_dense_solution():
    dense, weights, rhs = (value.astype(np.float32) for value in _problem())
    with _x64(False):
        design = detect_bordered_design(dense)
        info = weighted_information(design, weights, 0.4)
        result = jax.jit(solve)(design, info, rhs)
    expected = _dense_solution(_dense_information(dense.astype(float), weights.astype(float), 0.4), rhs)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("empty_kind", ["reference", "indicator"])
def test_missing_reference_or_empty_batch_preserves_nonzero_original_ridge(empty_kind):
    dense, weights, rhs = _problem()
    design = detect_bordered_design(dense)
    missing_code = design.num_groups if empty_kind == "reference" else 0
    keep = np.flatnonzero(design.codes != missing_code)
    subset = design.take(keep)
    # Missing reference makes the intercept linearly dependent on the retained
    # dummies; the original diagonal penalty, not recoding, resolves the system.
    ridge = np.linspace(0.2, 1.1, dense.shape[1])
    info = weighted_information_numpy(subset, weights[keep], ridge)
    actual = solve_numpy(subset, info, rhs)
    expected = _dense_solution(_dense_information(dense[keep], weights[keep], ridge), rhs)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)


def test_all_zero_original_column_stays_diagonal_with_its_own_penalty():
    dense, weights, rhs = _problem()
    dense = np.column_stack([dense, np.zeros(dense.shape[0])])
    rhs = np.column_stack([rhs, np.ones(rhs.shape[0])])
    design = detect_bordered_design(dense)
    assert dense.shape[1] - 1 in design.group_indices
    assert design.num_border == 3
    ridge = np.linspace(0.1, 1.0, dense.shape[1])
    actual = solve_numpy(design, weighted_information_numpy(design, weights, ridge), rhs)
    expected = _dense_solution(_dense_information(dense, weights, ridge), rhs)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_many_absent_batches_do_not_enlarge_the_dense_border():
    dense, weights, rhs = _problem(n=410, groups=40, genes=3)
    original = detect_bordered_design(dense)
    keep = (original.codes < 3) | (original.codes == original.num_groups)
    dense, weights = dense[keep], weights[keep]
    design = detect_bordered_design(dense)
    assert design.num_border == original.num_border == 3
    assert design.num_groups == original.num_groups == 40
    assert design.num_columns == dense.shape[1]
    np.testing.assert_array_equal(design.group_indices, original.group_indices)
    ridge = np.linspace(0.1, 1.0, dense.shape[1])
    expected = _dense_solution(_dense_information(dense, weights, ridge), rhs)
    actual = solve_numpy(design, weighted_information_numpy(design, weights, ridge), rhs)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    with _x64():
        actual_jax = jax.jit(solve)(design, weighted_information(design, weights, ridge), rhs)
    np.testing.assert_allclose(actual_jax, expected, rtol=2e-12, atol=2e-12)


def test_information_addition_preserves_one_shared_penalty():
    dense, weights, rhs = _problem()
    design = detect_bordered_design(dense)
    left = np.arange(0, dense.shape[0], 2)
    right = np.arange(1, dense.shape[0], 2)
    info = add_information(
        weighted_information_numpy(design.take(left), weights[left], 0.25),
        weighted_information_numpy(design.take(right), weights[right]),
    )
    actual = solve_numpy(design, info, rhs)
    expected = _dense_solution(_dense_information(dense, weights, 0.25), rhs)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_target_resample_broadcasting_and_batched_weights():
    dense, weights, rhs = _problem(n=120, genes=3)
    rng = np.random.default_rng(8)
    target_weights = weights[None] * np.array([0.7, 1.3])[:, None, None]
    right_hand_sides = rng.normal(size=(2, 4) + rhs.shape)
    design = detect_bordered_design(dense)
    info = weighted_information_numpy(design, target_weights, 0.4)
    broadcast_info = BorderedInfo(*(block[:, None] for block in info))
    expected = _dense_solution(_dense_information(dense, target_weights, 0.4)[:, None], right_hand_sides)
    actual = solve_numpy(design, broadcast_info, right_hand_sides)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    with _x64():
        jax_info = weighted_information(design, target_weights, 0.4)
        jax_broadcast_info = BorderedInfo(*(block[:, None] for block in jax_info))
        jax_result = jax.jit(solve)(design, jax_broadcast_info, right_hand_sides)
        np.testing.assert_allclose(jax_result, expected, rtol=2e-12, atol=2e-12)


def test_traced_row_subsetting_and_padding_preserve_original_geometry():
    dense, weights, _ = _problem(n=90)
    rows = np.array([[0, 2, 4, 6], [1, 3, 5, 7]], dtype=np.int32)
    with _x64():
        design = detect_bordered_design(dense)

        @jax.jit
        def run(design, weights, selections):
            def one(indices):
                part = design.take(indices).pad_rows(1)
                values = jnp.concatenate([weights[indices], jnp.zeros_like(weights[:1])])
                return transpose_dot(part, values)
            return jax.vmap(one)(selections)

        actual = run(design, weights, rows)
        expected = np.stack([(dense[idx].T @ weights[idx]).T for idx in rows])
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    padded = design.pad_rows(2)
    assert padded.border.dtype == np.float64
    np.testing.assert_array_equal(padded.codes[-2:], design.num_groups)
    np.testing.assert_array_equal(padded.border[-2:], 0)
    converted = design.astype(np.float32)
    assert converted.border.dtype == np.float32
    assert converted.codes.dtype == np.int32


def test_zero_border_design_and_batched_products():
    dense, weights, rhs = _problem()
    reference = detect_bordered_design(dense)
    dense = dense[:, reference.group_indices]
    rhs = rhs[:, reference.group_indices]
    design = detect_bordered_design(dense)
    assert design.num_border == 0
    info = weighted_information_numpy(design, weights, 0.4)
    result = solve_numpy(design, info, rhs)
    expected = _dense_solution(_dense_information(dense, weights, 0.4), rhs)
    np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)
    coefficients = np.stack([rhs.T, 2 * rhs.T])
    np.testing.assert_allclose(matmul_numpy(design, coefficients), np.einsum("nq,tqg->tng", dense, coefficients))
    values = np.stack([weights, 2 * weights])
    np.testing.assert_allclose(transpose_dot_numpy(design, values), np.einsum("nq,tng->tgq", dense, values))


@pytest.mark.parametrize("groups", [4, 40])
def test_indicator_contraction_and_scatter_reductions_agree(monkeypatch, groups):
    """The two segment-sum routes are one reduction written two ways.

    ``_segment_sum`` contracts against the indicator matrix for the few-group
    designs a batch covariate produces and scatters beyond that. Both must
    reproduce the dense original-coordinate algebra, including the reference
    sentinel rows that belong to no indicator column.
    """

    dense, weights, rhs = _problem(n=410, groups=groups, genes=3)
    design = detect_bordered_design(dense)
    assert design.num_groups == groups
    # The reference level must be present, or the sentinel path goes untested.
    assert np.any(np.asarray(design.codes) == design.num_groups)
    ridge = np.linspace(0.1, 1.0, dense.shape[1])
    expected = _dense_solution(_dense_information(dense, weights, ridge), rhs)
    expected_transpose = np.einsum("nq,ng->ngq", dense, np.ones_like(weights)).sum(axis=0)

    results = {}
    for name, limit in (("indicator", 256), ("scatter", 0)):
        monkeypatch.setattr(bordered_module, "_DENSE_SEGMENT_GROUP_LIMIT", limit)
        with _x64():
            info = weighted_information(design, weights, ridge)
            results[name] = (
                np.asarray(jax.jit(solve)(design, info, rhs)),
                np.asarray(transpose_dot(design, np.ones_like(weights))),
            )
        np.testing.assert_allclose(results[name][0], expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(results[name][1], expected_transpose, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(results["indicator"][0], results["scatter"][0], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(results["indicator"][1], results["scatter"][1], rtol=2e-12, atol=2e-12)


def test_indicator_contraction_respects_the_materialization_budget(monkeypatch):
    """A wide group axis must fall back rather than build a huge indicator."""

    dense, weights, _ = _problem(n=410, groups=40, genes=2)
    design = detect_bordered_design(dense)
    monkeypatch.setattr(bordered_module, "_DENSE_SEGMENT_ELEMENT_LIMIT", 1)
    with _x64():
        budgeted = np.asarray(transpose_dot(design, weights))
    monkeypatch.setattr(bordered_module, "_DENSE_SEGMENT_ELEMENT_LIMIT", 1 << 26)
    with _x64():
        contracted = np.asarray(transpose_dot(design, weights))
    np.testing.assert_allclose(budgeted, contracted, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(contracted, np.einsum("nq,ng->gq", dense, weights), rtol=2e-12, atol=2e-12)
