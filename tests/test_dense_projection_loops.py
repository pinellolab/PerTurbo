"""Numerical regression tests for the dense pool-projection q-squared loops.

The references here deliberately use ordinary NumPy/Python loops.  They
describe the implementation before those loops were moved into compiled JAX
control flow, including the omission of the cubic shift term from the dense
third-cumulant approximation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from perturbo._internal import saddlepoint  # noqa: E402
from perturbo._internal.saddlepoint import _LowMoiPoolProjection  # noqa: E402


def _case(q: int, weight_dtype):
    rng = np.random.default_rng(9100 + q)
    group_sizes = np.array([2, 5, 3, 4])
    n_controls, n_genes = max(q + 5, 13), 3
    n_cells = n_controls + int(group_sizes.sum())
    codes = np.full(n_cells, -1, dtype=np.int64)
    codes[n_controls:] = np.repeat(np.arange(group_sizes.size), group_sizes)
    control = np.arange(n_controls)
    own_rows = np.arange(n_controls, n_cells)

    if q == 58:
        # A realistic large nuisance basis: 56 batch indicators and two
        # continuous columns.  Cycle categories so every row has one batch.
        design = np.zeros((n_cells, q))
        design[np.arange(n_cells), np.arange(n_cells) % 56] = 1.0
        design[:, 56:] = rng.normal(size=(n_cells, 2))
    else:
        design = rng.normal(size=(n_cells, q))
        if q == 1:
            design[:, 0] = 1.0

    weight = rng.uniform(0.15, 1.8, size=(n_cells, n_genes)).astype(weight_dtype)
    contribution = rng.normal(size=(n_cells, n_genes))
    # Derive a well-conditioned control information matrix from the same
    # design, as callers do, while keeping unobserved one-hot levels solvable.
    information = np.empty((n_genes, q, q))
    for gene in range(n_genes):
        z = design[control]
        information[gene] = z.T @ (weight[control, gene, None] * z) + 0.4 * np.eye(q)

    return dict(
        weight=weight,
        design=design,
        information=information,
        contribution=contribution,
        control_rows=control,
        own_rows=own_rows,
        own_codes=codes[own_rows],
        all_rows=np.flatnonzero(codes >= 0),
        codes_all=codes[codes >= 0],
        target_valid=np.array([True, False, True, True]),
        num_cells=n_cells,
        num_genes=n_genes,
        num_targets=group_sizes.size,
    )


def _build(case):
    return _LowMoiPoolProjection.build(
        weight=case["weight"],
        nuisance_design=case["design"],
        batch_codes=None,
        control_information=case["information"],
        num_cells=case["num_cells"],
        num_genes=case["num_genes"],
        num_targets=case["num_targets"],
        control_rows=case["control_rows"],
        own_rows=case["own_rows"],
        own_codes=case["own_codes"],
        all_rows=case["all_rows"],
        codes_all=case["codes_all"],
        own_contribution=jnp.asarray(case["contribution"][case["own_rows"]]),
        target_valid=case["target_valid"],
    )


def _projection_oracle(case):
    z = case["design"]
    w = case["weight"].astype(np.float64)
    own_rows = case["own_rows"]
    own_codes = case["own_codes"]
    c_own = case["contribution"][own_rows]
    targets, genes, q = case["num_targets"], case["num_genes"], z.shape[1]
    score = np.zeros((targets, genes, q))
    own_information = np.zeros((targets, genes, q, q))

    for row_position, cell in enumerate(own_rows):
        target = own_codes[row_position]
        for gene in range(genes):
            score[target, gene] += z[cell] * c_own[row_position, gene]
            for k in range(q):
                for ell in range(q):
                    own_information[target, gene, k, ell] += (
                        w[cell, gene] * z[cell, k] * z[cell, ell]
                    )

    shift_direction = np.zeros_like(score)
    for target in range(targets):
        if not case["target_valid"][target]:
            continue
        for gene in range(genes):
            shift_direction[target, gene] = np.linalg.solve(
                case["information"][gene] + own_information[target, gene],
                score[target, gene],
            )

    corrected = np.empty_like(c_own)
    for position, cell in enumerate(own_rows):
        target = own_codes[position]
        corrected[position] = c_own[position] - w[cell] * (
            shift_direction[target] @ z[cell]
        )

    observed_shift = np.zeros((targets, genes))
    for cell, target in zip(case["all_rows"], case["codes_all"], strict=True):
        observed_shift[target] += w[cell] * (shift_direction[target] @ z[cell])
    return shift_direction, corrected, observed_shift


@pytest.mark.parametrize(
    ("q", "weight_dtype"),
    [(1, np.float32), (3, np.float64), (58, np.float32), (58, np.float64)],
)
def test_dense_projection_build_matches_python_oracle(q, weight_dtype, monkeypatch):
    case = _case(q, weight_dtype)
    if q == 3:
        # Exercise accumulation across row chunks, including a short final chunk.
        monkeypatch.setattr(saddlepoint, "_PROJECTION_ROWS_PER_CHUNK", 9)
    projection = _build(case)
    expected_e, expected_own, expected_observed_shift = _projection_oracle(case)

    np.testing.assert_allclose(projection.e, expected_e, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(projection.corrected_own, expected_own, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(
        projection.observed_shift, expected_observed_shift, rtol=2e-11, atol=2e-11
    )
    assert np.count_nonzero(np.asarray(projection.e)[~case["target_valid"]]) == 0


@pytest.mark.parametrize(("q", "weight_dtype"), [(1, np.float32), (4, np.float64), (58, np.float32)])
def test_dense_control_cumulants_match_explicit_shift_oracle(q, weight_dtype):
    case = _case(q, weight_dtype)
    projection = _build(case)
    rng = np.random.default_rng(2200 + q)
    targets = np.array([3, 0, 2], dtype=np.int32)
    controls, genes = len(case["control_rows"]), case["num_genes"]
    selection = rng.uniform(0.08, 0.91, size=(targets.size, controls))
    bernoulli = selection * (1.0 - selection)
    third_weight = bernoulli * (1.0 - 2.0 * selection)
    control_c = rng.normal(size=(controls, genes))
    base_mean = rng.normal(size=(targets.size, genes))
    base_variance = rng.uniform(0.5, 2.0, size=(targets.size, genes))
    base_third = rng.normal(size=(targets.size, genes))

    actual = projection.correct_control_cumulants(
        jnp.asarray(targets), jnp.asarray(selection), jnp.asarray(bernoulli),
        jnp.asarray(third_weight), jnp.asarray(control_c), jnp.asarray(control_c**2),
        jnp.asarray(base_mean), jnp.asarray(base_variance), jnp.asarray(base_third),
    )

    expected = [base_mean.copy(), base_variance.copy(), base_third.copy()]
    z_control = case["design"][case["control_rows"]]
    w_control = case["weight"][case["control_rows"]].astype(np.float64)
    e = np.asarray(projection.e)
    for batch, target in enumerate(targets):
        for gene in range(genes):
            nuisance_shift = w_control[:, gene] * (z_control @ e[target, gene])
            c = control_c[:, gene]
            expected[0][batch, gene] -= np.sum(selection[batch] * nuisance_shift)
            expected[1][batch, gene] += np.sum(
                bernoulli[batch] * (-2.0 * c * nuisance_shift + nuisance_shift**2)
            )
            # Preserve the dense approximation: terms through shift squared;
            # the cubic term is intentionally absent.
            expected[2][batch, gene] += np.sum(
                third_weight[batch]
                * (-3.0 * c**2 * nuisance_shift + 3.0 * c * nuisance_shift**2)
            )

    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=3e-11, atol=3e-11)
