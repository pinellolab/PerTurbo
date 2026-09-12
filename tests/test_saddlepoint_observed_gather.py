"""Equivalence checks for the allocation-free observed-score gather."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from perturbo.core import PerTurboData
from perturbo._internal.jax_kernels import prepare_control_only_target_scores
from perturbo._internal.joint_laplace import prepare_joint_nb_design
import perturbo._internal.score_resampling as score_resampling


@pytest.mark.parametrize("q", [1, 3, 58])
def test_observed_target_scores_match_full_weighted_nuisance_buffer(q):
    rng = np.random.default_rng(7400 + q)
    cells, genes = max(q + 9, 24), 3
    nuisance = rng.normal(size=(cells, q)).astype(np.float32)
    if q == 1:
        nuisance[:, 0] = 1.0
    elif q == 58:
        nuisance.fill(0.0)
        nuisance[np.arange(cells), np.arange(cells) % 56] = 1.0
        nuisance[:, 56:] = rng.normal(size=(cells, 2))
    residual = rng.normal(size=(cells, genes)).astype(np.float32)
    weight = rng.uniform(0.1, 1.2, size=(cells, genes)).astype(np.float32)

    # Sentinel padding gives unequal effective widths. Rows 0 and 1 may be
    # controls carrying target labels; the gather only cares that they occur
    # once in the observed assignment.
    residual = np.concatenate([residual, np.zeros((1, genes), np.float32)])
    weight = np.concatenate([weight, np.zeros((1, genes), np.float32)])
    nuisance = np.concatenate([nuisance, np.zeros((1, q), np.float32)])
    sentinel = cells
    indices = np.array(
        [[0, 1, 7, 9, sentinel], [2, 5, sentinel, sentinel, sentinel],
         [sentinel, sentinel, sentinel, sentinel, sentinel]],
        dtype=np.int32,
    )
    a = rng.normal(size=(genes, q, q)).astype(np.float32)
    control_information = a @ np.swapaxes(a, -1, -2) + 2.0 * np.eye(q, dtype=np.float32)
    nuisance_score = rng.normal(size=(q, genes)).astype(np.float32)
    weighted = (weight[:, :, None] * nuisance[:, None, :]).reshape(cells + 1, genes * q)

    old = prepare_control_only_target_scores(
        jnp.asarray(indices), jnp.asarray(residual), jnp.asarray(weight), jnp.asarray(weighted),
        jnp.asarray(nuisance), jnp.asarray(control_information), jnp.asarray(nuisance_score),
    )
    new = prepare_control_only_target_scores(
        jnp.asarray(indices), jnp.asarray(residual), jnp.asarray(weight), None,
        jnp.asarray(nuisance), jnp.asarray(control_information), jnp.asarray(nuisance_score),
    )

    np.testing.assert_array_equal(new[0], old[0])
    np.testing.assert_array_equal(new[1], old[1])
    np.testing.assert_array_equal(new[2], old[2])
    assert np.isnan(np.asarray(new[2])[2]).all()


def _small_design():
    rng = np.random.default_rng(91)
    group_size = 24
    labels = np.repeat(np.arange(4), group_size)
    offsets = rng.normal(scale=0.15, size=labels.size)
    theta = np.array([3.0, 9.0])
    counts = rng.negative_binomial(
        theta, theta / (theta + np.exp(offsets[:, None] + 1.2))
    )
    data = PerTurboData(
        counts=jnp.asarray(counts),
        pert_id=jnp.asarray(labels),
        pert_names=["NTC", "a", "b", "c"],
        gene_names=["g0", "g1"],
        size_factors=jnp.asarray(offsets[:, None]),
    )
    return prepare_joint_nb_design(data, control_perturbations=["NTC"], dispersion=theta)


def test_saddlepoint_only_public_result_matches_legacy_buffer(monkeypatch):
    design = _small_design()
    append_sentinel = score_resampling.append_zero_weight_row

    def assert_no_full_buffer(residual, weight, weighted):
        assert weighted.shape == (residual.shape[0], 0)
        return append_sentinel(residual, weight, weighted)

    monkeypatch.setattr(score_resampling, "append_zero_weight_row", assert_no_full_buffer)
    kwargs = dict(
        num_resamples=9,
        backend="jax",
        null_model="control_only",
        tail_approximation="saddlepoint",
        saddlepoint_screen_p_value=1.0,
        saddlepoint_only=True,
    )
    optimized = score_resampling.run_low_moi_score_permutations(design, **kwargs)
    original = score_resampling.prepare_control_only_target_scores
    saw_none = False

    def legacy_wrapper(indices, residual, weight, weighted, nuisance, information, score):
        nonlocal saw_none
        saw_none = weighted is None
        genes, q = weight.shape[1], nuisance.shape[1]
        full = (weight[:, :, None] * nuisance[:, None, :]).reshape(weight.shape[0], genes * q)
        return original(indices, residual, weight, full, nuisance, information, score)

    monkeypatch.setattr(score_resampling, "prepare_control_only_target_scores", legacy_wrapper)
    legacy = score_resampling.run_low_moi_score_permutations(design, **kwargs)

    assert saw_none
    # XLA fuses the in-kernel product with its reduction, while the legacy
    # route materializes that float32 product first.  Their score differs by
    # at most a few float32 ulps; the independently computed SPA inputs and
    # tails below must remain bitwise identical.
    eps = np.finfo(np.float32).eps
    np.testing.assert_allclose(
        optimized.observed_score, legacy.observed_score, rtol=4 * eps, atol=eps
    )
    np.testing.assert_array_equal(optimized.saddlepoint_observed_sum, legacy.saddlepoint_observed_sum)
    np.testing.assert_array_equal(optimized.parametric_log_p_value, legacy.parametric_log_p_value)
    np.testing.assert_array_equal(optimized.parametric_p_value, legacy.parametric_p_value)
    np.testing.assert_array_equal(optimized.parametric_fit_valid, legacy.parametric_fit_valid)
    np.testing.assert_array_equal(optimized.parametric_used_fallback, legacy.parametric_used_fallback)
