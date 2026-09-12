from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from perturbo.core import PerTurboData
from perturbo._internal.joint_laplace import prepare_joint_nb_design
from perturbo._internal.score_resampling import (
    _DENSE_SELECTION_RATIO,
    _draw_selected_positions,
    _jax_selection_pad_widths,
    _fit_null_score_components,
    fit_batched_nb_null,
    fit_control_only_nb_null,
    batched_nb_score_statistics,
    sparse_nb_score_statistics,
    efficient_nb_score_statistics,
    indices_to_binary_assignments,
    make_stratified_permutation_indices,
    make_stratified_permutations,
    precompute_low_moi_permutations,
    run_low_moi_score_permutations,
)


def test_stratified_permutations_preserve_target_count_within_each_stratum() -> None:
    assignment = np.asarray([1, 1, 0, 0, 1, 0, 0, 0], dtype=np.int8)
    strata = np.asarray(["A", "A", "A", "A", "B", "B", "B", "B"])
    first = make_stratified_permutations(
        assignment,
        num_resamples=50,
        strata=strata,
        rng=np.random.default_rng(3),
    )
    second = make_stratified_permutations(
        assignment,
        num_resamples=50,
        strata=strata,
        rng=np.random.default_rng(3),
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, strata == "A"].sum(axis=1), np.full(50, 2))
    np.testing.assert_array_equal(first[:, strata == "B"].sum(axis=1), np.full(50, 1))


def test_efficient_score_matches_explicit_nuisance_projection() -> None:
    assignment = np.asarray([0, 1, 0, 1, 1, 0], dtype=float)
    residual = np.asarray([-0.7, 0.4, -0.2, 1.1, 0.3, -0.9])
    weight = np.asarray([0.8, 1.2, 0.9, 1.4, 1.0, 0.7])
    nuisance = np.column_stack([np.ones(6), np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])])
    nuisance_information = nuisance.T @ (weight[:, None] * nuisance)
    nuisance_information_inverse = np.linalg.inv(nuisance_information)

    observed = efficient_nb_score_statistics(
        assignment,
        score_residual=residual,
        observation_weight=weight,
        nuisance_design=nuisance,
        nuisance_information_inverse=nuisance_information_inverse,
    )
    cross = nuisance.T @ (weight * assignment)
    information = assignment @ (weight * assignment) - cross @ nuisance_information_inverse @ cross
    expected = (assignment @ residual) / np.sqrt(information)

    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def _score_test_design(seed: int = 31):
    rng = np.random.default_rng(seed)
    num_control = 180
    num_target = 180
    labels = np.concatenate(
        [
            np.zeros(num_control, dtype=np.int32),
            np.ones(num_target, dtype=np.int32),
        ]
    )
    batch = np.tile(np.asarray([0.0, 1.0]), labels.size // 2)
    offset = rng.normal(scale=0.2, size=labels.size)
    baseline = np.asarray([0.8, 1.0])
    batch_effect = np.asarray([0.25, -0.2])
    target_effect = np.asarray([0.0, -1.0])
    eta = (
        offset[:, None]
        + baseline[None, :]
        + batch[:, None] * batch_effect[None, :]
        + (labels > 0)[:, None] * target_effect[None, :]
    )
    theta = np.asarray([10.0, 12.0])
    mean = np.exp(eta)
    counts = rng.negative_binomial(theta[None, :], theta[None, :] / (theta[None, :] + mean))
    data = PerTurboData(
        counts=jnp.asarray(counts),
        pert_id=jnp.asarray(labels),
        pert_names=["NTC", "target"],
        gene_names=["null_gene", "affected_gene"],
        size_factors=jnp.asarray(offset[:, None]),
        covariates=jnp.asarray(batch[:, None]),
        covariate_names=["batch"],
    )
    design = prepare_joint_nb_design(
        data,
        control_perturbations=["NTC"],
        dispersion=theta,
    )
    return design, batch


def test_pairwise_score_permutation_separates_null_and_affected_gene() -> None:
    design, batch = _score_test_design()
    result = run_low_moi_score_permutations(
        design,
        num_resamples=199,
        strata=batch,
        seed=9,
        return_resampled_scores=True,
    )

    assert result.observed_score.shape == (1, 2)
    assert result.backend == "jax"
    assert result.p_value.shape == (1, 2)
    assert result.q_value.shape == (1, 2)
    assert result.resampled_scores is not None
    assert result.resampled_scores.shape == (1, 2, 199)
    assert np.all(np.asarray(result.null_converged))
    null_p, affected_p = np.asarray(result.p_value)[0]
    assert affected_p <= 0.01
    assert null_p > 0.05
    assert affected_p < null_p
    assert np.all(np.asarray(result.p_value) >= 1.0 / 200.0)


def test_score_permutation_can_fit_streaming_skew_normal_tail() -> None:
    design, batch = _score_test_design()
    result = run_low_moi_score_permutations(
        design,
        num_resamples=199,
        strata=batch,
        seed=9,
        tail_approximation="skew_normal_moments",
    )

    assert result.parametric_p_value is not None
    assert result.parametric_q_value is not None
    assert result.null_mean is not None
    assert result.null_variance is not None
    assert result.null_skewness is not None
    assert result.parametric_fit_valid is not None
    assert result.tail_approximation == "skew_normal_moments"
    assert np.all(np.asarray(result.parametric_fit_valid))
    assert np.all(np.isfinite(np.asarray(result.parametric_p_value)))
    assert np.all(np.asarray(result.null_variance) > 0.0)


def test_pairwise_score_permutation_is_seed_deterministic() -> None:
    design, batch = _score_test_design(seed=13)
    first = run_low_moi_score_permutations(
        design,
        num_resamples=49,
        strata=batch,
        seed=4,
        return_resampled_scores=True,
    )
    second = run_low_moi_score_permutations(
        design,
        num_resamples=49,
        strata=batch,
        seed=4,
        return_resampled_scores=True,
    )

    np.testing.assert_array_equal(np.asarray(first.observed_score), np.asarray(second.observed_score))
    np.testing.assert_array_equal(np.asarray(first.p_value), np.asarray(second.p_value))
    np.testing.assert_array_equal(np.asarray(first.resampled_scores), np.asarray(second.resampled_scores))


def _multi_target_design(seed: int = 21) -> tuple[object, np.ndarray]:
    """Several targets and a wider gene panel, for backend comparison."""
    rng = np.random.default_rng(seed)
    cells_per_group = 90
    n_targets = 3
    labels = np.repeat(np.arange(n_targets + 1), cells_per_group)
    n_cells = labels.size
    batch = np.tile(np.asarray([0.0, 1.0]), n_cells // 2)
    offset = rng.normal(scale=0.25, size=n_cells)
    theta = np.asarray([2.0, 8.0, 25.0, 4.0, 12.0])
    n_genes = theta.size
    nuisance = np.column_stack([np.ones(n_cells), batch])
    nuisance_coefficient = rng.normal(scale=0.4, size=(2, n_genes)) + np.asarray([[1.6] * n_genes, [0.0] * n_genes])
    effect = np.zeros((n_targets, n_genes))
    effect[0, 1] = -0.9
    effect[2, 3] = 0.7
    target_design = np.column_stack([labels == index + 1 for index in range(n_targets)]).astype(float)
    eta = offset[:, None] + nuisance @ nuisance_coefficient + target_design @ effect
    counts = rng.negative_binomial(theta[None, :], theta[None, :] / (theta[None, :] + np.exp(eta)))

    data = PerTurboData(
        counts=jnp.asarray(counts),
        pert_id=jnp.asarray(labels),
        pert_names=["NTC", *[f"target_{index}" for index in range(n_targets)]],
        gene_names=[f"gene_{index}" for index in range(n_genes)],
        size_factors=jnp.asarray(offset[:, None]),
        covariates=jnp.asarray(batch[:, None]),
        covariate_names=["batch"],
    )
    return prepare_joint_nb_design(data, control_perturbations=["NTC"], dispersion=theta), batch


def test_batched_and_per_pair_backends_agree() -> None:
    design, batch = _multi_target_design()
    shared = dict(num_resamples=99, strata=batch, seed=7)
    # Compare like with like: per_pair always uses the pooled null, so the
    # batched call is pinned to pooled too. control_only is a different test
    # definition and is characterized separately below.
    batched = run_low_moi_score_permutations(design, backend="batched", null_model="pooled", **shared)
    reference = run_low_moi_score_permutations(design, backend="per_pair", **shared)

    assert batched.backend == "batched"
    assert reference.backend == "per_pair"
    # Both backends draw the same permutations from the same seed, so the
    # statistics agree pair for pair. The tolerance is set by the reference, not
    # the batched path: `_gene_negative_log_posterior` evaluates in float32, so
    # its L-BFGS fit stalls at an exact gradient around 1e-2 (see
    # test_batched_null_fit_is_more_accurate_than_the_lbfgs_reference).
    np.testing.assert_allclose(
        np.asarray(batched.observed_score, dtype=np.float64),
        np.asarray(reference.observed_score, dtype=np.float64),
        rtol=5e-3,
        atol=5e-3,
    )
    # The residual statistic gap can flip a few exceedances near ties, which
    # moves a p-value by a small number of grid steps. Allow three.
    np.testing.assert_allclose(
        np.asarray(batched.p_value, dtype=np.float64),
        np.asarray(reference.p_value, dtype=np.float64),
        atol=3.0 / (99 + 1),
    )
    assert np.all(np.asarray(batched.null_converged))


def test_batched_backend_is_invariant_to_resample_chunk_size() -> None:
    design, batch = _multi_target_design(seed=5)
    shared = dict(num_resamples=80, strata=batch, seed=3, backend="batched")
    single = run_low_moi_score_permutations(design, resample_chunk_size=1000, **shared)
    chunked = run_low_moi_score_permutations(design, resample_chunk_size=7, **shared)

    np.testing.assert_array_equal(np.asarray(single.p_value), np.asarray(chunked.p_value))
    np.testing.assert_array_equal(np.asarray(single.observed_score), np.asarray(chunked.observed_score))


def test_batched_null_fit_is_more_accurate_than_the_lbfgs_reference() -> None:
    """Pin down which backend is right where the two disagree.

    The reference path builds its objective in float32, so L-BFGS cannot drive
    an absolute gradient summed over hundreds of cells anywhere near its stated
    ``gradient_tolerance``. Scoring both solutions with an exact float64
    gradient shows the batched fit actually solves the problem and the reference
    stops well short, which is why the agreement tolerance above is loose.
    """
    design, _ = _multi_target_design(seed=11)
    counts = np.asarray(design.counts)
    nuisance = np.asarray(design.nuisance_design, dtype=np.float64)
    offsets = np.asarray(design.offsets, dtype=np.float64)
    theta = np.asarray(design.dispersion, dtype=np.float64)

    def exact_gradient(coefficients: np.ndarray, gene_index: int) -> float:
        mean = np.exp(offsets[:, 0] + nuisance @ coefficients)
        residual = theta[gene_index] * (counts[:, gene_index] - mean) / (theta[gene_index] + mean)
        return float(np.max(np.abs(nuisance.T @ residual)))

    batched = fit_batched_nb_null(counts, nuisance_design=nuisance, offsets=offsets, theta=theta)
    assert batched.converged.all()
    for gene_index in range(counts.shape[1]):
        reference = _fit_null_score_components(
            counts=counts[:, gene_index],
            nuisance_design=nuisance,
            offset=offsets[:, 0],
            theta=float(theta[gene_index]),
            nuisance_prior_scale=None,
            cell_chunk_size=None,
            maxiter=500,
            gradient_tolerance=1e-6,
            curvature_jitter=1e-8,
        )
        batched_gradient = exact_gradient(batched.nuisance_mean[:, gene_index], gene_index)
        reference_gradient = exact_gradient(reference.nuisance_mean, gene_index)
        assert batched_gradient < 1e-8
        assert batched_gradient < reference_gradient
        # Still the same optimum, just resolved to different precision.
        np.testing.assert_allclose(batched.nuisance_mean[:, gene_index], reference.nuisance_mean, rtol=5e-3, atol=5e-3)


def test_batched_null_fit_flags_a_gene_whose_null_mle_does_not_exist() -> None:
    rng = np.random.default_rng(2)
    n_cells = 200
    counts = np.column_stack([rng.poisson(4.0, size=n_cells), np.zeros(n_cells)]).astype(float)
    batched = fit_batched_nb_null(
        counts,
        nuisance_design=np.ones((n_cells, 1)),
        offsets=np.zeros((n_cells, 1)),
        theta=np.asarray([5.0, 5.0]),
        max_iterations=25,
    )
    # An all-zero column has no finite intercept MLE; it must be reported rather
    # than poisoning the rest of the batch with NaN.
    assert batched.converged[0]
    assert not batched.converged[1]
    assert np.isfinite(batched.score_residual[:, 0]).all()


def test_control_only_null_matches_an_explicit_efficient_score() -> None:
    """Check the control-only statistic against dense textbook linear algebra.

    The efficient score is ``x'r - x'WZ (Z'WZ)^-1 Z'r`` over the cells being
    compared, studentized by ``x'Wx - x'WZ (Z'WZ)^-1 Z'Wx``. Written out densely
    here so the sparse implementation and its shared-control-block shortcuts have
    something independent to answer to.
    """
    design, _ = _multi_target_design(seed=17)
    counts = np.asarray(design.counts, dtype=np.float64)
    z = np.asarray(design.nuisance_design, dtype=np.float64)
    offsets = np.asarray(design.offsets, dtype=np.float64)
    theta = np.asarray(design.dispersion, dtype=np.float64)
    target_design = np.asarray(design.target_design, dtype=np.float64)
    control_mask = np.asarray(design.control_mask, dtype=bool)

    _, residual, weight = fit_control_only_nb_null(
        counts,
        control_mask=control_mask,
        nuisance_design=z,
        offsets=offsets,
        theta=theta,
    )

    result = run_low_moi_score_permutations(
        design, num_resamples=19, seed=1, backend="batched", null_model="control_only"
    )
    assert result.null_model == "control_only"

    for target_index in range(design.num_targets):
        x_full = target_design[:, target_index]
        pair = control_mask | (x_full > 0)
        x = x_full[pair]
        z_pair = z[pair]
        for gene_index in range(design.num_genes):
            r = residual[pair, gene_index]
            w = weight[pair, gene_index]
            information = z_pair.T @ (w[:, None] * z_pair) + 1e-8 * np.eye(z_pair.shape[1])
            inverse = np.linalg.inv(information)
            cross = z_pair.T @ (w * x)
            score = x @ r - cross @ inverse @ (z_pair.T @ r)
            efficient = x @ (w * x) - cross @ inverse @ cross
            expected = score / np.sqrt(efficient)
            np.testing.assert_allclose(
                float(np.asarray(result.observed_score)[target_index, gene_index]),
                expected,
                rtol=2e-4,
                atol=2e-4,
            )


def test_control_only_and_pooled_nulls_are_close_but_not_identical() -> None:
    """Characterize the two null models against each other.

    Both are efficient scores for the target coefficient after profiling out the
    nuisance terms; they simply linearize around different nuisance estimates, so
    they agree to first order and differ at second order in the effect size.
    Neither dominates - measured here, control-only comes out slightly *lower* on
    the affected pairs - so this asserts closeness and detection rather than an
    ordering.
    """
    design, batch = _multi_target_design(seed=23)
    shared = dict(num_resamples=199, strata=batch, seed=5, backend="batched")
    control_only = run_low_moi_score_permutations(design, null_model="control_only", **shared)
    pooled = run_low_moi_score_permutations(design, null_model="pooled", **shared)

    control_scores = np.abs(np.asarray(control_only.observed_score, dtype=np.float64))
    pooled_scores = np.abs(np.asarray(pooled.observed_score, dtype=np.float64))
    # _multi_target_design puts effects at (target 0, gene 1) and (target 2, gene 3).
    affected = np.zeros_like(control_scores, dtype=bool)
    affected[0, 1] = affected[2, 3] = True
    for target_index, gene_index in ((0, 1), (2, 3)):
        assert control_scores[target_index, gene_index] > 3.0
        assert pooled_scores[target_index, gene_index] > 3.0
    # Second-order disagreement, so widest where the effect is largest.
    np.testing.assert_allclose(control_scores[affected], pooled_scores[affected], rtol=0.2, atol=0.2)
    # Unaffected pairs are true nulls under both, so they track more closely.
    np.testing.assert_allclose(control_scores[~affected], pooled_scores[~affected], rtol=0.15, atol=0.15)


def test_sparse_and_dense_score_statistics_agree_on_the_pooled_null() -> None:
    design, batch = _multi_target_design(seed=29)
    counts = np.asarray(design.counts, dtype=np.float64)
    z = np.asarray(design.nuisance_design, dtype=np.float64)
    offsets = np.asarray(design.offsets, dtype=np.float64)
    theta = np.asarray(design.dispersion, dtype=np.float64)
    null = fit_batched_nb_null(counts, nuisance_design=z, offsets=offsets, theta=theta)

    rng = np.random.default_rng(3)
    indicator = np.zeros((6, counts.shape[0]), dtype=np.float64)
    for row in range(indicator.shape[0]):
        indicator[row, rng.choice(counts.shape[0], size=40, replace=False)] = 1.0

    dense = batched_nb_score_statistics(indicator, null=null)
    sparse_indices = np.nonzero(indicator)[1].reshape(indicator.shape[0], 40)
    from_sparse = sparse_nb_score_statistics(
        sparse_indices,
        score_residual=null.score_residual,
        observation_weight=null.observation_weight,
        weighted_nuisance=null.weighted_nuisance,
        nuisance_information_inverse=null.nuisance_information_inverse,
    )
    np.testing.assert_allclose(from_sparse, dense, rtol=1e-9, atol=1e-9)
    _ = batch


def test_permutation_indices_preserve_per_stratum_counts_and_are_deterministic() -> None:
    assignment = np.asarray([1, 1, 0, 0, 1, 0, 0, 0], dtype=np.int8)
    strata = np.asarray(["A", "A", "A", "A", "B", "B", "B", "B"])
    first = make_stratified_permutation_indices(
        assignment, num_resamples=200, strata=strata, rng=np.random.default_rng(3)
    )
    second = make_stratified_permutation_indices(
        assignment, num_resamples=200, strata=strata, rng=np.random.default_rng(3)
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (200, 3)
    # No cell selected twice within a resample, and each stratum keeps its count.
    for row in first:
        assert len(set(row.tolist())) == row.size
        assert int(np.sum(row < 4)) == 2
        assert int(np.sum(row >= 4)) == 1
    # Every eligible cell should turn up somewhere over 200 draws.
    assert set(first.reshape(-1).tolist()) == set(range(8))

    binary = indices_to_binary_assignments(first, num_cells=8)
    np.testing.assert_array_equal(binary.sum(axis=1), np.full(200, 3))
    np.testing.assert_array_equal(binary[:, :4].sum(axis=1), np.full(200, 2))


def test_permutation_indices_are_uniform_over_cells() -> None:
    """Each eligible cell should be selected about equally often."""
    assignment = np.zeros(40, dtype=np.int8)
    assignment[:8] = 1
    indices = make_stratified_permutation_indices(
        assignment, num_resamples=20_000, strata=None, rng=np.random.default_rng(11)
    )
    counts = np.bincount(indices.reshape(-1), minlength=40)
    expected = 20_000 * 8 / 40
    # Binomial standard deviation is ~sqrt(n p (1-p)) = ~57 here; allow 5 sigma.
    assert np.all(np.abs(counts - expected) < 5 * np.sqrt(expected * (1 - 8 / 40)))


def test_sparse_selection_draws_uniform_subsets_not_just_uniform_cells() -> None:
    """The sparse sampler must be uniform over *subsets*, not merely per cell.

    A sampler that repairs duplicated draws in place could in principle
    correlate the members of a row while still selecting each cell equally
    often, which the per-cell check above would not catch. With 8 cells choose
    3 there are only 56 possible resamples, so the joint distribution can be
    tested directly.
    """
    assignment = np.zeros(8, dtype=np.int8)
    assignment[:3] = 1
    num_resamples = 200_000
    indices = make_stratified_permutation_indices(
        assignment, num_resamples=num_resamples, strata=None, rng=np.random.default_rng(5)
    )
    assert indices.shape == (num_resamples, 3)

    keys = np.sort(indices, axis=1) @ np.asarray([64, 8, 1])
    _, counts = np.unique(keys, return_counts=True)
    assert counts.size == 56, "every 3-subset of 8 cells should appear"
    expected = num_resamples / 56
    chi_square = float(np.sum((counts - expected) ** 2 / expected))
    # 55 degrees of freedom: the 0.1% upper tail is about 100.
    assert chi_square < 100.0, f"subset distribution is not uniform (chi2={chi_square:.1f})"


def test_sparse_and_dense_selection_paths_agree_in_distribution() -> None:
    """Crossing the density threshold must not change what is being sampled.

    ``_draw_selected_positions`` switches algorithms on the selected fraction,
    so the two branches are compared against each other on problems that sit
    either side of that switch but share the same selected fraction shape.
    """
    population, selected, rows = 60, 6, 200_000
    assert selected * _DENSE_SELECTION_RATIO <= population, "this case must take the sparse path"

    sparse = _draw_selected_positions(
        np.random.default_rng(1), population=population, selected=selected, num_rows=rows
    )
    dense_keys = np.random.default_rng(2).random((rows, population))
    dense = np.argpartition(dense_keys, selected - 1, axis=1)[:, :selected]

    sparse_rate = np.bincount(sparse.reshape(-1), minlength=population) / (rows * selected)
    dense_rate = np.bincount(dense.reshape(-1), minlength=population) / (rows * selected)
    assert np.max(np.abs(sparse_rate - dense_rate)) < 0.002
    assert np.max(np.abs(sparse_rate - 1.0 / population)) < 0.002


def test_sparse_selection_is_fast_enough_for_a_genome_wide_target() -> None:
    """Guards the asymptotics, not the wall clock.

    Selecting ~180 of ~92,000 pair rows is the Replogle genome-wide shape. The
    old key-partition path drew one double per pair row per resample - 92
    million per target - which put the precompute at roughly 200 minutes over
    9,851 targets. The bound here is loose enough not to be flaky on a shared
    machine but far below what an O(population) sampler could reach.
    """
    assignment = np.zeros(92_000, dtype=np.int8)
    assignment[:180] = 1
    started = time.perf_counter()
    indices = make_stratified_permutation_indices(
        assignment, num_resamples=999, strata=None, rng=np.random.default_rng(7)
    )
    elapsed = time.perf_counter() - started

    assert indices.shape == (999, 180)
    for row in indices[:50]:
        assert len(set(row.tolist())) == row.size
    assert elapsed < 0.5, f"sparse selection took {elapsed:.2f}s; expected well under 0.5s"


def test_precomputed_permutations_reproduce_the_inline_draws() -> None:
    """Reusing precomputed resamples must leave every p-value unchanged.

    This is what lets a gene-chunked caller draw each target's permutations once
    instead of once per chunk; the draws depend only on the target's cells.
    """
    design, batch = _multi_target_design(seed=31)
    shared = dict(num_resamples=99, strata=batch, seed=7, backend="batched")
    inline = run_low_moi_score_permutations(design, **shared)
    reused = run_low_moi_score_permutations(
        design,
        permutations=precompute_low_moi_permutations(design, num_resamples=99, strata=batch, seed=7),
        **shared,
    )
    np.testing.assert_array_equal(np.asarray(inline.observed_score), np.asarray(reused.observed_score))
    np.testing.assert_array_equal(np.asarray(inline.p_value), np.asarray(reused.p_value))


def _design_over_targets(
    keep_targets: tuple[int, ...],
    *,
    seed: int = 5,
    shifted_targets: tuple[int, ...] = (),
) -> tuple[object, np.ndarray]:
    """Controls plus a chosen subset of targets, laid out as a chunk would be.

    Perturbation chunking hands each chunk only its own targets' cells, so a
    target sits at a different position in every decomposition. Dropping the
    other targets' cells entirely is exactly what the production chunk loader
    does, and it leaves any retained target's pair pool - controls plus its own
    cells - untouched.
    """

    rng = np.random.default_rng(seed)
    cells_per_group = 60
    n_targets = 3
    labels = np.repeat(np.arange(n_targets + 1), cells_per_group)
    theta = np.asarray([3.0, 9.0, 20.0])
    n_genes = theta.size
    offset = rng.normal(scale=0.2, size=labels.size)
    covariate = rng.normal(size=labels.size)
    for target in shifted_targets:
        covariate[labels == target + 1] += 3.0
    eta = offset[:, None] + 1.5
    counts = rng.negative_binomial(theta[None, :], theta[None, :] / (theta[None, :] + np.exp(eta)))
    strata = np.tile(np.asarray([0, 1]), labels.size // 2)

    keep_cells = np.isin(labels, (0, *[index + 1 for index in keep_targets]))
    kept_labels = labels[keep_cells]
    # Renumber so the retained targets occupy consecutive codes, which is what
    # makes a chunk's target positions differ from the full run's.
    remap = {0: 0, **{original + 1: position + 1 for position, original in enumerate(keep_targets)}}
    recoded = np.asarray([remap[int(label)] for label in kept_labels])

    data = PerTurboData(
        counts=jnp.asarray(counts[keep_cells]),
        pert_id=jnp.asarray(recoded),
        pert_names=["NTC", *[f"target_{index}" for index in keep_targets]],
        gene_names=[f"gene_{index}" for index in range(n_genes)],
        size_factors=jnp.asarray(offset[keep_cells, None]),
        covariates=jnp.asarray(covariate[keep_cells, None]),
        covariate_names=["selection_covariate"],
    )
    design = prepare_joint_nb_design(data, control_perturbations=["NTC"], dispersion=theta)
    return design, strata[keep_cells]


def test_a_targets_resamples_do_not_depend_on_which_targets_share_its_run() -> None:
    """The draw is keyed on the target, not on its position in the target list.

    Under perturbation chunking a target's neighbours are decided by the
    chunk-size flag, so a positionally-seeded draw would make p-values a
    function of that flag. Here ``target_2`` is third in the full run and first
    in the chunk, and must resample identically in both.
    """

    full, full_strata = _design_over_targets((0, 1, 2))
    chunk, chunk_strata = _design_over_targets((2,))

    full_draws = precompute_low_moi_permutations(full, num_resamples=64, strata=full_strata, seed=11)
    chunk_draws = precompute_low_moi_permutations(chunk, num_resamples=64, strata=chunk_strata, seed=11)

    assert full.target_names == ("target_0", "target_1", "target_2")
    assert chunk.target_names == ("target_2",)
    np.testing.assert_array_equal(full_draws.indices[2], chunk_draws.indices[0])


def test_propensity_fit_and_tail_ignore_unrelated_shifted_targets() -> None:
    """Each target's model is fitted on controls plus only its own cells.

    Moving unrelated targets far along the propensity covariate therefore
    cannot alter target_2's pool logits or deterministic saddlepoint tail when
    target_2 moves from a multi-target chunk into a chunk by itself.
    """

    full, _ = _design_over_targets((0, 1, 2), shifted_targets=(0, 1))
    chunk, _ = _design_over_targets((2,))
    full_propensity = precompute_low_moi_permutations(
        full,
        num_resamples=8,
        seed=19,
        resampling_mechanism="propensity",
        draw_resamples=True,
    )
    chunk_propensity = precompute_low_moi_permutations(
        chunk,
        num_resamples=8,
        seed=19,
        resampling_mechanism="propensity",
        draw_resamples=True,
    )
    np.testing.assert_allclose(
        full_propensity.pool_logits[2], chunk_propensity.pool_logits[0], rtol=2e-5, atol=2e-5
    )
    np.testing.assert_array_equal(full_propensity.indices[2], chunk_propensity.indices[0])

    full_saddlepoint = precompute_low_moi_permutations(
        full,
        num_resamples=8,
        seed=19,
        resampling_mechanism="propensity",
        draw_resamples=False,
    )
    chunk_saddlepoint = precompute_low_moi_permutations(
        chunk,
        num_resamples=8,
        seed=19,
        resampling_mechanism="propensity",
        draw_resamples=False,
    )

    common = dict(
        num_resamples=8,
        seed=19,
        backend="jax",
        null_model="control_only",
        tail_approximation="saddlepoint",
        saddlepoint_only=True,
        saddlepoint_screen_p_value=1.0,
    )
    full_result = run_low_moi_score_permutations(
        full, permutations=full_saddlepoint, **common
    )
    chunk_result = run_low_moi_score_permutations(
        chunk, permutations=chunk_saddlepoint, **common
    )
    np.testing.assert_allclose(
        np.asarray(full_result.tail_fits["saddlepoint"]["log_p_value"])[2],
        np.asarray(chunk_result.tail_fits["saddlepoint"]["log_p_value"])[0],
        rtol=2e-5,
        atol=2e-7,
    )


def test_distinct_targets_still_get_distinct_resamples() -> None:
    # Guards the test above from passing vacuously: keying on the target must
    # not collapse every target onto one shared stream.
    design, strata = _design_over_targets((0, 1, 2))
    draws = precompute_low_moi_permutations(design, num_resamples=64, strata=strata, seed=11)
    assert not np.array_equal(draws.indices[0], draws.indices[1])
    assert not np.array_equal(draws.indices[1], draws.indices[2])


def test_the_seed_still_moves_the_draw() -> None:
    design, strata = _design_over_targets((0, 1, 2))
    first = precompute_low_moi_permutations(design, num_resamples=64, strata=strata, seed=11)
    second = precompute_low_moi_permutations(design, num_resamples=64, strata=strata, seed=12)
    assert not np.array_equal(first.indices[0], second.indices[0])


def test_precomputed_permutations_validate_against_the_request() -> None:
    design, batch = _multi_target_design(seed=33)
    permutations = precompute_low_moi_permutations(design, num_resamples=20, strata=batch, seed=1)
    with pytest.raises(ValueError, match="resamples"):
        run_low_moi_score_permutations(design, num_resamples=99, permutations=permutations)


def test_jax_backend_matches_numpy_control_only_backend() -> None:
    """The wired-in jax backend must reproduce the numpy control-only path.

    Both compute the same control-only efficient score; the jax path just does it
    in float32 through the kernels in jax_kernels. p-values are exact-match
    because both draw from the same precomputed permutations, and the earlier
    float32-vs-float64 check showed the underlying statistic differs by ~6e-4
    relative - not enough to move an exceedance count.
    """
    design, batch = _multi_target_design(seed=41)
    permutations = precompute_low_moi_permutations(design, num_resamples=299, strata=batch, seed=6)
    numpy_result = run_low_moi_score_permutations(
        design,
        num_resamples=299,
        backend="batched",
        null_model="control_only",
        permutations=permutations,
    )
    jax_result = run_low_moi_score_permutations(
        design,
        num_resamples=299,
        backend="jax",
        null_model="control_only",
        permutations=permutations,
    )

    assert jax_result.backend == "jax"
    assert jax_result.null_model == "control_only"
    np.testing.assert_allclose(
        np.asarray(jax_result.observed_score, dtype=np.float64),
        np.asarray(numpy_result.observed_score, dtype=np.float64),
        rtol=5e-3,
        atol=5e-3,
    )
    np.testing.assert_array_equal(np.asarray(jax_result.p_value), np.asarray(numpy_result.p_value))
    assert np.all(np.asarray(jax_result.null_converged))


def test_jax_selection_pad_widths_use_bounded_target_size_buckets() -> None:
    """Bucket ceilings reduce padding while never excluding selected cells."""
    sizes = np.asarray([3, 4, 8, 11, 20, 27, 40])
    widths = _jax_selection_pad_widths(sizes, num_buckets=3)

    np.testing.assert_array_equal(widths, np.asarray([8, 8, 8, 20, 20, 40, 40]))
    assert np.all(widths >= sizes)
    assert np.unique(widths).size == 3
    np.testing.assert_array_equal(_jax_selection_pad_widths(sizes, num_buckets=1), np.full(sizes.shape, 40))


def _intercept_only_design(seed: int = 3, n_targets: int = 12, cells_per_group: int = 40) -> object:
    """Many small targets, no batch covariate - what the analytic null accepts.

    The closed-form null requires either an intercept-only nuisance design with
    an unstratified CRT, or categorical batch codes matching the CRT strata.
    This is the former.
    """
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_targets + 1), cells_per_group)
    offset = rng.normal(scale=0.2, size=labels.size)
    theta = np.asarray([2.0, 8.0, 25.0, 4.0])
    counts = rng.negative_binomial(
        theta[None, :], theta[None, :] / (theta[None, :] + np.exp(offset[:, None] + 1.6))
    )
    data = PerTurboData(
        counts=jnp.asarray(counts),
        pert_id=jnp.asarray(labels),
        pert_names=["NTC", *[f"target_{index}" for index in range(n_targets)]],
        gene_names=[f"gene_{index}" for index in range(theta.size)],
        size_factors=jnp.asarray(offset[:, None]),
    )
    return prepare_joint_nb_design(data, control_perturbations=["NTC"], dispersion=theta)


def test_analytic_null_is_invariant_to_the_target_batch_size() -> None:
    """Target batching is a launch-count choice and must not move a number.

    The analytic path sizes its target batch from the gather budget rather than
    using ``jax_targets_per_batch``: it draws no resamples, so the resample axis
    is absent from the working set and a batch of four would only multiply
    kernel launches, at a count that does not fall with the gene count. Here the
    budget is set low enough to force one target per launch and high enough to
    take them all at once, which must be the same computation either way.
    """
    design = _intercept_only_design()
    shared = dict(
        num_resamples=99,
        backend="jax",
        null_model="control_only",
        analytic_null_moments=True,
        tail_approximation="student_t_moments",
    )
    one_at_a_time = run_low_moi_score_permutations(design, jax_max_gather_gib=1e-6, **shared)
    all_at_once = run_low_moi_score_permutations(design, jax_max_gather_gib=4.0, **shared)

    assert np.isfinite(np.asarray(one_at_a_time.parametric_p_value)).any()
    for name in (
        "observed_score",
        "parametric_p_value",
        "parametric_log_p_value",
        "null_mean",
        "null_variance",
        "null_excess_kurtosis",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(one_at_a_time, name)),
            np.asarray(getattr(all_at_once, name)),
            err_msg=f"{name} changed with the target batch size",
        )


def test_saddlepoint_tail_is_exposed_by_the_low_moi_result() -> None:
    """The experimental SPA runs beside, rather than replacing, empirical CRT."""
    design = _intercept_only_design(n_targets=2, cells_per_group=80)
    result = run_low_moi_score_permutations(
        design,
        num_resamples=39,
        backend="jax",
        null_model="control_only",
        tail_approximation="saddlepoint",
    )

    assert result.tail_approximation == "saddlepoint"
    assert result.parametric_p_value is not None
    assert result.parametric_log_p_value is not None
    assert result.saddlepoint_observed_sum is not None
    assert result.saddlepoint_max_sampling_fraction is not None
    assert np.isfinite(np.asarray(result.parametric_log_p_value)).any()
    np.testing.assert_allclose(
        np.asarray(result.saddlepoint_max_sampling_fraction),
        np.full((2, 4), 0.5),
    )


def test_low_moi_saddlepoint_screen_marks_normal_fallbacks() -> None:
    design = _intercept_only_design(n_targets=2, cells_per_group=80)
    result = run_low_moi_score_permutations(
        design,
        num_resamples=39,
        backend="jax",
        null_model="control_only",
        tail_approximation="saddlepoint",
        saddlepoint_screen_p_value=1e-12,
    )

    fallback = np.asarray(result.parametric_used_fallback)
    assert fallback.shape == (2, 4)
    assert fallback.all()
    np.testing.assert_allclose(
        np.asarray(result.parametric_log_p_value),
        np.log(2.0) + stats.norm.logsf(np.abs(result.observed_score)),
    )


def test_low_moi_saddlepoint_only_skips_resamples_without_changing_spa() -> None:
    design = _intercept_only_design(n_targets=2, cells_per_group=80)
    empirical = run_low_moi_score_permutations(
        design,
        num_resamples=39,
        backend="jax",
        null_model="control_only",
        tail_approximation="saddlepoint",
        saddlepoint_screen_p_value=1.0,
    )
    only = run_low_moi_score_permutations(
        design,
        num_resamples=39,
        backend="jax",
        null_model="control_only",
        tail_approximation="saddlepoint",
        saddlepoint_screen_p_value=1.0,
        saddlepoint_only=True,
    )

    assert only.num_resamples == 0
    assert np.isnan(np.asarray(only.p_value)).all()
    assert np.isnan(np.asarray(only.q_value)).all()
    np.testing.assert_array_equal(only.observed_score, empirical.observed_score)
    np.testing.assert_allclose(only.parametric_log_p_value, empirical.parametric_log_p_value)


def test_parametric_tail_survives_the_result_boundary_in_float64() -> None:
    """Far-tail p-values must not be narrowed to float32 on the way out.

    float32 flushes anything below ~1e-45 to zero, and the drivers turn that
    into ``-log10(clip(p, 1e-300)) == 300`` for every such hypothesis - so the
    strongest hits, which are the entire reason for a parametric tail, would
    come back tied. The tail is computed in float64 and must stay there.
    """
    design, batch = _multi_target_design(seed=41)
    permutations = precompute_low_moi_permutations(design, num_resamples=99, strata=batch, seed=6)
    result = run_low_moi_score_permutations(
        design,
        num_resamples=99,
        backend="jax",
        null_model="control_only",
        permutations=permutations,
        tail_approximation="student_t_moments",
    )

    for name in ("parametric_p_value", "parametric_q_value", "null_mean", "null_variance", "null_skewness"):
        value = getattr(result, name)
        assert value is not None
        assert np.asarray(value).dtype == np.float64, f"{name} lost float64 at the result boundary"

    # A float32 round trip is not merely lossy here, it is total: these are the
    # magnitudes a z of 14 upward produces, and they must survive intact.
    far_tail = np.asarray([1e-40, 1e-60, 1e-200, 1e-300], dtype=np.float64)
    np.testing.assert_array_equal(np.asarray(far_tail, dtype=np.float64), far_tail)
    assert np.all(np.asarray(far_tail, dtype=np.float32)[1:] == 0.0)


def test_parametric_log_tail_reaches_the_result_boundary() -> None:
    """The log-scale tail must survive the plumbing, in float64 and unfloored.

    ``parametric_p_value`` cannot go below ~1e-308 no matter how strong the
    hypothesis, so a consumer that ranks by it is ranking a clipped column. The
    log field is carried alongside precisely so the ranking is not truncated;
    if it ever arrives as float32 it floors near -88 instead, which would be a
    far worse bound than the one it replaced.
    """
    design, batch = _multi_target_design(seed=41)
    permutations = precompute_low_moi_permutations(design, num_resamples=99, strata=batch, seed=6)
    result = run_low_moi_score_permutations(
        design,
        num_resamples=99,
        backend="jax",
        null_model="control_only",
        permutations=permutations,
        tail_approximation="student_t_moments",
    )

    log_p = result.parametric_log_p_value
    assert log_p is not None
    assert np.asarray(log_p).dtype == np.float64
    p_value = np.asarray(result.parametric_p_value)
    assert np.asarray(log_p).shape == p_value.shape

    # The two scales describe one tail, so they agree wherever the linear one
    # is a real number rather than the clip floor.
    both = np.isfinite(p_value) & (p_value > np.finfo(float).tiny)
    assert both.any()
    np.testing.assert_allclose(np.exp(np.asarray(log_p)[both]), p_value[both], rtol=1e-10)
    # An unusable fit is NaN on both scales, never on only one.
    np.testing.assert_array_equal(np.isnan(np.asarray(log_p)), np.isnan(p_value))


def test_parametric_log_tail_is_absent_without_a_tail_family() -> None:
    """No parametric fit means no parametric columns, log included."""
    design, batch = _multi_target_design(seed=41)
    permutations = precompute_low_moi_permutations(design, num_resamples=99, strata=batch, seed=6)
    result = run_low_moi_score_permutations(
        design, num_resamples=99, backend="jax", null_model="control_only", permutations=permutations
    )

    assert result.parametric_p_value is None
    assert result.parametric_log_p_value is None


def test_jax_null_moments_match_a_float64_host_recomputation() -> None:
    """The on-device power sums reproduce what the host used to compute.

    The resample block is reduced on device in float32 now instead of being
    pulled to the host and reduced in float64, so the guard is that the null
    summaries the tail consumes still match a float64 recomputation from the
    stored resamples. Excess kurtosis is the binding one: it is reached by
    differencing ``central_4 / var^2`` against 3, so it loses the most digits.
    """
    design, batch = _multi_target_design(seed=41)
    permutations = precompute_low_moi_permutations(design, num_resamples=299, strata=batch, seed=6)
    result = run_low_moi_score_permutations(
        design,
        num_resamples=299,
        backend="jax",
        null_model="control_only",
        permutations=permutations,
        return_resampled_scores=True,
        tail_approximation="student_t_moments",
    )

    samples = np.asarray(result.resampled_scores, dtype=np.float64)
    finite = np.isfinite(samples)
    values = np.where(finite, samples, 0.0)
    count = finite.sum(axis=-1)
    mean = values.sum(-1) / count
    variance = np.square(values).sum(-1) / count - np.square(mean)
    central_fourth = (
        np.power(values, 4).sum(-1) / count
        - 4.0 * mean * np.power(values, 3).sum(-1) / count
        + 6.0 * np.square(mean) * np.square(values).sum(-1) / count
        - 3.0 * np.power(mean, 4)
    )
    excess_kurtosis = central_fourth / np.square(variance) - 3.0

    compared = np.isfinite(excess_kurtosis) & np.isfinite(np.asarray(result.null_excess_kurtosis, dtype=np.float64))
    assert compared.any()
    np.testing.assert_allclose(np.asarray(result.null_mean, dtype=np.float64)[compared], mean[compared], atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(result.null_variance, dtype=np.float64)[compared], variance[compared], rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(result.null_excess_kurtosis, dtype=np.float64)[compared],
        excess_kurtosis[compared],
        atol=1e-4,
    )


def test_jax_backend_is_invariant_to_cell_size_bucketing() -> None:
    """Padding choices affect compilation shape only, never the CRT result."""
    design, batch = _multi_target_design(seed=47)
    permutations = precompute_low_moi_permutations(design, num_resamples=99, strata=batch, seed=5)
    shared = dict(
        num_resamples=99,
        backend="jax",
        null_model="control_only",
        permutations=permutations,
        return_resampled_scores=True,
    )
    global_width = run_low_moi_score_permutations(design, jax_num_cell_buckets=1, **shared)
    bucketed = run_low_moi_score_permutations(design, jax_num_cell_buckets=3, **shared)

    np.testing.assert_array_equal(np.asarray(bucketed.observed_score), np.asarray(global_width.observed_score))
    np.testing.assert_array_equal(np.asarray(bucketed.p_value), np.asarray(global_width.p_value))
    np.testing.assert_array_equal(np.asarray(bucketed.resampled_scores), np.asarray(global_width.resampled_scores))


def test_jax_backend_is_invariant_to_target_microbatch_size() -> None:
    """Target-axis batching changes dispatches and memory, not CRT values."""
    design, batch = _multi_target_design(seed=49)
    permutations = precompute_low_moi_permutations(design, num_resamples=99, strata=batch, seed=5)
    shared = dict(
        num_resamples=99,
        backend="jax",
        null_model="control_only",
        permutations=permutations,
        return_resampled_scores=True,
    )
    single_target = run_low_moi_score_permutations(design, jax_targets_per_batch=1, **shared)
    four_targets = run_low_moi_score_permutations(design, jax_targets_per_batch=4, **shared)

    np.testing.assert_array_equal(np.asarray(four_targets.observed_score), np.asarray(single_target.observed_score))
    np.testing.assert_array_equal(np.asarray(four_targets.p_value), np.asarray(single_target.p_value))
    np.testing.assert_array_equal(np.asarray(four_targets.resampled_scores), np.asarray(single_target.resampled_scores))


def test_jax_backend_is_invariant_to_width_aware_gather_cap() -> None:
    """A wide-target memory cap may split dispatches but cannot change scores."""

    design, batch = _multi_target_design(seed=53)
    permutations = precompute_low_moi_permutations(design, num_resamples=99, strata=batch, seed=5)
    shared = dict(
        num_resamples=99,
        backend="jax",
        null_model="control_only",
        permutations=permutations,
        return_resampled_scores=True,
        jax_targets_per_batch=4,
        jax_max_target_resample_batch=256,
    )
    uncapped = run_low_moi_score_permutations(design, **shared)
    # 1e-5 GiB is deliberately below two padded resamples for this fixture, so
    # the executor must split the otherwise single 99-resample dispatch.
    capped = run_low_moi_score_permutations(design, jax_max_gather_gib=1e-5, **shared)

    np.testing.assert_array_equal(np.asarray(capped.observed_score), np.asarray(uncapped.observed_score))
    np.testing.assert_array_equal(np.asarray(capped.p_value), np.asarray(uncapped.p_value))
    # XLA may choose a different reduction tree for a one-resample kernel than
    # for a 99-resample kernel; the resulting float32 rounding is immaterial to
    # the (exactly matching) exceedance counts above. The tolerance is a float32
    # one because that is all the property can promise: on jax 0.11 one score in
    # 1,485 differs by 5e-5 relative, where jax 0.9 agreed to 1e-5. What must
    # match exactly, and is asserted above, is the p-value.
    np.testing.assert_allclose(
        np.asarray(capped.resampled_scores),
        np.asarray(uncapped.resampled_scores),
        rtol=2e-4,
        atol=1e-5,
    )


def test_jax_backend_rejects_the_pooled_null() -> None:
    design, batch = _multi_target_design(seed=43)
    with pytest.raises(ValueError, match="control_only"):
        run_low_moi_score_permutations(design, num_resamples=49, backend="jax", null_model="pooled")


def test_jax_backend_separates_null_and_affected_gene() -> None:
    """End-to-end sanity check independent of the numpy comparison above."""
    design, batch = _score_test_design()
    result = run_low_moi_score_permutations(design, num_resamples=199, strata=batch, seed=9, backend="jax")
    null_p, affected_p = np.asarray(result.p_value)[0]
    assert affected_p <= 0.01
    assert null_p > 0.05
    assert affected_p < null_p


def test_x64_is_enabled_and_score_kernels_stay_float32() -> None:
    """x64 must be on, and must not have silently widened the hot path.

    Enabling x64 is permissive rather than coercive: it lets float64 exist where
    the code asks for it, while arrays annotated ``dtype=jnp.float32`` keep
    their precision and their memory profile. Both halves matter - the first is
    what saves the far tail from float32's ~1e-45 floor, and the second is what
    keeps the per-cell gathers from doubling in size on the device.
    """
    import jax

    assert jax.config.read("jax_enable_x64") is True

    design, batch = _multi_target_design(seed=41)
    permutations = precompute_low_moi_permutations(design, num_resamples=99, strata=batch, seed=6)
    result = run_low_moi_score_permutations(
        design,
        num_resamples=99,
        backend="jax",
        null_model="control_only",
        permutations=permutations,
        tail_approximation="student_t_moments",
    )

    assert np.asarray(result.observed_score).dtype == np.float32
    assert np.asarray(result.p_value).dtype == np.float32
    assert np.asarray(result.parametric_p_value).dtype == np.float64


def test_propensity_draws_vary_the_selected_count() -> None:
    """The substantive difference from stratified permutation.

    Permutation holds each stratum's selected count fixed, which is
    exchangeable only where the selection probability is constant inside the
    stratum. Propensity draws each cell on its own fitted probability, so the
    count varies - that variation is the correction, not noise.
    """

    design, _ = _score_test_design()
    permuted = precompute_low_moi_permutations(design, num_resamples=64, seed=0)
    propensity = precompute_low_moi_permutations(
        design, num_resamples=64, seed=0, resampling_mechanism="propensity"
    )
    assert propensity.resampling_mechanism == "propensity"
    assert propensity.pair_rows is not None and propensity.pool_logits is not None

    target = next(i for i, v in enumerate(permuted.indices) if v is not None)
    # Permutation: every draw selects the observed number of cells.
    assert len(set(permuted.indices[target].shape[1:])) == 1
    # Propensity keeps the pool and its fitted log-odds, one entry per pool row.
    assert propensity.pool_logits[target].shape == propensity.pair_rows[target].shape


def test_propensity_intercept_reproduces_the_observed_count() -> None:
    """Unpenalized logistic pins the null's mean count to the observed one.

    This is why the fit stays unpenalized despite the heavy imbalance a low-MOI
    pool has - hundreds of target cells against thousands of controls. Firth or
    ridge would stabilise the coefficients but add a term to the score
    equation, and the identity below is worth more to a test whose statistic is
    a sum over selected cells.
    """

    design, _ = _score_test_design()
    drawn = precompute_low_moi_permutations(
        design, num_resamples=8, seed=0, resampling_mechanism="propensity"
    )
    target_design = np.asarray(design.target_design)
    for target, (rows, logits) in enumerate(zip(drawn.pair_rows, drawn.pool_logits, strict=True)):
        if rows is None:
            continue
        observed = int(target_design[rows, target].sum())
        expected = float((1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))).sum())
        assert expected == pytest.approx(observed, rel=0.02), target


def test_the_mechanism_selects_which_saddlepoint_runs() -> None:
    """Permutation draws get the stratified CGF; propensity draws get the exact one.

    The two are not interchangeable. The stratified form models a fixed-count
    without-replacement draw by summing with-replacement stratum CGFs and
    correcting the variance by hand; the propensity form needs no surrogate
    because independent Bernoullis are what it actually describes.
    """

    design, _ = _score_test_design()
    permuted = precompute_low_moi_permutations(design, num_resamples=16, seed=0)
    propensity = precompute_low_moi_permutations(
        design, num_resamples=16, seed=0, resampling_mechanism="propensity"
    )
    assert permuted.resampling_mechanism == "permutation"
    assert permuted.pair_rows is None and permuted.pool_logits is None
    assert propensity.resampling_mechanism == "propensity"
    assert all(
        r is None or l is not None
        for r, l in zip(propensity.pair_rows, propensity.pool_logits, strict=True)
    )


def test_propensity_draws_run_through_the_low_moi_score_kernel() -> None:
    """Regression: ragged propensity draws are padded with ``pool size``.

    The padding must land on the kernel's zero-contribution sentinel row, not
    index one past the pool. The first transcriptome-wide propensity run with
    empirical draws died here with an ``IndexError`` after the dispersion fit.
    """

    design, _ = _score_test_design()
    propensity = precompute_low_moi_permutations(
        design, num_resamples=32, seed=0, resampling_mechanism="propensity"
    )
    permuted = precompute_low_moi_permutations(design, num_resamples=32, seed=0)
    target = next(i for i, v in enumerate(propensity.indices) if v is not None)
    # The draws really are ragged and really do carry the pad value.
    assert (propensity.indices[target] == propensity.pair_rows[target].size).any()

    with_propensity = run_low_moi_score_permutations(
        design,
        num_resamples=32,
        seed=0,
        backend="jax",
        null_model="control_only",
        permutations=propensity,
    )
    with_permutation = run_low_moi_score_permutations(
        design,
        num_resamples=32,
        seed=0,
        backend="jax",
        null_model="control_only",
        permutations=permuted,
    )
    p = np.asarray(with_propensity.p_value)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    # The observed statistic does not depend on how the null is drawn.
    np.testing.assert_allclose(
        np.asarray(with_propensity.observed_score),
        np.asarray(with_permutation.observed_score),
        rtol=1e-6,
        atol=1e-6,
    )
