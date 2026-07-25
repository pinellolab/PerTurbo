"""Tests for the censored negative binomial likelihood."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
import scipy.sparse as sp

from perturbo.api import PerTurboData, fit_perturbation_effects, fit_control
from perturbo.censored_negative_binomial import CensoredNegativeBinomial, compute_gene_count_censoring_thresholds


def test_censored_negative_binomial_uses_tail_mass_above_threshold() -> None:
    logits = jnp.array([0.0, 0.0])
    total_count = jnp.array([3.0, 3.0])
    threshold = 2
    sample = jnp.array([2, 5])

    censored_nb = CensoredNegativeBinomial(
        total_count=total_count,
        logits=logits,
        censoring_threshold=threshold,
    )
    neg_bin = dist.NegativeBinomialLogits(total_count=total_count, logits=logits)

    expected = jnp.array(
        [
            neg_bin.log_prob(sample)[0],
            jnp.log1p(-neg_bin.cdf(jnp.asarray(threshold, dtype=jnp.float32))[1]),
        ]
    )
    assert jnp.allclose(censored_nb.log_prob(sample), expected)


def test_compute_gene_count_censoring_thresholds_uses_per_gene_ceiling_quantiles() -> None:
    counts = np.array(
        [
            [0, 0, 1],
            [1, 2, 3],
            [2, 4, 9],
            [9, 6, 12],
        ],
        dtype=np.int32,
    )
    thresholds = compute_gene_count_censoring_thresholds(counts, percentile=75.0, threshold_floor=0)
    np.testing.assert_array_equal(thresholds, np.array([4, 5, 10], dtype=np.int32))


def test_compute_gene_count_censoring_thresholds_accepts_sparse_counts() -> None:
    counts = sp.csr_matrix(
        np.array(
            [
                [0, 0, 1],
                [1, 2, 3],
                [2, 4, 9],
                [9, 6, 12],
            ],
            dtype=np.int32,
        )
    )
    thresholds = compute_gene_count_censoring_thresholds(counts, percentile=75.0, threshold_floor=0)
    np.testing.assert_array_equal(thresholds, np.array([4, 5, 10], dtype=np.int32))


def test_api_supports_censored_negative_binomial_likelihood() -> None:
    counts = jnp.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
            [0, 12, 0],
            [9, 0, 1],
        ],
        dtype=jnp.int32,
    )
    pert_id = jnp.array([0, 1, 0, 1, 0, 1])
    data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2", "g3"],
    )
    control_fit = fit_control(
        data,
        num_steps=1,
        prior="normal",
        model_name="censored_nb",
        count_censoring_percentile=99.5,
        minibatch_size=3,
    )
    beta_fit = fit_perturbation_effects(
        data,
        control_fit,
        num_steps=1,
        prior="normal",
        model_name="censored_negbin",
        count_censoring_percentile=99.5,
        minibatch_size=3,
    )

    assert beta_fit.posterior_mean.shape == (2, counts.shape[1])
    assert beta_fit.posterior_scale.shape == (2, counts.shape[1])


def test_api_requires_explicit_censoring_percentile_for_censored_likelihood() -> None:
    data = PerTurboData(
        counts=jnp.array([[0, 1], [2, 0]], dtype=jnp.int32),
        pert_id=jnp.array([0, 1], dtype=jnp.int32),
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2"],
    )

    with pytest.raises(ValueError, match="count_censoring_percentile"):
        fit_control(
            data,
            num_steps=1,
            prior="normal",
            model_name="censored_nb",
        )


def test_stage2_reuses_stage1_censoring_threshold_without_recomputing() -> None:
    counts = jnp.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
            [0, 12, 0],
            [9, 0, 1],
        ],
        dtype=jnp.int32,
    )
    pert_id = jnp.array([0, 1, 0, 1, 0, 1])
    data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2", "g3"],
    )
    control_fit = fit_control(
        data,
        num_steps=1,
        prior="normal",
        model_name="censored_nb",
        count_censoring_percentile=99.5,
        minibatch_size=3,
    )
    beta_fit = fit_perturbation_effects(
        data,
        control_fit,
        num_steps=1,
        prior="normal",
        model_name="censored_nb",
        minibatch_size=3,
    )

    assert control_fit.count_censoring_threshold is not None
    assert beta_fit.posterior_mean.shape == (2, counts.shape[1])


def test_stage2_censored_requires_stage1_censoring_threshold() -> None:
    data = PerTurboData(
        counts=jnp.array([[0, 1], [2, 0], [0, 3], [1, 0]], dtype=jnp.int32),
        pert_id=jnp.array([0, 1, 0, 1], dtype=jnp.int32),
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2"],
    )
    control_fit = fit_control(
        data,
        num_steps=1,
        prior="normal",
        model_name="nb",
        minibatch_size=2,
    )

    with pytest.raises(ValueError, match="control_fit.count_censoring_threshold"):
        fit_perturbation_effects(
            data,
            control_fit,
            num_steps=1,
            prior="normal",
            model_name="censored_nb",
            minibatch_size=2,
        )
