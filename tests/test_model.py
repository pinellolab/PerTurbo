"""Basic tests for model variants."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from perturbo.log_normal_negative_binomial import LogNormalNegativeBinomial
from perturbo.model import (
    CensoredNegativeBinomialModel,
    GuideSharedNegativeBinomialModel,
    LogNormalNegativeBinomialModel,
    MixtureNegativeBinomialModel,
    NegBinModel,
    create_plates,
)


def _make_toy_data() -> tuple[jnp.ndarray, jnp.ndarray]:
    counts = jnp.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
            [2, 1, 0],
        ],
        dtype=jnp.int32,
    )
    pert_id = jnp.array([0, 1, 0, 1, 0])
    return counts, pert_id


def _trace_model(
    model,
    counts: jnp.ndarray,
    pert_id: jnp.ndarray,
    *,
    covariates: jnp.ndarray | None = None,
    num_factors: int | None = None,
    count_censoring_threshold: jnp.ndarray | None = None,
):
    seeded = numpyro.handlers.seed(model, jax.random.PRNGKey(0))
    return numpyro.handlers.trace(seeded).get_trace(
        counts,
        pert_id,
        covariates=covariates,
        num_cells=counts.shape[0],
        num_genes=counts.shape[1],
        num_perts=int(pert_id.max()) + 1,
        num_factors=num_factors,
        count_censoring_threshold=count_censoring_threshold,
    )


def test_negbin_model_runs_one_step() -> None:
    counts, pert_id = _make_toy_data()
    svi = SVI(
        NegBinModel,
        AutoNormal(NegBinModel),
        numpyro.optim.Adam(step_size=0.01),
        Trace_ELBO(),
    )
    init_key = jax.random.PRNGKey(0)
    svi_state = svi.init(
        init_key,
        counts,
        pert_id,
        num_cells=counts.shape[0],
        num_genes=counts.shape[1],
        num_perts=int(pert_id.max()) + 1,
    )
    svi.update(
        svi_state,
        counts,
        pert_id,
        num_cells=counts.shape[0],
        num_genes=counts.shape[1],
        num_perts=int(pert_id.max()) + 1,
    )



def _unmasked(fn):
    """The observation distribution beneath any MaskedDistribution wrappers.

    The censored NB path (commit a38145f) masks the observation site, and the
    other likelihoods now share that wrapper, so the tests look through it.
    """
    while isinstance(fn, dist.MaskedDistribution):
        fn = fn.base_dist
    return fn


def test_negbin_model_uses_negative_binomial_observation() -> None:
    counts, pert_id = _make_toy_data()
    trace = _trace_model(NegBinModel, counts, pert_id)
    assert isinstance(_unmasked(trace["obs"]["fn"]), dist.NegativeBinomialLogits)
    assert "noise_scale" not in trace


def test_perturbation_dispersion_adds_nonnegative_inverse_dispersion() -> None:
    counts = jnp.zeros((2, 1), dtype=jnp.int32)
    pert_id = jnp.array([0, 1], dtype=jnp.int32)
    model = numpyro.handlers.condition(
        NegBinModel,
        data={
            "beta_0": jnp.array([0.0]),
            "theta": jnp.array([2.0]),
            "beta": jnp.zeros((2, 1)),
            "dispersion_excess_inverse": jnp.array([[0.0], [0.5]]),
            "size_factor": jnp.zeros((2, 1)),
        },
    )
    trace = numpyro.handlers.trace(numpyro.handlers.seed(model, jax.random.PRNGKey(0))).get_trace(
        counts,
        pert_id,
        size_factors=jnp.zeros((2, 1)),
        num_cells=2,
        num_genes=1,
        num_perts=2,
        fit_perturbation_dispersion=True,
    )
    total_count = trace["obs"]["fn"].base_dist.total_count
    np.testing.assert_allclose(np.asarray(total_count).reshape(-1), [2.0, 1.0])


def test_guide_dispersion_sums_active_guides_in_inverse_dispersion() -> None:
    counts = jnp.zeros((2, 1), dtype=jnp.int32)
    guide_matrix = jnp.array([[1, 0, 1], [0, 1, 0]], dtype=jnp.float32)
    guide_to_element = jnp.array([[1, 0], [0, 1], [1, 0]], dtype=jnp.float32)
    model = numpyro.handlers.condition(
        GuideSharedNegativeBinomialModel,
        data={
            "beta_0": jnp.array([0.0]), "theta": jnp.array([2.0]),
            "beta": jnp.zeros((2, 1)), "size_factor": jnp.zeros((2, 1)),
            "guide_dispersion_excess_inverse": jnp.array([[0.25], [0.5], [0.75]]),
        },
    )
    trace = numpyro.handlers.trace(numpyro.handlers.seed(model, jax.random.PRNGKey(0))).get_trace(
        counts, jnp.zeros((2, 2), dtype=jnp.float32), guide_matrix=guide_matrix,
        guide_to_element=guide_to_element, size_factors=jnp.zeros((2, 1)),
        num_cells=2, num_genes=1, num_perts=2, num_guides=3,
        fit_perturbation_dispersion=True,
    )
    total_count = trace["obs"]["fn"].base_dist.total_count
    np.testing.assert_allclose(np.asarray(total_count).reshape(-1), [2.0 / 3.0, 1.0])


def test_lognormal_model_uses_lognormal_negative_binomial_observation() -> None:
    counts, pert_id = _make_toy_data()
    trace = _trace_model(LogNormalNegativeBinomialModel, counts, pert_id)
    assert isinstance(_unmasked(trace["obs"]["fn"]), LogNormalNegativeBinomial)
    assert "noise_scale" in trace


def test_censored_model_uses_masked_negative_binomial_observation() -> None:
    counts, pert_id = _make_toy_data()
    threshold = jnp.array([1, 0, 1], dtype=jnp.int32)
    trace = _trace_model(
        CensoredNegativeBinomialModel,
        counts,
        pert_id,
        count_censoring_threshold=threshold,
    )
    obs_fn = trace["obs"]["fn"]
    assert isinstance(obs_fn, dist.MaskedDistribution)
    assert isinstance(_unmasked(obs_fn), dist.NegativeBinomialLogits)
    expected_mask = counts <= threshold
    # The censoring mask may sit beneath another (all-true) mask; the product of
    # the wrappers' masks is what the observation actually sees.
    combined = jnp.ones_like(expected_mask, dtype=bool)
    fn = obs_fn
    while isinstance(fn, dist.MaskedDistribution):
        combined = combined & jnp.broadcast_to(jnp.asarray(fn._mask, dtype=bool), expected_mask.shape)
        fn = fn.base_dist
    assert jnp.array_equal(combined, expected_mask)


def test_mixture_nb_model_runs_one_step() -> None:
    counts, pert_id = _make_toy_data()
    svi = SVI(
        MixtureNegativeBinomialModel,
        AutoNormal(MixtureNegativeBinomialModel),
        numpyro.optim.Adam(step_size=0.01),
        Trace_ELBO(),
    )
    init_key = jax.random.PRNGKey(0)
    svi_state = svi.init(
        init_key,
        counts,
        pert_id,
        num_cells=counts.shape[0],
        num_genes=counts.shape[1],
        num_perts=int(pert_id.max()) + 1,
    )
    svi.update(
        svi_state,
        counts,
        pert_id,
        num_cells=counts.shape[0],
        num_genes=counts.shape[1],
        num_perts=int(pert_id.max()) + 1,
    )


def test_mixture_nb_model_uses_mixture_observation() -> None:
    counts, pert_id = _make_toy_data()
    trace = _trace_model(MixtureNegativeBinomialModel, counts, pert_id)
    assert isinstance(_unmasked(trace["obs"]["fn"]), dist.MixtureSameFamily)
    assert "pi_outlier" in trace
    assert "theta_outlier" in trace
    assert "outlier_mean_shift" in trace


def test_mixture_nb_outlier_component_is_independent_of_beta() -> None:
    counts = jnp.array([[0], [0]], dtype=jnp.int32)
    pert_id = jnp.array([0, 1], dtype=jnp.int32)
    params = {
        "beta_0": jnp.array([0.0], dtype=jnp.float32),
        "theta": jnp.array([1.0], dtype=jnp.float32),
        "beta": jnp.array([[0.0], [3.0]], dtype=jnp.float32),
        "pi_outlier": jnp.array([0.1], dtype=jnp.float32),
        "theta_outlier": jnp.array([0.5], dtype=jnp.float32),
        "outlier_mean_shift": jnp.array([1.0], dtype=jnp.float32),
        "size_factor": jnp.zeros((2, 1), dtype=jnp.float32),
    }
    model = numpyro.handlers.condition(MixtureNegativeBinomialModel, data=params)
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, jax.random.PRNGKey(0))
    ).get_trace(
        counts,
        pert_id,
        num_cells=2,
        num_genes=1,
        num_perts=2,
        size_factors=jnp.zeros((2, 1), dtype=jnp.float32),
    )
    logits_components = _unmasked(trace["obs"]["fn"]).component_distribution.logits
    # Inlier mode depends on beta, so these should differ.
    assert float(logits_components[0, 0, 0]) != float(logits_components[1, 0, 0])
    # Outlier mode is beta-independent baseline shift, so these should match.
    assert float(logits_components[0, 0, 1]) == float(logits_components[1, 0, 1])


def test_model_traces_latent_factors() -> None:
    counts, pert_id = _make_toy_data()
    trace = _trace_model(NegBinModel, counts, pert_id, num_factors=2)
    assert "factor_loadings" in trace
    assert "factor_scores" in trace
    assert trace["factor_loadings"]["value"].shape == (2, 1, counts.shape[1])
    assert trace["factor_scores"]["value"].shape == (2, counts.shape[0], 1)


def test_model_traces_covariate_coefficients() -> None:
    counts, pert_id = _make_toy_data()
    covariates = jnp.array(
        [
            [0.1, -1.0],
            [0.3, 0.5],
            [-0.2, 1.0],
            [1.4, -0.7],
            [0.0, 0.2],
        ],
        dtype=jnp.float32,
    )
    trace = _trace_model(NegBinModel, counts, pert_id, covariates=covariates)
    assert "covariate_coef" in trace
    assert trace["covariate_coef"]["value"].shape == (covariates.shape[1], counts.shape[1])


def test_model_samples_obs_when_counts_is_none() -> None:
    _, pert_id = _make_toy_data()
    num_cells = pert_id.shape[0]
    num_genes = 3
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(NegBinModel, jax.random.PRNGKey(0))
    ).get_trace(
        None,
        pert_id,
        num_cells=num_cells,
        num_genes=num_genes,
        num_perts=int(pert_id.max()) + 1,
    )
    assert "obs" in trace
    assert trace["obs"]["is_observed"] is False
    assert trace["obs"]["value"].shape == (num_cells, num_genes)


def test_model_can_skip_obs_sampling_explicitly() -> None:
    counts, pert_id = _make_toy_data()
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(NegBinModel, jax.random.PRNGKey(0))
    ).get_trace(
        counts,
        pert_id,
        num_cells=counts.shape[0],
        num_genes=counts.shape[1],
        num_perts=int(pert_id.max()) + 1,
        skip_obs_sampling=True,
    )
    assert "obs" not in trace


def test_model_fixed_subsample_scales_obs_and_local_sites_consistently() -> None:
    counts, pert_id = _make_toy_data()
    cell_idx = jnp.array([0, 2], dtype=jnp.int32)
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(NegBinModel, jax.random.PRNGKey(0))
    ).get_trace(
        counts,
        pert_id,
        num_cells=counts.shape[0],
        num_genes=counts.shape[1],
        num_perts=int(pert_id.max()) + 1,
        num_factors=2,
        cell_idx=cell_idx,
    )
    expected_scale = counts.shape[0] / cell_idx.shape[0]
    assert trace["size_factor"]["scale"] == expected_scale
    assert trace["factor_scores"]["scale"] == expected_scale
    assert trace["obs"]["scale"] == expected_scale
    assert trace["obs"]["value"].shape == (cell_idx.shape[0], counts.shape[1])


def test_model_random_subsample_scales_obs_and_local_sites_consistently() -> None:
    counts, pert_id = _make_toy_data()
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(NegBinModel, jax.random.PRNGKey(0))
    ).get_trace(
        counts,
        pert_id,
        num_cells=counts.shape[0],
        num_genes=counts.shape[1],
        num_perts=int(pert_id.max()) + 1,
        num_factors=2,
        subsample_size=2,
    )
    expected_scale = counts.shape[0] / 2
    assert trace["size_factor"]["scale"] == expected_scale
    assert trace["factor_scores"]["scale"] == expected_scale
    assert trace["obs"]["scale"] == expected_scale
    assert trace["obs"]["value"].shape == (2, counts.shape[1])


def test_autonormal_fixed_subsample_scales_local_sites_consistently() -> None:
    counts, pert_id = _make_toy_data()
    cell_idx = jnp.array([1, 4], dtype=jnp.int32)
    guide = AutoNormal(NegBinModel, create_plates=create_plates)
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(guide, jax.random.PRNGKey(0))
    ).get_trace(
        counts,
        pert_id,
        num_cells=counts.shape[0],
        num_genes=counts.shape[1],
        num_perts=int(pert_id.max()) + 1,
        num_factors=2,
        cell_idx=cell_idx,
    )
    expected_scale = counts.shape[0] / cell_idx.shape[0]
    assert trace["size_factor"]["scale"] == expected_scale
    assert trace["factor_scores"]["scale"] == expected_scale


def test_censored_negbin_model_runs_one_step() -> None:
    counts, pert_id = _make_toy_data()
    threshold = jnp.array([1, 1, 2], dtype=jnp.int32)
    svi = SVI(
        CensoredNegativeBinomialModel,
        AutoNormal(CensoredNegativeBinomialModel),
        numpyro.optim.Adam(step_size=0.01),
        Trace_ELBO(),
    )
    init_key = jax.random.PRNGKey(0)
    svi_state = svi.init(
        init_key,
        counts,
        pert_id,
        num_cells=counts.shape[0],
        num_genes=counts.shape[1],
        num_perts=int(pert_id.max()) + 1,
        count_censoring_threshold=threshold,
    )
    svi.update(
        svi_state,
        counts,
        pert_id,
        num_cells=counts.shape[0],
        num_genes=counts.shape[1],
        num_perts=int(pert_id.max()) + 1,
        count_censoring_threshold=threshold,
    )
