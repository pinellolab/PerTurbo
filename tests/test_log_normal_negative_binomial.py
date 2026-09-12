"""Tests for the log-normal negative binomial utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from perturbo.api import PerTurboData, fit_perturbation_effects, fit_control
from perturbo.log_normal_negative_binomial import LogNormalNegativeBinomial
from perturbo.model import LogNormalNegativeBinomialModel


def test_lognormal_nb_matches_negative_binomial_at_zero_noise() -> None:
    logits = jnp.zeros((2, 3))
    total_count = jnp.array([[5.0, 2.5, 3.5], [4.0, 1.0, 6.0]])
    noise_scale = jnp.zeros_like(logits)
    sample = jnp.array([[0, 1, 0], [2, 0, 1]])

    lognormal_nb = LogNormalNegativeBinomial(total_count, logits, noise_scale, num_quad_points=6)
    neg_bin = dist.NegativeBinomialLogits(total_count=total_count, logits=logits)

    assert jnp.allclose(lognormal_nb.log_prob(sample), neg_bin.log_prob(sample))


def test_lognormal_negative_binomial_model_runs_one_step() -> None:
    key = jax.random.PRNGKey(0)
    num_cells = 6
    num_genes = 3
    num_perts = 2
    counts = jnp.eye(num_genes, num_cells).T.astype(jnp.int32)
    pert_id = jnp.arange(num_cells) % num_perts

    svi = SVI(
        LogNormalNegativeBinomialModel,
        AutoNormal(LogNormalNegativeBinomialModel),
        numpyro.optim.Adam(step_size=0.01),
        Trace_ELBO(),
    )
    init_key, step_key = jax.random.split(key)
    svi_state = svi.init(
        init_key,
        counts,
        pert_id,
        num_cells=num_cells,
        num_genes=num_genes,
        num_perts=num_perts,
    )
    svi.update(
        svi_state,
        # step_key,
        counts,
        pert_id,
        num_cells=num_cells,
        num_genes=num_genes,
        num_perts=num_perts,
    )


def test_api_supports_lognormal_likelihood() -> None:
    counts = jnp.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
        ],
        dtype=jnp.int32,
    )
    pert_id = jnp.array([0, 1, 0, 1])
    data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2", "g3"],
    )
    control_fit = fit_control(data, num_steps=1, prior="cauchy", model_name="lognormal_nb")
    fit_perturbation_effects(data, control_fit, num_steps=1, prior="normal", model_name="lognormal_nb")


def test_api_supports_lognormal_likelihood_with_baseline_uncertainty() -> None:
    counts = jnp.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
        ],
        dtype=jnp.int32,
    )
    pert_id = jnp.array([0, 1, 0, 1])
    data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2", "g3"],
    )
    control_fit = fit_control(data, num_steps=1, prior="cauchy", model_name="lognormal_nb")
    beta_fit = fit_perturbation_effects(
        data,
        control_fit,
        num_steps=1,
        prior="normal",
        model_name="lognormal_nb",
        propagate_baseline_uncertainty=True,
    )
    assert beta_fit.posterior_mean.shape == (2, counts.shape[1])
    assert beta_fit.posterior_scale.shape == (2, counts.shape[1])


def test_nonzero_noise_likelihood_matches_independent_integration() -> None:
    """The quadrature must integrate the log-normal mixture, not a reweighting of it.

    ``hermegauss`` returns weights; using them as log-weights made every mixture with
    a non-zero noise scale wrong. Ported from PerTurbo #59.
    """
    import numpy as np
    from scipy.integrate import quad
    from scipy.special import expit
    from scipy.stats import nbinom, norm

    total_count, logits, noise = 5.0, 0.0, 0.5
    distribution = LogNormalNegativeBinomial(
        jnp.asarray(total_count), jnp.asarray(logits), jnp.asarray(noise), num_quad_points=32
    )
    values = np.array([0, 1, 5, 15])
    expected = np.array(
        [
            quad(
                lambda z: nbinom.pmf(value, total_count, expit(-logits - noise * z)) * norm.pdf(z),
                -10.0,
                10.0,
                epsabs=1e-12,
            )[0]
            for value in values
        ]
    )
    np.testing.assert_allclose(np.exp(distribution.log_prob(values)), expected, rtol=1e-6, atol=1e-10)


def test_sample_shape_is_applied_once() -> None:
    distribution = LogNormalNegativeBinomial(jnp.array([3.0, 5.0]), jnp.zeros(2), jnp.full(2, 0.5))
    assert distribution.sample(jax.random.key(0), sample_shape=(3, 4)).shape == (3, 4, 2)
