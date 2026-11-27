"""Direct model parity and latent scope for simulation grid draws."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from numpyro.infer import Predictive

import perturbo.model as models
from perturbo.simulation.predictive import (
    sample_predictive_counts,
    simulator_provenance,
    validate_simulation_output_root,
)


@pytest.mark.parametrize("likelihood", ["nb", "lnnb", "mixture_nb", "censored_nb"])
def test_exact_direct_model_draw(likelihood):
    sites = dict(beta_0=jnp.array([0.5, 1.0]), theta=jnp.array([2.0, 4.0]), beta=jnp.array([[0.0, 0.0], [-0.5, 0.2]]))
    if likelihood == "lnnb":
        sites["noise_scale"] = jnp.array([0.2, 0.5])
    if likelihood == "mixture_nb":
        sites.update(
            pi_outlier=jnp.array([0.1, 0.4]),
            theta_outlier=jnp.array([0.5, 1.0]),
            outlier_mean_shift=jnp.array([1.0, 2.0]),
        )
    design = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    offsets = jnp.array([[-0.5], [0.0], [0.5], [1.0]])
    key = jax.random.PRNGKey(42)
    counts, used = sample_predictive_counts(
        sites, design, key, size_factors=offsets, likelihood=likelihood, chunk_rows=None
    )
    direct = Predictive(numpyro.handlers.condition(models.BaseModel, data=sites), num_samples=1)(
        key, None, design, size_factors=offsets, num_cells=4, num_genes=2, num_perts=2, likelihood=likelihood
    )
    np.testing.assert_array_equal(counts, direct["obs"][0])
    np.testing.assert_array_equal(used, offsets)
    assert counts.dtype == np.int32


def test_covariates_and_aligned_factor_scores_enter_actual_observation_law(monkeypatch):
    sites = dict(
        beta_0=np.array([1.0, 2.0]),
        theta=np.array([2.0, 4.0]),
        beta=np.zeros((1, 2)),
        covariate_coef=np.array([[0.4, -0.6]]),
        factor_loadings=np.array([[[0.2, 0.3]]]),
        factor_scores=np.arange(5).reshape(1, 5, 1),
    )
    cov = np.arange(5).reshape(5, 1).astype(np.float32)
    original, means = models._sample_observations, []

    def spy(**kwargs):
        means.append(np.asarray(kwargs["logits"] + jnp.log(kwargs["theta"])))
        return original(**kwargs)

    monkeypatch.setattr(models, "_sample_observations", spy)
    sample_predictive_counts(
        sites,
        np.zeros((5, 1)),
        jax.random.PRNGKey(0),
        size_factors=np.zeros((5, 1)),
        covariates=cov,
        num_factors=1,
        chunk_rows=2,
    )
    expected = np.array([1.0, 2.0]) + cov @ np.array([[0.4, -0.6]]) + cov @ np.array([[0.2, 0.3]])
    np.testing.assert_allclose(np.concatenate(means), expected, atol=2e-6)


def test_chunking_does_not_redraw_global_guide_effects(monkeypatch):
    sites = dict(beta_0=np.zeros(1), theta=np.ones(1), beta=np.zeros((1, 1)), guide_random_effect_tau=np.ones(1))
    original, means = models._sample_observations, []

    def spy(**kwargs):
        means.append(np.asarray(kwargs["logits"] + jnp.log(kwargs["theta"])))
        return original(**kwargs)

    monkeypatch.setattr(models, "_sample_observations", spy)
    sample_predictive_counts(
        sites,
        np.ones((7, 1)),
        jax.random.PRNGKey(1),
        size_factors=np.zeros((7, 1)),
        guide_matrix=np.ones((7, 1)),
        guide_to_element=np.ones((1, 1)),
        guide_random_effects=True,
        chunk_rows=3,
    )
    values = np.concatenate(means)
    np.testing.assert_array_equal(values, np.repeat(values[:1], 7, axis=0))
    assert abs(values[0, 0]) > 1e-5


@pytest.mark.parametrize("guide_aware", [False, True])
def test_perturbation_specific_dispersion_uses_model_site(guide_aware, monkeypatch):
    sites = dict(beta_0=np.zeros(2), theta=np.array([2.0, 4.0]), beta=np.zeros((2, 2)))
    name = "guide_dispersion_excess_inverse" if guide_aware else "dispersion_excess_inverse"
    sites[name] = np.array([[0.0, 0.0], [0.5, 0.25]])
    original, theta = models._sample_observations, []

    def spy(**kwargs):
        theta.append(np.asarray(kwargs["theta"]))
        return original(**kwargs)

    monkeypatch.setattr(models, "_sample_observations", spy)
    design = np.eye(2)
    extra = dict(guide_matrix=design, guide_to_element=design) if guide_aware else {}
    sample_predictive_counts(
        sites, design, jax.random.PRNGKey(0), size_factors=np.zeros((2, 1)), fit_perturbation_dispersion=True, **extra
    )
    np.testing.assert_allclose(theta[-1], [[2.0, 4.0], [1.0, 2.0]])


@pytest.mark.parametrize("bad", [dict(chunk_rows=0), dict(num_factors=1), dict(covariates=np.ones((3, 1)))])
def test_invalid_predictive_configuration_fails_early(bad):
    with pytest.raises(ValueError):
        sample_predictive_counts(
            dict(beta_0=np.zeros(2), theta=np.ones(2), beta=np.zeros((1, 2))),
            np.ones((3, 1)),
            jax.random.PRNGKey(0),
            **bad,
        )


def test_misspelled_or_inactive_conditioned_sites_are_rejected():
    with pytest.raises(ValueError, match="unused"):
        sample_predictive_counts(
            dict(beta_0=np.zeros(1), theta=np.ones(1), beta=np.zeros((1, 1)), factor_scores=np.zeros((1, 3, 1))),
            np.ones((3, 1)),
            jax.random.PRNGKey(0),
        )


def test_invalid_element_index_cannot_silently_wrap_to_last_effect():
    with pytest.raises(ValueError, match="element axis"):
        sample_predictive_counts(
            dict(beta_0=np.zeros(1), theta=np.ones(1), beta=np.ones((1, 1))),
            np.array([0, -1], np.int32),
            jax.random.PRNGKey(0),
        )


def test_grid_resume_cannot_relabel_historical_counts(tmp_path):
    validate_simulation_output_root(tmp_path)
    design = tmp_path / "design.json"
    design.write_text(json.dumps(dict(grid=[])))
    with pytest.raises(ValueError, match="choose a new output root"):
        validate_simulation_output_root(tmp_path)
    design.write_text(json.dumps(dict(simulation_generator=simulator_provenance(), grid=[])))
    validate_simulation_output_root(tmp_path)
    design.unlink()
    (tmp_path / "truth.parquet").write_bytes(b"old truth")
    with pytest.raises(ValueError, match="lack a generator receipt"):
        validate_simulation_output_root(tmp_path)


def test_relative_guides_use_conditioned_efficiencies_in_the_observation_model():
    guides = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    parents = jnp.ones((2, 1))
    design = jnp.clip(guides @ parents, 0, 1)
    sites = {
        "beta_0": jnp.array([2.0, 1.0]),
        "theta": jnp.array([7.0, 5.0]),
        "beta": jnp.array([[-1.0, 0.5]]),
        "guide_relative_efficiency": jnp.array([[0.25, 0.25], [0.9, 0.9]]),
    }
    offsets = jnp.zeros((4, 1))
    key = jax.random.PRNGKey(11)
    geometry = dict(guide_matrix=guides, guide_to_element=parents, guide_effect_strategy="relative")
    actual, _ = sample_predictive_counts(sites, design, key, size_factors=offsets, **geometry)
    direct = Predictive(numpyro.handlers.condition(models.GuideSharedEffectModel, data=sites), num_samples=1)(
        key, None, design, num_cells=4, num_genes=2, num_perts=1, size_factors=offsets, **geometry
    )
    np.testing.assert_array_equal(actual, direct["obs"][0])
    np.testing.assert_allclose(direct["guide_effect"][0], parents @ sites["beta"] * sites["guide_relative_efficiency"])
