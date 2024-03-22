from typing import Literal, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoNormal, init_to_median
from tensorflow_probability.substrates.jax import distributions as tfd


def _create_plates(
    genes,
    guide_obs=None,
    covariates=None,
    n_cells=None,
    n_guides=None,
    subsample_size=None,
    guide_targets=None,
    n_genes=None,
    **kwargs,
):
    if guide_obs is not None:
        n_cells, n_guides = guide_obs.shape
    if genes is not None:
        n_cells, n_genes = genes.shape
    if guide_obs is not None and genes is not None:
        assert genes.shape[0] == guide_obs.shape[0]
    if covariates is None:
        n_covariates = 1
    else:
        n_covariates = covariates.shape[1]
    if guide_targets is None:
        n_elements = 1
    else:
        n_elements = guide_targets.shape

    cell_plate = numpyro.plate("cells", n_cells, dim=-2, subsample_size=subsample_size)
    covariate_plate = numpyro.plate("covariates", n_covariates, dim=-2)
    gene_plate = numpyro.plate("genes", n_genes, dim=-1)
    guide_plate_T = numpyro.plate("guides_T", n_guides, dim=-2)
    element_plate = numpyro.plate("elements", n_elements, dim=-2)
    guide_plate = numpyro.plate("guides", n_guides, dim=-1)
    plates = (cell_plate, covariate_plate, guide_plate, guide_plate_T, element_plate, gene_plate)
    return plates


def perturbseq_model_turbo(
    genes: Optional[jnp.ndarray] = None,
    guide_obs: Optional[jnp.ndarray] = None,
    covariates: Optional[jnp.ndarray] = None,
    guide_targets: Optional[jnp.ndarray] = None,
    log2_fc: Optional[jnp.ndarray] = None,
    gene_mean: Optional[jnp.ndarray] = None,
    gene_disp: Optional[jnp.ndarray] = None,
    efficiency: Optional[jnp.ndarray] = None,
    n_guides: Optional[int] = 5,
    n_cells: Optional[int] = 101,
    n_genes: Optional[int] = 1,
    prior_inclusion_prob: float = 0.05,
    non_effect_scale: float = 0.05,
    effect_scale: float = 3.0,
    likelihood: Literal["NegBin", "Poisson", "PoissonLogNorm"] = "NegBin",
    efficiency_alpha: float = 3.0,
    efficiency_beta: float = 1.0,
    eps: float = 1e-6,
    subsample_size: Optional[int] = None,
) -> jnp.ndarray:
    if guide_obs is not None:
        n_cells, n_guides = guide_obs.shape
    if genes is not None:
        n_cells, n_genes = genes.shape
    if guide_obs is not None and genes is not None:
        assert genes.shape[0] == guide_obs.shape[0]
    if covariates is None:
        covariates = jnp.zeros((n_cells, 1))
    if guide_targets is None:
        guide_targets = jnp.ones((n_guides, 1))

    # create plates
    cell_plate, covariates_plate, guide_plate, guide_plate_T, element_plate, gene_plate = _create_plates(
        genes,
        covariates=covariates,
        guide_targets=guide_targets,
        n_cells=n_cells,
        n_guides=n_guides,
        n_genes=n_genes,
        subsample_size=subsample_size,
    )

    # sample gene-level params
    with gene_plate:
        baseline_mean = numpyro.sample("mean", dist.LogNormal(0.0, 4.0), obs=gene_mean)
        dispersion = numpyro.sample("dispersion", dist.LogNormal(2.0, 2.0), obs=gene_disp)

    # sample guide efficiency
    with guide_plate_T, gene_plate:
        guide_efficiency = numpyro.sample("efficiency", dist.Beta(efficiency_alpha, efficiency_beta), obs=efficiency)

    # sample element effect sizes
    with element_plate, gene_plate:
        inclusion_probs = jnp.stack([1.0 - prior_inclusion_prob, prior_inclusion_prob], axis=-1)
        effect_scales = jnp.stack([non_effect_scale, effect_scale], axis=-1)
        mix_dist = dist.Categorical(inclusion_probs)
        effect_dist = dist.Normal(0.0, effect_scales)
        log2_fc = numpyro.sample("log2_fold_change", dist.MixtureSameFamily(mix_dist, effect_dist), obs=log2_fc)

    with covariates_plate, gene_plate:
        covariate_weights = numpyro.sample("covariate_weights", dist.Normal(0.0, 1.0))

    # sample gene values for each cell
    with cell_plate:
        if guide_obs is not None:
            guide_obs = numpyro.subsample(guide_obs, event_dim=0)
        if genes is not None:
            genes = numpyro.subsample(genes, event_dim=0)
        covariates = numpyro.subsample(covariates, event_dim=0)
        covariate_effect = covariates @ covariate_weights

        with guide_plate:
            guide_prob = jnp.array(1 / n_guides)
            guide_obs = numpyro.sample("guide_obs", dist.Binomial(1, probs=guide_prob), obs=guide_obs)

        guide_effect = guide_obs @ (log2_fc * guide_efficiency) * jnp.log(2)
        guide_effect += covariate_effect

        if likelihood == "Poisson":
            obs_dist = dist.Poisson(jnp.exp(guide_effect) * baseline_mean)
        elif likelihood == "NegBin":
            logits = guide_effect + jnp.log(baseline_mean) - jnp.log(dispersion)
            obs_dist = dist.NegativeBinomialLogits(logits=logits, total_count=dispersion)
        elif likelihood == "PoissonLogNorm":
            obs_dist = tfd.PoissonLogNormalQuadratureCompound(
                loc=guide_effect + jnp.log(baseline_mean),
                scale=1 / dispersion,
                quadrature_fn=tfd.quadrature_scheme_lognormal_gauss_hermite,
            )

        with gene_plate:
            gene_obs = numpyro.sample("gene_obs", obs_dist, obs=genes)

    return gene_obs


def perturbseq_guide_autonormal_turbo(init_loc_fn=init_to_median):
    return AutoNormal(
        perturbseq_model_turbo,
        init_loc_fn=init_loc_fn,
        create_plates=_create_plates,
    )
