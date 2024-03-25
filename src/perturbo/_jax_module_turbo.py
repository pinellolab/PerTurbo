from typing import Literal, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoDelta, AutoGuideList, AutoNormal, init_to_median
from tensorflow_probability.substrates.jax import distributions as tfd


def _create_plates(
    genes,
    n_factors=None,
    n_cell_factors=None,
    guide_obs=None,
    covariates=None,
    n_cells=None,
    n_guides=None,
    subsample_size=None,
    guide_target_elements=None,
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
    if guide_target_elements is None:
        n_elements = n_guides
    else:
        n_elements = guide_target_elements.shape[1]
    cell_plate = numpyro.plate("cells", n_cells, dim=-2, subsample_size=subsample_size)
    covariate_plate = numpyro.plate("covariates", n_covariates, dim=-2)
    gene_plate = numpyro.plate("genes", n_genes, dim=-1)
    guide_plate_T = numpyro.plate("guides_T", n_guides, dim=-2)
    element_plate = numpyro.plate("elements", n_elements, dim=-2)
    guide_plate = numpyro.plate("guides", n_guides, dim=-1)
    factor_plate = numpyro.plate("factors", n_factors, dim=-1)
    loading_plate = numpyro.plate("factors_T", n_factors, dim=-2)
    cell_factor_plate = numpyro.plate("cell_factors", n_cell_factors, dim=-1)
    cell_loading_plate = numpyro.plate("cell_factors_T", n_cell_factors, dim=-2)

    plates = (
        cell_plate,
        covariate_plate,
        guide_plate,
        guide_plate_T,
        element_plate,
        gene_plate,
        factor_plate,
        loading_plate,
        cell_factor_plate,
        cell_loading_plate,
    )
    return plates


def perturbseq_model_turbo(
    genes: Optional[jnp.ndarray] = None,
    guide_obs: Optional[jnp.ndarray] = None,
    covariates: Optional[jnp.ndarray] = None,
    guide_target_elements: Optional[jnp.ndarray] = None,
    element_target_genes: Optional[jnp.ndarray] = None,
    log2_fc: Optional[jnp.ndarray] = None,
    gene_mean: Optional[jnp.ndarray] = None,
    gene_disp: Optional[jnp.ndarray] = None,
    efficiency: Optional[jnp.ndarray] = None,
    n_guides: Optional[int] = 5,
    n_cells: Optional[int] = 101,
    n_genes: Optional[int] = 1,
    n_factors: Optional[int] = 1000,
    n_cell_factors: Optional[int] = 5,
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
    if guide_target_elements is None:
        guide_target_elements = jnp.eye(n_guides)

    # create plates
    (
        cell_plate,
        covariates_plate,
        guide_plate,
        guide_plate_T,
        element_plate,
        gene_plate,
        factor_plate,
        loading_plate,
        cell_factor_plate,
        cell_loading_plate,
    ) = _create_plates(
        genes,
        covariates=covariates,
        guide_target_elements=guide_target_elements,
        n_factors=n_factors,
        n_cell_factors=n_cell_factors,
        n_cells=n_cells,
        n_guides=n_guides,
        n_genes=n_genes,
        subsample_size=subsample_size,
    )

    # sample gene-level params
    with gene_plate:
        baseline_mean = numpyro.sample("mean", dist.LogNormal(0.0, 4.0), obs=gene_mean)
        dispersion = numpyro.sample("dispersion", dist.LogNormal(0.0, 3.0), obs=gene_disp)

    # sample guide efficiency
    # with guide_plate_T, gene_plate:
    with guide_plate_T:
        guide_efficiency = numpyro.sample("efficiency", dist.Beta(efficiency_alpha, efficiency_beta), obs=efficiency)
        guide_efficiency = 1

    # sample element effect sizes (dense from spike and slab)
    inclusion_probs = jnp.stack([1.0 - prior_inclusion_prob, prior_inclusion_prob], axis=-1)
    effect_scales = jnp.stack([non_effect_scale, effect_scale], axis=-1)
    mix_dist = dist.Categorical(inclusion_probs)
    effect_dist = dist.Normal(0.0, effect_scales)
    spike_and_slab_dist = dist.MixtureSameFamily(mix_dist, effect_dist)
    with element_plate, gene_plate:
        log2_fc = numpyro.sample("log2_fold_change", spike_and_slab_dist, obs=log2_fc)

    ## factor model
    # sample cis element effect sizes (sparse)
    # with element_plate:
    #     log2_fc = numpyro.sample("log2_fold_change", dist.Normal(0, effect_scale), obs=log2_fc) * element_target_genes

    # with element_plate, factor_plate:
    #     factors = numpyro.sample("factor", dist.Cauchy(0, 0.01))

    # with loading_plate, gene_plate:
    #     loadings = numpyro.sample("loading", dist.Cauchy(0, 0.01))
    # # log2_fc = numpyro.deterministic("log2_fold_change", factors @ loadings)
    # log2_fc += factors @ loadings

    with covariates_plate, gene_plate:
        covariate_weights = numpyro.sample("covariate_weight", dist.Normal(0.0, 1.0))

    with cell_loading_plate, gene_plate:
        cell_loadings = numpyro.sample("cell_loading", dist.Normal(0, 0.1))
        # log2_fc = numpyro.deterministic("log2_fold_change", factors @ loadings)

    # sample gene values for each cell
    with cell_plate:
        with cell_factor_plate:
            cell_factors = numpyro.sample("cell_factor", dist.Normal(0, 0.1))
        baseline_mean *= jnp.exp(cell_factors @ cell_loadings)
        if guide_obs is not None:
            guide_obs = numpyro.subsample(guide_obs, event_dim=0)
        if genes is not None:
            genes = numpyro.subsample(genes, event_dim=0)
        covariates = numpyro.subsample(covariates, event_dim=0)
        covariate_effect = covariates @ covariate_weights

        with guide_plate:
            guide_prob = jnp.array(1 / n_guides)
            guide_obs = numpyro.sample("guide_obs", dist.Binomial(1, probs=guide_prob), obs=guide_obs)

        # (n_cells x n_guide) @ (n_guide x n_element @ n_element x n_gene) = n_cells x n_genes
        guide_effect = guide_obs @ (guide_target_elements @ log2_fc * guide_efficiency) * jnp.log(2)
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
    guide = AutoNormal(
        perturbseq_model_turbo,
        init_loc_fn=init_loc_fn,
        create_plates=_create_plates,
    )
    return guide
    guide = AutoGuideList(
        perturbseq_model_turbo,
        init_loc_fn=init_loc_fn,
        create_plates=_create_plates,
    )
    guide.append(
        AutoNormal(
            numpyro.handlers.block(
                numpyro.handlers.seed(perturbseq_model_turbo, rng_seed=0), hide=["factor", "loading"]
            )
        )
    )
    guide.append(
        AutoDelta(
            numpyro.handlers.block(
                numpyro.handlers.seed(perturbseq_model_turbo, rng_seed=1), expose=["factor", "loading"]
            )
        )
    )
    return guide
