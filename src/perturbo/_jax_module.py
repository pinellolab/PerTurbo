import argparse
import os
from typing import Literal, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.random import PRNGKey
from numpyro.infer import Predictive
from numpyro.infer.autoguide import AutoNormal, init_to_median

from perturbo._jax_utils import generate_guide_arrays, run_mcmc, run_svi


def _create_plates(
    genes=None,
    guide_obs=None,
    covariates=None,
    n_cells=None,
    n_guides=None,
    guide_target_elements=None,
    subsample_size=None,
    **kwargs,
):
    if guide_obs is not None:
        n_cells, n_guides = guide_obs.shape
    if genes is not None:
        n_cells, _ = genes.shape
    if guide_obs is not None and genes is not None:
        assert genes.shape[0] == guide_obs.shape[0]
    if covariates is None:
        n_covariates = 1
    else:
        n_covariates = covariates.shape[1]
    if guide_target_elements is None:
        n_elements = n_guides
    else:
        _, n_elements = guide_target_elements.shape

    cell_plate = numpyro.plate("cells", n_cells, dim=-2, subsample_size=subsample_size)
    covariate_plate = numpyro.plate("covariates", n_covariates, dim=-2)
    guide_plate = numpyro.plate("guides", n_guides, dim=-1)
    element_plate = numpyro.plate("elements", n_elements, dim=-2)
    plates = (cell_plate, covariate_plate, guide_plate, element_plate)
    return plates


def perturbseq_model(
    genes: Optional[jnp.ndarray] = None,
    guide_obs: Optional[jnp.ndarray] = None,
    covariates: Optional[jnp.ndarray] = None,
    log2_fc: Optional[jnp.ndarray] = None,
    gene_mean: Optional[jnp.ndarray] = None,
    gene_disp: Optional[jnp.ndarray] = None,
    efficiency: Optional[jnp.ndarray] = None,
    size_factor: Optional[jnp.ndarray] = None,
    guide_target_elements: Optional[jnp.ndarray] = None,
    n_guides: Optional[int] = 5,
    n_cells: Optional[int] = 101,
    prior_inclusion_prob: float = 0.05,
    non_effect_scale: float = 0.02,
    effect_scale: float = 1.0,
    likelihood: Literal["NegBin", "Poisson", "PoissonLogNorm"] = "NegBin",
    efficiency_alpha: float = 5.0,
    efficiency_beta: float = 1.0,
    fit_efficiency: bool = False,
    eps: float = 1e-6,
    subsample_size: Optional[int] = None,
) -> jnp.ndarray:
    if guide_obs is not None:
        n_cells, n_guides = guide_obs.shape
    if genes is not None:
        n_cells = genes.shape[0]
    if guide_obs is not None and genes is not None:
        assert genes.shape[0] == guide_obs.shape[0]
    if covariates is None:
        covariates = jnp.zeros((n_cells, 1))
    # create plates
    cell_plate, covariates_plate, guide_plate, element_plate = _create_plates(
        genes,
        covariates=covariates,
        n_cells=n_cells,
        n_guides=n_guides,
        guide_target_elements=guide_target_elements,
        subsample_size=subsample_size,
    )

    if guide_target_elements is None:
        guide_target_elements = jnp.eye(n_guides)

    # sample gene-level params
    baseline_mean = numpyro.sample("mean", dist.LogNormal(0.0, 4.0), obs=gene_mean)
    dispersion = numpyro.sample("dispersion", dist.LogNormal(2.0, 2.0), obs=gene_disp)

    # sample guide efficiency
    with guide_plate:
        if fit_efficiency:
            guide_efficiency = numpyro.sample(
                "efficiency", dist.Beta(efficiency_alpha, efficiency_beta), obs=efficiency
            )
        elif efficiency is None:
            guide_efficiency = jnp.ones((n_guides,))
        else:
            guide_efficiency = efficiency
        # alpha = numpyro.sample("efficiency", dist.Uniform(), obs=efficiency)
        # guide_efficiency = jnp.expand_dims(guide_efficiency, axis=-1)
    # guide_efficiency = jnp.ones((n_guides, 1))

    # sample element effect sizes
    with element_plate:
        inclusion_probs = jnp.array([1.0 - prior_inclusion_prob, prior_inclusion_prob])
        effect_scales = jnp.array([non_effect_scale, effect_scale])
        mix_dist = dist.Categorical(inclusion_probs)
        effect_dist = dist.Normal(0.0, effect_scales)
        log2_fc = numpyro.sample("log2_fold_change", dist.MixtureSameFamily(mix_dist, effect_dist), obs=log2_fc)

    # element_by_guide = jnp.ones((1, n_guides))
    # guide_target_elements = jnp.ones((n_guides, 1))

    with covariates_plate:
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
        # (n_cell x n_guides) * (n_guides)
        cell_guide_efficiency = guide_obs * guide_efficiency
        # (n_cell x n_guides) @ (n_guides x n_element)
        cell_element_efficiency = cell_guide_efficiency @ guide_target_elements
        # (n_cell x n_guides) @ (n_guides x n_element)
        guide_effect = jnp.exp(cell_element_efficiency @ log2_fc * jnp.log(2))
        mean = guide_effect * baseline_mean * jnp.exp(covariate_effect)

        if size_factor is not None:
            size_factor = numpyro.subsample(size_factor, event_dim=0)
            mean *= size_factor

        # scaled_guide_effect = 1 + (guide_obs @ guide_efficiency) * jnp.expm1(guide_effect)
        # mean = scaled_guide_effect * baseline_mean * jnp.exp(covariate_effect)
        assert likelihood == "NegBin", "Only NegBin likelihood supported"
        obs_dist = dist.NegativeBinomial2(mean + 1e-4, dispersion)
        gene_obs = numpyro.sample("gene_obs", obs_dist, obs=genes)

    return gene_obs


def perturbseq_guide_autonormal(init_loc_fn=init_to_median):
    return AutoNormal(
        perturbseq_model,
        init_loc_fn=init_loc_fn,
        create_plates=_create_plates,
    )


def main(args):
    n_genes = 1  # currently only support single_gene analysis
    n_control = args.n_control
    n_guides = args.n_guides
    n_perturbed = args.n_perturbed
    guide_obs = generate_guide_arrays(n_control=n_control, n_guides_per_element=n_guides, n_cells_per_guide=n_perturbed)

    print("Simulating data with parameters:")
    print("# guides:", n_guides)
    print("# control cells:", n_control)
    print("# perturbed cells:", n_perturbed)
    print(f"gene mean, dispersion: {args.gene_mean:0.2f}, {args.gene_disp:0.2f}")
    print("log_2 fold change:", args.log2_fc)
    print("guide efficiency:", args.efficiency)

    # Define your model parameters
    model_params = {
        "guide_obs": guide_obs,
        "log2_fc": jnp.array(args.log2_fc).reshape(1, -1),
        "efficiency": jnp.array(args.efficiency),
        "gene_mean": jnp.array(args.gene_mean),
        "gene_disp": jnp.array(args.gene_disp),
    }

    # Sample data from the prior with frozen values
    predictive = Predictive(perturbseq_model, num_samples=1)
    samples = predictive(PRNGKey(args.random_seed), None, **model_params)
    gene_obs = samples["gene_obs"][0, ...]

    # set model args, kwargs for inference
    model_args = (gene_obs,)
    model_kwargs = {"guide_obs": guide_obs}

    # construct AutoNormal guide for model
    perturbseq_guide = perturbseq_guide_autonormal()
    svi_result = run_svi(model_args, model_kwargs, perturbseq_model, perturbseq_guide, random_seed=args.random_seed)

    unconstrained_locs = get_unconstrained_locs(svi_result)

    # Create output directories if they don't exist
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save SVI results
    predictive_svi = Predictive(perturbseq_guide, params=svi_result.params, num_samples=args.num_samples)
    svi_posterior_samples = predictive_svi(PRNGKey(args.random_seed), None, guide_obs=guide_obs, n_genes=n_genes)
    jnp.savez(os.path.join(output_dir, "svi"), **svi_posterior_samples)

    # Save MCMC results
    mcmc = run_mcmc(
        model_args,
        model_kwargs,
        perturbseq_model,
        unconstrained_locs,
        num_samples=args.num_samples,
        random_seed=args.random_seed,
    )
    mcmc_posterior_samples = mcmc.get_samples()
    jnp.savez(os.path.join(output_dir, "mcmc"), **mcmc_posterior_samples)


def get_unconstrained_locs(svi_result):
    unconstrained_locs = {}
    for k, v in svi_result.params.items():
        param, param_type = k.split("_auto_")
        if param_type == "loc":
            unconstrained_locs[param] = v
    return unconstrained_locs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for JAX/NumPyro PerTurbo model")
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--n_control", default=1000, type=int, help="number of control cells")
    parser.add_argument("--n_perturbed", default=50, type=int, help="number of perturbed cells")
    parser.add_argument("--gene_mean", default=2.0, type=float, help="mean expression level of genes")
    parser.add_argument("--gene_disp", default=10.0, type=float, help="dispersion of genes")
    parser.add_argument("--n_guides", default=2, type=int, help="number of guides")
    parser.add_argument("--log2_fc", default=0.0, type=float, help="log2 fold change")
    parser.add_argument("--num_samples", default=1000, type=int, help="Number of posterior samples from MCMC/SVI")
    parser.add_argument("--random_seed", default=0, type=int, help="Random seed for Jax")
    parser.add_argument(
        "--efficiency", default=[0.5, 0.1], type=float, nargs="*", help="Guide efficiency (list of values)"
    )
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory for posterior samples")

    args = parser.parse_args()
    numpyro.set_platform(args.device)
    main(args)
