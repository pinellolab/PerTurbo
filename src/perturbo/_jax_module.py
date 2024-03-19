import argparse
import os

import jax.numpy as jnp
from networkx import efficiency
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Predictive, TraceMeanField_ELBO
from numpyro.infer.autoguide import AutoNormal, init_to_median


def create_plates(genes, guide_obs=None, n_cells=None, n_guides=None, subsample_size=None, n_genes=None, **kwargs):
    if guide_obs is not None:
        n_cells, n_guides = guide_obs.shape
    if genes is not None:
        n_cells, n_genes = genes.shape
    if guide_obs is not None and genes is not None:
        assert genes.shape[0] == guide_obs.shape[0]

    cell_plate = numpyro.plate("cells", n_cells, dim=-2, subsample_size=subsample_size)
    genes_plate = numpyro.plate("genes", n_genes, dim=-1)
    guide_plate_T = numpyro.plate("guides_T", n_guides, dim=-2)
    guide_plate = numpyro.plate("guides", n_guides, dim=-1)
    plates = (cell_plate, guide_plate, guide_plate_T, genes_plate)
    return plates


# @config_enumerate
def perturbseq_model(
    genes,
    guide_obs=None,
    log2_fc=None,
    gene_mean=None,
    gene_disp=None,
    efficiency=None,
    n_guides=5,
    n_cells=101,
    n_genes=1,
    prior_inclusion_prob=0.05,
    non_effect_scale=0.05,
    effect_scale=3.0,
    efficiency_type="mixture",
    efficiency_alpha=3.0,
    efficiency_beta=1.0,
    eps=1e-6,
    subsample_size=None,
):
    # infer cell/guide numbers from data
    if guide_obs is not None:
        n_cells, n_guides = guide_obs.shape
    if genes is not None:
        n_cells, n_genes = genes.shape
    if guide_obs is not None and genes is not None:
        assert genes.shape[0] == guide_obs.shape[0]

    # create plates
    cell_plate, guide_plate, guide_plate_T, gene_plate = create_plates(
        genes, n_cells=n_cells, n_guides=n_guides, n_genes=n_genes, subsample_size=subsample_size
    )

    # constants and gene-level params
    with gene_plate:
        baseline_mean = numpyro.sample("mean", dist.LogNormal(0.0, 4.0), obs=gene_mean)
        dispersion = numpyro.sample("dispersion", dist.LogNormal(2.0, 2.0), obs=gene_disp)

    # sample guide efficiency & effect sizes
    with gene_plate, guide_plate_T:
        guide_efficiency = numpyro.sample("efficiency", dist.Beta(efficiency_alpha, efficiency_beta), obs=efficiency)

    with gene_plate:
        log2_fc = sample_effect_sizes(non_effect_scale, effect_scale, prior_inclusion_prob, log2_fc)

    # sample gene values for each cell
    with cell_plate:
        if guide_obs is not None:
            guide_obs = numpyro.subsample(guide_obs, event_dim=0)
        if genes is not None:
            genes = numpyro.subsample(genes, event_dim=0)

        with guide_plate:
            guide_prob = jnp.array(1 / n_guides)
            guide_obs = numpyro.sample("guide_obs", dist.Binomial(1, probs=guide_prob), obs=guide_obs)

        if efficiency_type == "mixture":
            # guide_inactivity = (1 - guide_obs * guide_efficiency).prod()
            guide_inactivity_logit = jnp.log(
                1 - jnp.tile(jnp.expand_dims(guide_obs, axis=-1), (1, 1, n_genes)) * guide_efficiency
            ).sum(axis=-2)

            # guide_activity_logit = jnp.log(1 - jnp.exp(guide_inactivity_logit))
            guide_activity_logit = jnp.log1p(-jnp.exp(guide_inactivity_logit) + eps)

            # mix_probs = jnp.stack([guide_inactivity, 1 - guide_inactivity], axis=-1)
            mix_logits = jnp.stack([guide_inactivity_logit, guide_activity_logit], axis=-1)
            mix_dist = dist.Categorical(logits=mix_logits)
            guide_effect = log2_fc * jnp.log(2)
            base_logit = jnp.log(baseline_mean) - jnp.log(dispersion)
            logits = jnp.stack([base_logit, guide_effect + base_logit], axis=-1)
            component_dist = dist.NegativeBinomialLogits(logits=logits, total_count=dispersion)
            obs_dist = dist.MixtureSameFamily(mix_dist, component_dist)

        elif efficiency_type == "scale":
            guide_effect = guide_obs @ (log2_fc * guide_efficiency) * jnp.log(2)
            logits = guide_effect + jnp.log(baseline_mean) - jnp.log(dispersion)
            obs_dist = dist.NegativeBinomialLogits(logits=logits, total_count=dispersion)
        with gene_plate:
            gene_obs = numpyro.sample("gene_obs", obs_dist, obs=genes)

    return gene_obs


def sample_effect_sizes(non_effect_scale, effect_scale, prior_inclusion_prob, log2_fc=None):
    inclusion_probs = jnp.stack([1 - prior_inclusion_prob, prior_inclusion_prob], axis=-1)
    effect_scales = jnp.stack([non_effect_scale, effect_scale], axis=-1)
    mix_dist = dist.Categorical(inclusion_probs)
    effect_dist = dist.Normal(0.0, effect_scales)
    log2_fc = numpyro.sample("log2_fold_change", dist.MixtureSameFamily(mix_dist, effect_dist), obs=log2_fc)
    return log2_fc


def run_svi(args, kwargs, model, guide, lr=0.01, n_steps=1000):
    # adam_params = {"lr": lr}
    adam = numpyro.optim.Adam(step_size=lr)
    svi = SVI(model, guide, adam, loss=TraceMeanField_ELBO())
    svi_result = svi.run(random.PRNGKey(0), n_steps, args, **kwargs)
    return svi_result


def render_model():
    numpyro.render_model(perturbseq_model, model_args=(None,), render_distributions=True, render_params=True)


def run_mcmc(guide_obs, gene_obs, unconstrained_locs=None, dense_mass=False, num_samples=1000):
    n_chains = 1
    kernel = NUTS(perturbseq_model, dense_mass=dense_mass)
    mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=1000, num_chains=n_chains)
    mcmc.run(random.PRNGKey(0), gene_obs, guide_obs=guide_obs, init_params=unconstrained_locs)
    return mcmc


def generate_guides_array(n_control, n_guides, n_perturbed):
    guides_control = jnp.zeros((n_control, n_guides))
    guides_perturbed = jnp.eye(n_guides).repeat(n_perturbed // n_guides, axis=0)
    guide_obs = jnp.vstack((guides_control, guides_perturbed))
    return guide_obs


def main(args):
    n_genes = 1  # currently only support single_gene analysis
    prng_key = random.PRNGKey(args.random_seed)
    n_control = args.n_control
    n_guides = args.n_guides
    n_perturbed = args.n_perturbed
    guide_obs = generate_guides_array(n_control=n_control, n_guides=n_guides, n_perturbed=n_perturbed)

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
        "log2_fc": jnp.array(args.log2_fc),
        "efficiency": jnp.array(args.efficiency).reshape(-1, 1),
        "gene_mean": jnp.array(args.gene_mean),
        "gene_disp": jnp.array(args.gene_disp),
    }

    # Sample data from the prior with frozen values
    predictive = Predictive(perturbseq_model, num_samples=1)
    samples = predictive(prng_key, None, **model_params)
    gene_obs = samples["gene_obs"][0, ...]

    # construct AutoNormal guide for model
    perturbseq_guide = AutoNormal(
        handlers.block(handlers.seed(perturbseq_model, prng_key), hide=["include"]),
        init_loc_fn=init_to_median,
        create_plates=create_plates,
    )
    svi_result = run_svi(gene_obs, {"guide_obs": guide_obs}, perturbseq_model, perturbseq_guide, n_steps=4000, lr=0.01)

    unconstrained_locs = {}
    for k, v in svi_result.params.items():
        param, param_type = k.split("_auto_")
        if param_type == "loc":
            unconstrained_locs[param] = v

    # Create output directories if they don't exist
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save SVI results
    predictive_svi = Predictive(perturbseq_guide, params=svi_result.params, num_samples=args.num_samples)
    svi_posterior_samples = predictive_svi(prng_key, None, guide_obs=guide_obs, n_genes=n_genes)
    jnp.savez(os.path.join(output_dir, "svi"), **svi_posterior_samples)

    # Save MCMC results
    mcmc = run_mcmc(guide_obs, gene_obs, unconstrained_locs, num_samples=args.num_samples)
    mcmc_posterior_samples = mcmc.get_samples()
    jnp.savez(os.path.join(output_dir, "mcmc"), **mcmc_posterior_samples)


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
