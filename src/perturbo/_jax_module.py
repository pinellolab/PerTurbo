import argparse

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
from numpyro import handlers
from numpyro.infer import (
    MCMC,
    NUTS,
    SVI,
    Predictive,
    TraceMeanField_ELBO,
)
from numpyro.infer.autoguide import AutoNormal, init_to_median


def create_plates(genes, guides=None, n_cells=None, n_guides=None, subsample_size=None, n_genes=None, **kwargs):
    if guides is not None:
        n_cells, n_guides = guides.shape
    if genes is not None:
        n_cells, n_genes = genes.shape
    if guides is not None and genes is not None:
        assert genes.shape[0] == guides.shape[0]

    cell_plate = numpyro.plate("cells", n_cells, dim=-2, subsample_size=subsample_size)
    genes_plate = numpyro.plate("genes", n_genes, dim=-1)

    guide_plate_T = numpyro.plate("guides_T", n_guides, dim=-2)
    guide_plate = numpyro.plate("guides", n_guides, dim=-1)
    plates = (cell_plate, guide_plate, guide_plate_T, genes_plate)
    return plates


# @config_enumerate
def perturbseq_model(
    genes,
    guides=None,
    log2_fc=None,
    gene_mean=None,
    gene_disp=None,
    efficiency=None,
    n_guides=5,
    n_cells=10,
    n_genes=1,
    efficiency_type="mixture",
    efficiency_alpha=3,
    efficiency_beta=1,
    eps=1e-6,
    subsample_size=None,
):
    # infer cell/guide numbers from data
    if guides is not None:
        n_cells, n_guides = guides.shape
    if genes is not None:
        n_cells, n_genes = genes.shape
    if guides is not None and genes is not None:
        assert genes.shape[0] == guides.shape[0]

    # create plates
    cell_plate, guide_plate, guide_plate_T, gene_plate = create_plates(
        genes,
        n_cells=n_cells,
        n_guides=n_guides,
        n_genes=n_genes,
        subsample_size=subsample_size,
    )

    # constants and gene-level params
    with gene_plate:
        baseline_mean = numpyro.sample("mean", dist.LogNormal(0.0, 4.0), obs=gene_mean)
        dispersion = numpyro.sample("dispersion", dist.LogNormal(2.0, 2.0), obs=gene_disp)

    # sample guide efficiency & effect sizes
    # efficiency_alpha = jnp.array(5.0)
    # efficiency_beta = jnp.array(2.0)
    with gene_plate:
        with guide_plate_T:
            guide_efficiency = numpyro.sample(
                "efficiency",
                dist.Beta(efficiency_alpha, efficiency_beta),
                obs=efficiency,
            )

    prior_inclusion_prob = 0.05
    inclusion_probs = jnp.stack([1 - prior_inclusion_prob, prior_inclusion_prob], axis=-1)

    with gene_plate:
        # log2_fc = numpyro.sample("log2_fold_change", dist.Cauchy(0.0, 0.1), obs=log2_fc)
        # log2_fc = numpyro.sample("log2_fold_change", dist.Cauchy(0.0, 0.1), obs=log2_fc)

        non_effect_scale = 0.1
        effect_scale = 3.0
        effect_scales = jnp.stack([non_effect_scale, effect_scale], axis=-1)
        # inclusion = numpyro.sample("include", dist.Categorical(inclusion_probs))
        # log2_fc = numpyro.sample(
        #     "log2_fold_change", dist.Normal(0.0, effect_scales[inclusion]), obs=log2_fc
        # )
        # inclusion = numpyro.sample(
        #     "include",
        #     ,
        #     infer={"enumerate": "parallel"},
        # )

        log2_fc = numpyro.sample(
            "log2_fold_change",
            dist.MixtureSameFamily(dist.Categorical(inclusion_probs), dist.Normal(0.0, effect_scales)),
            obs=log2_fc,
        )

    # sample gene values for each cell
    with cell_plate:
        if guides is not None:
            guides = numpyro.subsample(guides, event_dim=0)
        if genes is not None:
            genes = numpyro.subsample(genes, event_dim=0)

        with guide_plate:
            guide_prob = jnp.array(1 / n_guides)
            guide_obs = numpyro.sample("guide_obs", dist.Binomial(1, probs=guide_prob), obs=guides)

        if efficiency_type == "mixture":
            # guide_inactivity = (1 - guide_obs * guide_efficiency).prod(
            #     axis=-1, keepdims=True
            # )
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
            # logits = guide_effect + jnp.log(baseline_mean)
            # obs_dist = JaxNegativeBinomialMeanDisp(
            #     mean=jnp.exp(logits), inverse_dispersion=dispersion
            # )
        with gene_plate:
            gene_obs = numpyro.sample("gene_obs", obs_dist, obs=genes)

    return gene_obs


def train(args, kwargs, model, guide, lr=0.01, n_steps=1000):
    # adam_params = {"lr": lr}
    adam = numpyro.optim.Adam(step_size=lr)
    svi = SVI(
        model,
        guide,
        adam,
        loss=TraceMeanField_ELBO(),
    )
    svi_result = svi.run(random.PRNGKey(0), n_steps, args, **kwargs)
    return svi_result


def main(args):
    n_genes = 1  # currently only support single_gene analysis

    n_control = args.n_control
    n_guides = args.n_guides
    n_perturbed = args.n_perturbed
    guides = generate_guides_array(n_control=n_control, n_guides=n_guides, n_perturbed=n_perturbed)

    numpyro.render_model(
        perturbseq_model,
        model_args=(None,),
        render_distributions=True,
        render_params=True,
    )

    rng_key = random.PRNGKey(0)

    # Define your model parameters
    model_params = {
        "guides": guides,
        "log2_fc": jnp.array(args.log2_fc),
        "efficiency": jnp.array(args.efficiency),
        "gene_mean": jnp.array(args.gene_mean),
        "gene_disp": jnp.array(args.gene_disp),
    }

    # Create a predictive model
    predictive = Predictive(perturbseq_model, num_samples=1)

    # Sample from the model
    samples = predictive(rng_key, None, **model_params)

    gene_obs = samples["gene_obs"][0, ...]

    perturbseq_guide = AutoNormal(
        handlers.block(handlers.seed(perturbseq_model, random.PRNGKey(0)), hide=["include"]),
        init_loc_fn=init_to_median,
        create_plates=create_plates,
    )
    svi_result = train(
        gene_obs,
        {"guides": guides, "subsample_size": None},
        perturbseq_model,
        perturbseq_guide,
        n_steps=4000,
        lr=0.01,
    )

    unconstrained_locs = {}
    for k, v in svi_result.params.items():
        param, param_type = k.split("_auto_")
        if param_type == "loc":
            unconstrained_locs[param] = v

    predictive_svi = Predictive(
        perturbseq_guide,
        params=svi_result.params,
        num_samples=10000,
    )

    svi_posterior_samples = predictive_svi(rng_key, None, guides=guides, n_genes=n_genes)

    n_chains = 1
    kernel = NUTS(
        perturbseq_model,
        # dense_mass=[("mean", "dispersion", "log2_fold_change")],
    )

    mcmc = MCMC(kernel, num_samples=10000, num_warmup=1000, num_chains=n_chains)
    mcmc.run(random.PRNGKey(0), gene_obs, **{"guides": guides}, init_params=unconstrained_locs)
    mcmc_posterior_samples = mcmc.get_samples()

    return svi_posterior_samples, mcmc_posterior_samples


def generate_guides_array(n_control, n_guides, n_perturbed):
    guides_control = jnp.zeros((n_control, n_guides))
    guides_perturbed = jnp.eye(n_guides).repeat(n_perturbed // n_guides, axis=0)
    guides = jnp.vstack((guides_control, guides_perturbed))
    return guides


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for JAX/NumPyro PerTurbo model")
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--n_control", default=1000, type=int, help="number of control cells")
    parser.add_argument("--n_perturbed", default=50, type=int, help="number of perturbed cells")
    parser.add_argument("--gene_mean", default=2.0, type=float, help="mean expression level of genes")
    parser.add_argument("--gene_disp", default=10.0, help="dispersion of genes")
    parser.add_argument("--n_guides", default=2, type=int, help="number of guides")
    parser.add_argument("--log2_fc", default=0.0, type=float, help="log2 fold change")
    parser.add_argument("--efficiency", default=0.9, type=float, help="guide efficiency")

    args = parser.parse_args()
    numpyro.set_platform(args.device)
    svi_posterior_samples, mcmc_posterior_samples = main(args)
    for samples in [svi_posterior_samples, mcmc_posterior_samples]:
        for k, v in samples.items():
            print(k, v.shape)
