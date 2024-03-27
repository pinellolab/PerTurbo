from typing import Literal, Optional

import jax.numpy as jnp
import numpy as np
import numpyro
import optax
from jax.random import PRNGKey
from mudata import MuData
from numpyro.infer import MCMC, NUTS, SVI, TraceMeanField_ELBO


def render_perturbseq_model(model):
    gene_obs = jnp.zeros((1, 1))
    guide_obs = jnp.zeros((1,))
    return numpyro.render_model(
        model,
        model_args=(gene_obs,),
        model_kwargs={"guide_obs": guide_obs},
        render_distributions=True,
        render_params=True,
    )


def run_mcmc(args, kwargs, model, unconstrained_locs=None, dense_mass=False, num_samples=1000, random_seed=0):
    n_chains = 1
    kernel = NUTS(model, dense_mass=dense_mass)
    mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=1000, num_chains=n_chains)
    mcmc.run(PRNGKey(random_seed), *args, **kwargs, init_params=unconstrained_locs)
    return mcmc


def run_svi(args, kwargs, model, guide, lr=0.03, n_steps=5000, random_seed=0, decay_rate=0.03):
    # decay lr exponentially to a final value of lr * decay_rate, starting at n_steps // 2
    learning_rate = optax.exponential_decay(lr, n_steps // 2, decay_rate, transition_begin=n_steps // 2, staircase=True)
    adam = numpyro.optim.Adam(step_size=learning_rate)
    svi = SVI(model, guide, adam, loss=TraceMeanField_ELBO())
    svi_result = svi.run(PRNGKey(random_seed), n_steps, *args, **kwargs)
    return svi_result


def get_covariates_array(
    adata,
    columns=None,
    transform: Optional[Literal["center", "z_score", "robust_z_score"]] = None,
):
    """
    Take selected columns (or all numeric columns) from adata.obs and optionally transform them.

    Transform can be either centering, z_scoring, or robust z_scoring.
    """
    if columns is None:
        columns = adata.obs.select_dtypes(include=[np.number]).columns.tolist()

    covariates_array = jnp.array(adata.obs[columns].values)
    if transform == "center":
        covariates_array = covariates_array - jnp.mean(covariates_array, axis=0)
    elif transform == "z_score":
        feature_means = jnp.mean(covariates_array, axis=0)
        feature_stds = jnp.std(covariates_array - feature_means, axis=0)
        covariates_array = (covariates_array - feature_means) / feature_stds
    elif transform == "robust_z_score":
        feature_medians = jnp.median(covariates_array, axis=0)
        feature_mads = jnp.median(jnp.abs(covariates_array - feature_medians), axis=0)
        covariates_array = (covariates_array - feature_medians) / feature_mads

    return covariates_array


def get_mdata_subset(
    mdata: MuData,
    targeting_guides: Optional[list[str]] = None,
    control_guides: Optional[list[str]] = None,
    genes: Optional[list[str]] = None,
    subset_cells: bool = True,
    rna_modality: str = "gene",
    guide_modality: str = "guide",
):
    rna_subset = mdata[rna_modality][:, genes]
    grna_subset = mdata[guide_modality][:, targeting_guides + control_guides]

    return MuData({rna_modality: rna_subset, guide_modality: grna_subset})


def get_model_args_from_mudata(mdata, rna_modality="gene", guide_modality="guide"):
    rna_subset = mdata[rna_modality]
    grna_subset = mdata[guide_modality]
    gene_obs = jnp.array(rna_subset.X.toarray()).astype(int)
    guide_obs = jnp.array(grna_subset.X).astype(bool)
    model_args = (gene_obs,)
    model_kwargs = {"guide_obs": guide_obs}
    return model_args, model_kwargs
