from dbm.ndbm import library
from typing import Literal, Optional

import jax.numpy as jnp
import numpy as np
import numpyro
import optax
from jax.random import PRNGKey, choice
from mudata import MuData
from numpyro.infer import MCMC, NUTS, SVI, TraceMeanField_ELBO
from pandas import DataFrame
from scipy.sparse import spmatrix


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


def generate_guide_arrays(n_control=100, n_cells_per_guide=100, n_guides_per_element=1, n_elements=1):
    n_guides = n_guides_per_element * n_elements
    guides_control = jnp.zeros((n_control, n_guides))
    guides_perturbed = jnp.eye(n_guides).repeat(n_cells_per_guide, axis=0)
    guide_obs = jnp.vstack((guides_control, guides_perturbed))

    guide_by_element = jnp.eye(n_elements).repeat(n_guides_per_element, axis=0)
    return guide_obs, guide_by_element


def run_mcmc(
    args,
    kwargs,
    model,
    unconstrained_locs=None,
    dense_mass=False,
    num_samples=1000,
    random_seed=0,
    target_accept_prob=0.7,
):
    n_chains = 1
    kernel = NUTS(model, dense_mass=dense_mass, target_accept_prob=target_accept_prob, step_size=0.1)
    mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=1000, num_chains=n_chains)
    mcmc.run(PRNGKey(random_seed), *args, **kwargs, init_params=unconstrained_locs)
    return mcmc


def get_targeting_guides(
    mdata,
    gene,
    gene_modality=None,
    guide_modality=None,
    element_target_genes_varm_field=None,
    guide_target_elements_varm_field=None,
):
    element_target_genes_df = mdata[gene_modality].varm[element_target_genes_varm_field].loc[gene]
    tested_elements = list(element_target_genes_df[element_target_genes_df > 0].index)
    guide_target_elements_df = mdata[guide_modality].varm[guide_target_elements_varm_field][tested_elements]
    return list(guide_target_elements_df[guide_target_elements_df.sum(axis=1) > 0].index)


def run_svi(args, kwargs, model, guide, lr=0.03, n_steps=5000, random_seed=0, decay_rate=0.03, decay_lr=False):
    if decay_lr:
        # decay lr exponentially to a final value of lr * decay_rate, starting at n_steps // 2
        lr = optax.exponential_decay(lr, n_steps // 2, decay_rate, transition_begin=n_steps // 2, staircase=True)

    adam = numpyro.optim.Adam(step_size=lr)
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
    guides: Optional[list[str]] = None,
    genes: Optional[list[str]] = None,
    guide_target_elements_varm_field=None,
    n_extra_cells=1000,
    subset_cells: bool = False,
    rna_modality: str = "gene",
    guide_modality: str = "guide",
):
    if genes is None:
        rna_subset = mdata[rna_modality]
    else:
        rna_subset = mdata[rna_modality][:, genes]
    if guides is None:
        grna_subset = mdata[guide_modality]
    else:
        grna_subset = mdata[guide_modality][:, guides]

    if guide_target_elements_varm_field is not None:
        grna_by_element = grna_subset.varm[guide_target_elements_varm_field]
        grna_subset.varm[guide_target_elements_varm_field] = grna_by_element.loc[:, grna_by_element.sum() != 0]

    if subset_cells:
        targeted_cells = jnp.where(grna_subset.X.sum(axis=1) > 0)[0]
        control_cells = jnp.where(grna_subset.X.sum(axis=1) == 0)[0]
        assert len(control_cells) > n_extra_cells, "n_extra_cells must be smaller than number of non-targeted cells"
        selected_control_cells = choice(PRNGKey(0), control_cells, (n_extra_cells,), replace=False)
        selected_cells = np.concatenate([targeted_cells, selected_control_cells])
        rna_subset = rna_subset[selected_cells, :]
        grna_subset = grna_subset[selected_cells, :]

    return MuData({rna_modality: rna_subset, guide_modality: grna_subset})


def convert_counts_to_jnp_array(adata, layer=None):
    X = adata.X
    if layer is not None:
        X = adata.layers[layer]
    X_max = X.max()
    if X_max <= 1:
        dtype = jnp.bool_
    elif X_max < 2**15:
        dtype = jnp.int16
    else:
        dtype = jnp.int32
    if isinstance(X, spmatrix):
        X_dense = X.toarray()
        X_jnp = jnp.array(X_dense, dtype=dtype)
    else:
        X_jnp = jnp.array(X, dtype=dtype)

    assert jnp.all(jnp.equal(X_jnp, X_jnp.astype(int))), "Layer must contain integer-valued counts"

    return X_jnp


def convert_varm_to_jnp_array(adata, varm_field):
    varm = adata.varm[varm_field]
    if isinstance(varm, np.ndarray):
        values = varm
    elif isinstance(varm, spmatrix):
        values = varm.toarray()
    elif isinstance(varm, DataFrame):
        values = varm.values
    else:
        raise ValueError("Unsupported varm field type")
    assert jnp.all(jnp.equal(values, values.astype(bool))), ".varm must contain binary observations"
    varm_jnp = jnp.array(values, dtype=jnp.bool_)

    return varm_jnp


def get_model_args(
    mdata=None,
    rna_adata=None,
    grna_adata=None,
    covariates=None,
    rna_modality=None,
    guide_modality=None,
    library_size_column=None,
    covariates_transform="z_score",
    gene_counts_layer=None,
    guide_counts_layer=None,
    guide_target_elements_varm_field=None,
    element_target_genes_varm_field=None,
):
    if mdata is not None:
        assert rna_adata is None and grna_adata is None
        rna_adata = mdata[rna_modality]
        grna_adata = mdata[guide_modality]

    gene_obs = convert_counts_to_jnp_array(rna_adata, layer=gene_counts_layer)
    guide_obs = convert_counts_to_jnp_array(grna_adata, layer=guide_counts_layer)
    model_args = (gene_obs,)
    model_kwargs = {"guide_obs": guide_obs}
    if guide_target_elements_varm_field is not None:
        grna_varm_jnp = convert_varm_to_jnp_array(grna_adata, guide_target_elements_varm_field)
        model_kwargs.update({"guide_target_elements": grna_varm_jnp})
    if element_target_genes_varm_field is not None:
        rna_varm_jnp = convert_varm_to_jnp_array(rna_adata, element_target_genes_varm_field)
        model_kwargs.update({"element_target_genes": rna_varm_jnp.T})  # we transpose this here!
    if covariates is not None:
        covariates_jnp = get_covariates_array(rna_adata, columns=covariates, transform=covariates_transform)
        model_kwargs.update({"covariates": covariates_jnp})
    if library_size_column is not None:
        lib_size_jnp = get_covariates_array(rna_adata, columns=[library_size_column])
        log_lib_size = jnp.log(lib_size_jnp / jnp.mean(lib_size_jnp))
        model_kwargs.update({"size_factor": jnp.exp(log_lib_size)})
    return model_args, model_kwargs
