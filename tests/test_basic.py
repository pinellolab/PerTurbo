import logging

import numpy as np
import pandas as pd
import pyro
import pytest
from mudata import AnnData, MuData
from scipy.sparse import csr_matrix
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro import poutine
import perturbo

rna_key = "rna"
perturb_key = "grna"
guide_by_element_key = "guide_by_element"
gene_by_element_key = "gene_by_element"


@pytest.fixture
def adata():
    """Create an example AnnData object representing a single cell perturbation screen"""
    n_cells = 20
    n_genes = 10
    n_grna = 5
    n_elements = 3

    # generate fake transcript counts
    total_rna = pd.DataFrame(
        {
            "lib_size": np.random.lognormal(10, 1, size=(n_cells)),
            "batch_id": np.random.choice(["batch_1", "batch_2"], size=(n_cells)),
        }
    )
    rna_counts = np.random.negative_binomial(100, 0.9, size=(n_cells, n_genes)).astype(np.float32)
    rna_adata = AnnData(csr_matrix(rna_counts), obs=total_rna)
    rna_adata.var_names = "gene" + rna_adata.var_names

    # generate fake guide status (for AnnData only version)
    rna_adata.obsm[perturb_key] = np.random.binomial(1, 0.5, size=(n_cells, n_grna))

    # generate gene/element pairing
    rna_adata.uns["elements"] = [f"element{str(i)}" for i in range(n_elements)]
    gene_by_element = np.random.binomial(1, 0.5, size=(n_genes, n_elements)).astype(np.float32)
    rna_adata.varm[gene_by_element_key] = pd.DataFrame(
        gene_by_element, index=rna_adata.var_names, columns=rna_adata.uns["elements"]
    )

    return rna_adata


@pytest.fixture
def mdata(adata: AnnData):
    """Create an example MuData object representing a single cell perturbation screen"""
    n_grna = 5
    n_elements = 3

    # generate fake transcript counts
    rna_adata = adata
    n_cells = len(adata)

    # generate fake guide status
    grna_counts = np.random.binomial(1, 0.5, size=(n_cells, n_grna)).astype(np.float32)
    perturb_adata = AnnData(csr_matrix(grna_counts))
    perturb_adata.var_names = "guide" + perturb_adata.var_names
    perturb_adata.uns["elements"] = rna_adata.uns["elements"]

    guide_by_element = np.random.binomial(1, 0.8, size=(n_grna, n_elements)).astype(np.float32)
    perturb_adata.varm[guide_by_element_key] = pd.DataFrame(
        guide_by_element, index=perturb_adata.var_names, columns=perturb_adata.uns["elements"]
    )

    # combine into MuData
    return MuData({rna_key: rna_adata, perturb_key: perturb_adata})


def test_package_has_version():
    """Check that our package has an associated version number"""
    logging.info("version: " + perturbo.__version__)
    assert perturbo.__version__ is not None


@pytest.mark.parametrize("use_gene_by_element", [True, False])
@pytest.mark.parametrize("use_guide_by_element", [True, False])
@pytest.mark.parametrize("effect_prior", ["cauchy", "normal_mixture"])
def test_model_mdata(mdata: MuData, tmp_path, use_guide_by_element, use_gene_by_element, effect_prior):
    """Check that we can register our MuData object with our model and perform training"""
    if use_gene_by_element and not use_guide_by_element:
        pytest.skip("gene_by_element without guide_by_element test not implemented!")

    pyro.clear_param_store()
    perturbo.PERTURBO.setup_mudata(
        mdata,
        # size_factor_key="lib_size",
        batch_key="batch_id",
        guide_element_uns_key="elements" if use_guide_by_element else None,
        rna_element_uns_key="elements" if use_gene_by_element else None,
        categorical_covariates_keys=["lib_size"],
        guide_by_element_key=guide_by_element_key if use_guide_by_element else None,
        gene_by_element_key=gene_by_element_key if use_gene_by_element else None,
        modalities={
            "rna_layer": rna_key,
            "perturbation_layer": perturb_key,
        },
    )

    model = perturbo.PERTURBO(mdata, n_factors=None, effect_prior_dist=effect_prior)
    assert model.summary_stats.n_cells == len(mdata)
    assert model.summary_stats.n_vars == len(mdata[rna_key].var)
    assert model.summary_stats.n_perturbations == len(mdata[perturb_key].var)

    model.train(max_epochs=10, lr=0.1)
    model.train(max_epochs=10, lr=0.1, batch_size=None)
    samples = model.sample_posterior(num_samples=1, return_observed=True)

    assert samples["obs"].shape[-2:] == (
        model.summary_stats.n_cells,
        model.summary_stats.n_vars,
    )
    args, kwargs = model._get_data_subset()
    guide_trace = poutine.trace(model.module.guide).get_trace(*args, **kwargs)  # record the globals
    trained_model = poutine.replay(model.module, trace=guide_trace)  # replay the globals
    inferred_model = infer_discrete(trained_model, temperature=1, first_available_dim=-3)
    trace = poutine.trace(inferred_model).get_trace(*args, **kwargs)

    fx = model.get_element_effects()
    assert isinstance(fx, pd.DataFrame)
    assert len(model.history["elbo_train"]) == 20
    assert isinstance(model.history["elbo_train"], pd.DataFrame)
    model.save(tmp_path / "model", save_anndata=True)
    model = perturbo.PERTURBO.load(tmp_path / "model")
    # model.train(max_epochs=1, lr=0.1)
    # p_loc, p_scale = model.moduleget_perturbation_effects()


# def test_model_adata(adata: AnnData, tmp_path):
#     """Check that we can register our AnnData object with our model and perform training"""

#     pyro.clear_param_store()
#     perturbo.PERTURBO.setup_anndata(
#         adata,
#         perturb_key,
#         categorical_covariates_keys=["batch_id"],
#         batch_key="batch_id",
#     )
#     model = perturbo.PERTURBO(adata)

#     n_cells, n_vars = adata.shape
#     assert model.summary_stats.n_cells == n_cells
#     assert model.summary_stats.n_vars == n_vars
#     assert model.summary_stats.n_perturbations == adata.obsm[perturb_key].shape[1]
#     model.train(max_epochs=10, lr=0.1)
#     samples = model.get_posterior_samples()

#     assert samples["obs"].shape[-2:] == (
#         model.summary_stats.n_cells,
#         model.summary_stats.n_vars,
#     )

#     element_mu, element_sigma = model.module.get_element_effects()
#     assert element_mu.shape == (
#         model.summary_stats.n_perturbations,
#         model.summary_stats.n_vars,
#     )
#     model.save(tmp_path / "model", save_anndata=True)
#     model = perturbo.PERTURBO.load(tmp_path / "model")
#     model.train(max_epochs=1, lr=0.1)

#     e_loc, e_scale = model.module.get_element_effects()
#     p_loc, p_scale = model.module.get_perturbation_effects()
