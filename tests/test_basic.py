import logging

import numpy as np
import pandas as pd
import pyro
import pytest
from mudata import MuData

import perturbo

from .conftest import gene_by_element_key, guide_by_element_key, perturb_key, rna_key


def test_package_has_version():
    """Check that our package has an associated version number"""
    logging.info("version: " + perturbo.__version__)
    assert perturbo.__version__ is not None


@pytest.mark.parametrize("efficiency_mode", ["mixture", "scaled"])
@pytest.mark.parametrize("use_guide_by_element", [True, False])
@pytest.mark.parametrize("use_gene_by_element", [True, False])
@pytest.mark.parametrize("fit_guide_efficacy", [True, False])
@pytest.mark.parametrize("sparse_tensors", [True, False])
# @pytest.mark.parametrize("n_factors", [None, 2])
@pytest.mark.parametrize("n_pert_factors", [None, 2])
def test_model_mdata(
    mdata: MuData,
    tmp_path,
    efficiency_mode,
    use_guide_by_element,
    use_gene_by_element,
    fit_guide_efficacy,
    sparse_tensors,
    n_pert_factors,
):
    """Check that we can register our MuData object with our model and perform training"""
    if use_gene_by_element and not use_guide_by_element:
        pytest.skip("gene_by_element without guide_by_element not implemented!")

    if n_pert_factors and fit_guide_efficacy:
        pytest.skip("cannot fit guide efficacy if using n_pert_factors!")

    pyro.clear_param_store()
    perturbo.PERTURBO.setup_mudata(
        mdata,
        library_size_key="lib_size",
        batch_key="batch_id",
        continuous_covariates_keys=["cov1"],
        guide_element_uns_key="elements" if use_guide_by_element else None,
        rna_element_uns_key="elements" if use_gene_by_element else None,
        guide_by_element_key=guide_by_element_key if use_guide_by_element else None,
        gene_by_element_key=gene_by_element_key if use_gene_by_element else None,
        modalities={
            "rna_layer": rna_key,
            "perturbation_layer": perturb_key,
        },
    )

    model = perturbo.PERTURBO(
        mdata,
        # control_guides=["guide0"],
        control_guides=[0],
        n_pert_factors=n_pert_factors,
        efficiency_mode=efficiency_mode,
        fit_guide_efficacy=fit_guide_efficacy,
        sparse_effect_tensors=sparse_tensors,
    )
    assert model.summary_stats.n_cells == len(mdata)
    assert model.summary_stats.n_vars == len(mdata[rna_key].var)
    assert model.summary_stats.n_perturbations == len(mdata[perturb_key].var)

    model.train(
        max_epochs=5,
        lr=0.1,
        batch_size=2,
        load_sparse_tensor=sparse_tensors,
    )
    model.train(
        max_epochs=5,
        lr=0.1,
        batch_size=None,
        load_sparse_tensor=sparse_tensors,
    )
    element_effects = model.get_element_effects()
    assert isinstance(element_effects, pd.DataFrame)
    assert isinstance(model.history["elbo_train"], pd.DataFrame)

    # test model save/load
    model.save(tmp_path / "model", save_anndata=True)
    model = perturbo.PERTURBO.load(tmp_path / "model")

    # test simulator
    n_cells = mdata[rna_key].n_obs
    n_cells_new = n_cells * 2  # greater than original
    n_genes = mdata[rna_key].n_vars

    grna_counts = mdata[perturb_key].X
    cell_idx = np.arange(3)

    # run simulator once with original settings
    guide_by_element = np.random.binomial(1, 0.8, size=(model.module.n_perturbations, model.module.n_elements))
    element_by_gene_lfc = np.random.normal(0, 1, size=(model.module.n_elements, model.module.n_genes))

    guide_efficacy = np.random.random_sample(size=(model.module.n_perturbations, 1))

    mdata_new = perturbo.simulation.simulate_data_from_trained_model(
        model,
        guide_obs=grna_counts.todense()[cell_idx, :],
        guide_by_element=guide_by_element,
        element_by_gene_lfc=element_by_gene_lfc,
        guide_efficacy=guide_efficacy,
        cell_indices=cell_idx,
    )

    # run simulator once with new data shape/model
    n_grna_new = 16
    n_elements_new = 4

    grna_counts_new = np.zeros((n_cells_new, n_grna_new), dtype=np.float32)
    guide_efficacy_new = np.random.uniform(size=(n_grna_new,))
    for i in range(n_cells_new):
        grna_counts_new[i, np.random.choice(n_grna_new)] = 1

    n_genes_new = n_genes // 2  # less than original
    new_genes_idx = np.random.choice(n_genes, size=n_genes_new, replace=False)
    guide_by_element_new = np.random.binomial(1, 0.8, size=(n_grna_new, n_elements_new))
    element_by_gene_lfc_new = np.random.normal(0, 1, size=(n_elements_new, n_genes_new))

    mdata_new = perturbo.simulation.simulate_data_from_trained_model(
        model,
        guide_obs=grna_counts_new,
        guide_by_element=guide_by_element_new,
        element_by_gene_lfc=element_by_gene_lfc_new,
        guide_efficacy=guide_efficacy_new,
        gene_indices=new_genes_idx,
        module_init_kwargs={"efficiency_mode": "mixture_high_moi"},
    )

    assert mdata_new[rna_key].shape == (n_cells_new, n_genes_new)
    assert mdata_new[perturb_key].shape == (n_cells_new, n_grna_new)

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
