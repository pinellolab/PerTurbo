import logging

import numpy as np
import pandas as pd
import pytest
from mudata import AnnData, MuData

import perturbo

rna_key = "rna"
perturb_key = "grna"


@pytest.fixture
def adata():
    """Create an example AnnData object representing a single cell perturbation screen"""
    n_cells = 20
    n_genes = 10
    n_grna = 5

    # generate fake transcript counts
    total_rna = pd.DataFrame(
        {
            "lib_size": np.random.lognormal(10, 1, size=(n_cells)),
            "batch_id": np.random.choice(["batch_1", "batch_2"], size=(n_cells)),
        }
    )
    rna_counts = np.random.negative_binomial(100, 0.9, size=(n_cells, n_genes)).astype(
        np.float64
    )
    rna_adata = AnnData(rna_counts, obs=total_rna)

    # generate fake guide status
    rna_adata.obsm[perturb_key] = np.random.binomial(1, 0.5, size=(n_cells, n_grna))

    return rna_adata


@pytest.fixture
def mdata(adata: AnnData):
    """Create an example MuData object representing a single cell perturbation screen"""
    n_grna = 5

    # generate fake transcript counts
    rna_adata = adata
    n_cells = len(adata)

    # generate fake guide status
    perturb_adata = AnnData(
        np.random.binomial(1, 0.5, size=(n_cells, n_grna)).astype(np.float64)
    )
    perturb_adata.var_names = "guide" + perturb_adata.var_names

    # combine into MuData
    return MuData({rna_key: rna_adata, perturb_key: perturb_adata})


def test_package_has_version():
    """Check that our package has an associated version number"""
    logging.info("version: " + perturbo.__version__)
    assert perturbo.__version__ is not None


def test_model_mdata(mdata: MuData, tmp_path):
    """Check that we can register our MuData object with our model and perform training"""
    perturbo.PERTURBO.setup_mudata(
        mdata,
        # size_factor_key="lib_size",
        # batch_key="batch_id",
        categorical_covariates_keys=["batch_id"],
        modalities={
            "rna_layer": rna_key,
            "perturbation_layer": perturb_key,
        },
    )
    model = perturbo.PERTURBO(mdata)
    assert model.summary_stats.n_cells == len(mdata)
    assert model.summary_stats.n_vars == len(mdata[rna_key].var)
    assert model.summary_stats.n_perturbations == len(mdata[perturb_key].var)

    model.train(max_epochs=10, lr=0.1)
    samples = model.get_posterior_samples()
    assert samples["obs"].shape[-2:] == (
        model.summary_stats.n_cells,
        model.summary_stats.n_vars,
    )
    print(model.view_anndata_setup())

    model.save(tmp_path / "model", save_anndata=True)
    model = perturbo.PERTURBO.load(tmp_path / "model")
    model.train(max_epochs=1, lr=0.1)

def test_model_adata(adata: AnnData, tmp_path):
    """Check that we can register our AnnData object with our model and perform training"""
    perturbo.PERTURBO.setup_anndata(
        adata,
        perturb_key,
        categorical_covariates_keys=["batch_id"],
        batch_key="batch_id",
    )
    model = perturbo.PERTURBO(adata)

    n_cells, n_vars = adata.shape
    assert model.summary_stats.n_cells == n_cells
    assert model.summary_stats.n_vars == n_vars
    assert model.summary_stats.n_perturbations == adata.obsm[perturb_key].shape[1]
    model.train(max_epochs=10, lr=0.1)
    samples = model.get_posterior_samples()

    assert samples["obs"].shape[-2:] == (
        model.summary_stats.n_cells,
        model.summary_stats.n_vars,
    )

    model.save(tmp_path / "model", save_anndata=True)
    model = perturbo.PERTURBO.load(tmp_path / "model")
    model.train(max_epochs=1, lr=0.1)
