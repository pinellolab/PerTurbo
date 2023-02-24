import numpy as np
import pytest
import logging
from mudata import AnnData, MuData

import perturbvi


rna_key = "rna"
perturb_key = "grna"


@pytest.fixture
def mdata():
    n_cells = 20
    n_genes = 10
    n_grna = 5

    rna_adata = AnnData(np.random.poisson(size=(n_cells, n_genes)), dtype=np.float64)
    perturb_adata = AnnData(np.random.binomial(1, 0.5, size=(n_cells, n_grna)), dtype=np.float64)
    perturb_adata.var_names = "guide" + perturb_adata.var_names
    return MuData({rna_key: rna_adata, perturb_key: perturb_adata})


def test_package_has_version():
    logging.info("version: " + perturbvi.__version__)


def test_model_init(mdata):
    perturbvi.PERTURBVI.setup_mudata(
        mdata,
        modalities={
            "rna_layer": rna_key,
            "perturbation_layer": perturb_key,
        },
    )
    model = perturbvi.PERTURBVI(mdata)
    assert model.summary_stats.n_cells == len(mdata)
    assert model.summary_stats.n_vars == len(mdata[rna_key].var)
    assert model.summary_stats.n_perturbations == len(mdata[perturb_key].var)

    model.train(max_epochs=2)


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_fail():
    assert 1 == 0
