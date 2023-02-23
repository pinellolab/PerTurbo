import numpy as np
import pytest
import logging
from mudata import AnnData, MuData

import perturbvi


def test_package_has_version():
    logging.info("version: " + perturbvi.__version__)


def test_model_init():
    rna_key = "rna"
    perturb_key = "grna"

    rna_adata = AnnData(np.random.poisson(size=(20, 10)), dtype=np.float64)
    perturb_adata = AnnData(np.random.binomial(1, 0.5, size=(20, 5)), dtype=np.float64)
    perturb_adata.var_names = 'guide' + perturb_adata.var_names
    mdata = MuData({rna_key: rna_adata, perturb_key: perturb_adata})

    perturbvi.PERTURBVI.setup_mudata(
        mdata,
        modalities={
            "rna_layer": rna_key,
            "perturbation_layer": perturb_key,
        },
    )
    model = perturbvi.PERTURBVI(mdata)
    logging.info(model)


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_fail():
    assert 1 == 0
