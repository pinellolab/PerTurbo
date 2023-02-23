import numpy as np
import pytest
import logging
from mudata import AnnData, MuData

import perturbvi


def test_package_has_version():
    logging.info("version: " + perturbvi.__version__)


# @pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_model_init():
    rna_key = "rna"
    perturb_key = "guide"

    adata1 = AnnData(np.random.normal(size=(20, 10)))
    adata2 = AnnData(np.random.normal(size=(20, 5)))
    mdata = MuData({rna_key: adata1, perturb_key: adata2})
    model = perturbvi.PerturbVIModel(mdata)
    model.setup_mudata(mdata, rna_layer=rna_key, perturbation_layer=perturb_key)

    logging.info(model)
