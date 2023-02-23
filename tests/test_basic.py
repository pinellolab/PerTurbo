import pytest

import perturbvi
from mudata import AnnData, MuData
import numpy as np


def test_package_has_version():
    perturbvi.__version__


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_model_init():
    adata1 = AnnData(np.random.normal(size=(20, 10)))
    adata2 = AnnData(np.random.normal(size=(20, 5)))
    mdata = MuData({'data1':adata1, 'data2':adata2})
    perturbvi.PerturbVIPyroModel(mdata)