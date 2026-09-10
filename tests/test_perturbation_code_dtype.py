"""Perturbation codes must not inherit pandas' narrow categorical dtype.

A chunk holding a single perturbation gave int8 codes, and stage two's jitted step,
which combines the codes with the (much larger) perturbation count of the first
chunk, failed with ``OverflowError: Python integer 742 out of bounds for int8``.
"""
import anndata as ad
import numpy as np
import pandas as pd

from perturbo.core import _extract_names


def _adata(labels):
    X = np.ones((len(labels), 3), dtype=np.float32)
    obs = pd.DataFrame({"pert": labels}, index=[f"c{i}" for i in range(len(labels))])
    var = pd.DataFrame(index=[f"g{i}" for i in range(3)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_single_perturbation_codes_are_wide():
    adata = _adata(["only_one"] * 5)
    _, names, codes = _extract_names(adata, None, "pert")
    assert names == ["only_one"]
    assert codes.dtype == np.int32
    assert codes.tolist() == [0] * 5


def test_override_with_many_categories_keeps_wide_codes():
    labels = [f"p{i}" for i in range(3)]
    adata = _adata(labels)
    override = [f"p{i}" for i in range(800)]  # more than int8 can index
    _, names, codes = _extract_names(adata, None, "pert", pert_names_override=override)
    assert len(names) == 800
    assert codes.dtype == np.int32
    # a sentinel as large as the perturbation count must be representable alongside the codes
    combined = np.where(codes >= 0, codes, len(names))
    assert combined.max() < len(names) + 1
