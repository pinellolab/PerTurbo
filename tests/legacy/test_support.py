import pytest
from mudata import MuData

from perturbo.legacy.simulation._support_functions import mudata_filtering

from .conftest import gene_by_element_key, guide_by_element_key, perturb_key, rna_key


def test_mudata_filtering(mdata):
    result = mudata_filtering(
        mdata=mdata,
        gene_by_element_key=gene_by_element_key,
        guide_by_element_key=guide_by_element_key,
        grna_modality=perturb_key,
        rna_modality=rna_key,
    )
    assert result is not None
    # Add more assertions based on expected behavior
