import numpy as np
import pandas as pd
import pytest
from mudata import AnnData, MuData
from scipy.sparse import csr_matrix

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
            "batch_id": np.random.choice(["batch_1", "batch_2", "batch_3"], size=(n_cells)),
            "cov1": np.random.normal(size=n_cells),
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
    n_cells = adata.n_obs

    # generate fake guide status (low MOI)
    grna_counts = np.zeros((n_cells, n_grna), dtype=np.float32)
    for i in range(n_cells):
        grna_counts[i, np.random.choice(n_grna)] = 1

    perturb_adata = AnnData(csr_matrix(grna_counts))
    perturb_adata.var_names = "guide" + perturb_adata.var_names
    perturb_adata.uns["elements"] = rna_adata.uns["elements"]

    guide_by_element = np.random.binomial(1, 0.8, size=(n_grna, n_elements)).astype(np.float32)

    perturb_adata.varm[guide_by_element_key] = pd.DataFrame(
        guide_by_element, index=perturb_adata.var_names, columns=perturb_adata.uns["elements"]
    )

    # combine into MuData
    return MuData({rna_key: rna_adata, perturb_key: perturb_adata})
