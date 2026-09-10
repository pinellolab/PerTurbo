from __future__ import annotations

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
from scipy import sparse

from perturbo.preprocessing.gasperini_atscale_subset import build_gasperini_atscale_tss_subset


def test_build_gasperini_atscale_tss_subset_keeps_tss_guides_controls_and_target_genes() -> None:
    cell_names = ["c0", "c1", "c2", "c3"]
    gene_names = ["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5"]
    gene_symbols = ["ACTB", "DROP1", "TP53", "OTHER", "UNUSED"]
    guide_names = [
        "bc_0001",
        "bc_0002",
        "bc_0003",
        "bc_0004",
        "bc_0005",
    ]
    intended_targets = [
        "ACTB_TSS",
        "random_ctrl_1",
        "scrambled_1",
        "TP53_TSS",
        "drop_me",
    ]

    gene = ad.AnnData(
        X=np.array(
            [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8],
            ],
            dtype=np.int32,
        ),
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=gene_names),
    )
    gene.var["symbol"] = gene_symbols
    gene.uns["intended_targets"] = intended_targets
    guide = ad.AnnData(
        X=np.array(
            [
                [1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 0, 0, 1],
            ],
            dtype=np.int8,
        ),
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=guide_names),
    )
    guide.varm["guide_intended_target_pairs"] = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
            ],
            dtype=bool,
        )
    )
    guide.uns["intended_targets"] = intended_targets

    mdata = md.MuData({"gene": gene, "guide": guide})

    subset, summary = build_gasperini_atscale_tss_subset(mdata)

    assert summary.guides_before == 5
    assert summary.guides_after == 4
    assert summary.tss_guides == 2
    assert summary.control_guides == 2
    assert summary.targeted_genes == 2
    assert summary.cells_before == 4
    assert summary.cells_after == 3
    assert summary.genes_before == 5
    assert summary.genes_after == 2

    assert list(subset["guide"].var_names.astype(str)) == [
        "bc_0001",
        "bc_0002",
        "bc_0004",
        "bc_0005",
    ]
    assert list(subset["gene"].var_names.astype(str)) == ["ENSG1", "ENSG3"]
    assert list(subset["guide"].uns["intended_targets"]) == [
        "ACTB_TSS",
        "random_ctrl_1",
        "scrambled_1",
        "TP53_TSS",
    ]
    assert list(subset["gene"].uns["intended_targets"]) == ["ACTB", "TP53"]
    assert subset["guide"].shape == (3, 4)
    assert subset["gene"].shape == (3, 2)
    assert subset["guide"].varm["guide_intended_target_pairs"].shape == (4, 4)
