"""`--crt-pool auto` decides on the design the data has, not on how the file was written.

Both pools are valid for a screen whose realised MOI is a little above one, and they
answer different questions, so the rule is: test against a control pool when cells
carry about one perturbation each and there are enough unperturbed cells to be a pool.
"""
from types import SimpleNamespace

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from perturbo.core import measure_realized_moi


def _mudata(guides_per_cell, n_cells=2_000, n_guides=20, n_control_guides=4, seed=0):
    rng = np.random.default_rng(seed)
    assign = np.zeros((n_cells, n_guides), dtype=np.int8)
    for cell in range(n_cells):
        k = max(0, int(guides_per_cell[cell]))
        if k:
            assign[cell, rng.choice(n_guides, size=min(k, n_guides), replace=False)] = 1
    names = [f"non-targeting_{i}" if i < n_control_guides else f"g{i}" for i in range(n_guides)]
    guide = ad.AnnData(X=sp.csr_matrix(assign), var=pd.DataFrame(index=names))
    rna = ad.AnnData(X=np.ones((n_cells, 3), dtype=np.float32), var=pd.DataFrame(index=["a", "b", "c"]))
    return md.MuData({"rna": rna, "grna": guide})


def _measure(mdata):
    return measure_realized_moi(
        mdata, perturbation_modality_key="grna", perturbation_layer=None,
        perturbation_key=None, control_substring="non-targeting",
    )


def test_one_guide_per_cell_reads_as_moi_one():
    m = _mudata(np.ones(2_000, dtype=int))
    out = _measure(m)
    assert out["median_guides_per_cell"] == 1.0
    assert out["n_control_cells"] > 0, "cells carrying only a control guide are the pool"


def test_a_realised_moi_slightly_above_one_is_still_below_the_threshold():
    """The case the IGVF screens present: intended MOI 1, realised a little higher."""
    rng = np.random.default_rng(1)
    counts = np.where(rng.random(4_000) < 0.25, 2, 1)  # ~1.25 guides per cell
    out = _measure(_mudata(counts, n_cells=4_000))
    assert out["mean_guides_per_cell"] == pytest.approx(1.25, abs=0.05)
    assert out["median_guides_per_cell"] < 1.5, "a mild excess stays on the low-MOI side"


def test_a_high_moi_screen_reads_as_high():
    rng = np.random.default_rng(2)
    out = _measure(_mudata(rng.poisson(10, size=2_000), n_cells=2_000))
    assert out["median_guides_per_cell"] >= 1.5


def test_backed_sparse_high_moi_measurement_streams_csr_dataset(tmp_path):
    assignments = sp.csr_matrix(
        np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=np.int8,
        )
    )
    guide = ad.AnnData(
        X=assignments,
        var=pd.DataFrame(index=["non-targeting_0", "g1", "g2", "g3"]),
    )
    path = tmp_path / "backed-guides.h5ad"
    guide.write_h5ad(path)
    backed = ad.read_h5ad(path, backed="r")
    try:
        assert not hasattr(backed.X, "tocsr")
        out = measure_realized_moi(
            SimpleNamespace(mod={"grna": backed}),
            perturbation_modality_key="grna",
            perturbation_layer=None,
            control_substring="non-targeting",
        )
    finally:
        backed.file.close()

    assert out["median_guides_per_cell"] == 1.5
    assert out["mean_guides_per_cell"] == 1.25
    assert out["n_control_cells"] == 1


def test_only_cells_carrying_nothing_but_controls_count_as_controls():
    """A cell with a control guide and a targeting guide is perturbed, not a control."""
    n = 6
    assign = np.zeros((n, 20), dtype=np.int8)
    assign[0, 0] = 1              # control only
    assign[1, 1] = 1              # control only
    assign[2, 0] = assign[2, 9] = 1  # control plus targeting: perturbed
    assign[3, 9] = 1              # targeting
    # cells 4 and 5 carry nothing at all
    names = [f"non-targeting_{i}" if i < 4 else f"g{i}" for i in range(20)]
    guide = ad.AnnData(X=sp.csr_matrix(assign), var=pd.DataFrame(index=names))
    rna = ad.AnnData(X=np.ones((n, 2), dtype=np.float32), var=pd.DataFrame(index=["a", "b"]))
    out = _measure(md.MuData({"rna": rna, "grna": guide}))
    assert out["n_control_cells"] == 2


def test_anndata_input_is_moi_one_by_construction():
    labels = ["non-targeting"] * 30 + ["t1"] * 10
    adata = ad.AnnData(
        X=np.ones((40, 2), dtype=np.float32),
        obs=pd.DataFrame({"pert": labels}, index=[f"c{i}" for i in range(40)]),
        var=pd.DataFrame(index=["a", "b"]),
    )
    out = measure_realized_moi(
        adata, perturbation_modality_key=None, perturbation_layer=None,
        perturbation_key="pert", control_substring="non-targeting",
    )
    assert out["median_guides_per_cell"] == 1.0
    assert out["n_control_cells"] == 30


def test_controls_are_recognised_through_the_element_they_map_to():
    """The IGVF adapter names control *elements* 'non-targeting|...' and leaves guide ids
    alone, and the CLI matches the substring against the per-cell element label. So a
    guide whose own name says nothing must still count as a control through its element."""
    n_cells, n_guides = 300, 10
    rng = np.random.default_rng(3)
    assign = np.zeros((n_cells, n_guides), dtype=np.int8)
    assign[np.arange(n_cells), rng.integers(0, n_guides, size=n_cells)] = 1
    guide = ad.AnnData(X=sp.csr_matrix(assign), var=pd.DataFrame(index=[f"guide_{i}" for i in range(n_guides)]))
    # guides 0-3 map to two control elements, the rest to targeting elements
    element_names = ["non-targeting|1", "non-targeting|2", "GENE_A", "GENE_B"]
    mapping = np.zeros((n_guides, 4), dtype=np.float32)
    mapping[[0, 1], 0] = 1
    mapping[[2, 3], 1] = 1
    mapping[4:7, 2] = 1
    mapping[7:, 3] = 1
    guide.varm["element_map"] = mapping
    guide.uns["element_names"] = element_names
    rna = ad.AnnData(X=np.ones((n_cells, 2), dtype=np.float32), var=pd.DataFrame(index=["a", "b"]))
    mdata = md.MuData({"rna": rna, "grna": guide})

    without_map = measure_realized_moi(
        mdata, perturbation_modality_key="grna", perturbation_layer=None, control_substring="non-targeting",
    )
    with_map = measure_realized_moi(
        mdata, perturbation_modality_key="grna", perturbation_layer=None, control_substring="non-targeting",
        perturbation_element_varm_key="element_map", perturbation_element_names_uns_key="element_names",
    )
    assert without_map["n_control_cells"] == 0, "no guide name carries the substring"
    expected = int(np.isin(assign.argmax(axis=1), [0, 1, 2, 3]).sum())
    assert with_map["n_control_cells"] == expected
