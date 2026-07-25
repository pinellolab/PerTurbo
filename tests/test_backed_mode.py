from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from perturbo.core import load_analysis_cells, load_controls
from perturbo.preprocessing.counts import (
    compute_gene_clip_thresholds,
    count_gene_outliers_per_cell,
)


def _make_test_adata(n_obs: int = 50, n_vars: int = 10, seed: int = 42) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    counts = sp.random(n_obs, n_vars, density=0.4, format="csr", dtype=np.float32, random_state=rng)
    counts.data[:] = rng.integers(1, 100, size=counts.data.shape).astype(np.float32)
    perts = np.array(["ctrl"] * (n_obs // 2) + ["pertA"] * (n_obs - n_obs // 2))
    obs = pd.DataFrame({"perturbation": perts}, index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    return ad.AnnData(X=counts, obs=obs, var=var)


@pytest.fixture
def backed_and_memory_adata(tmp_path):
    adata = _make_test_adata()
    path = tmp_path / "test.h5ad"
    adata.write_h5ad(path)
    backed = ad.read_h5ad(path, backed="r")
    return adata, backed


def test_load_controls_backed_matches_memory(backed_and_memory_adata) -> None:
    mem_adata, backed_adata = backed_and_memory_adata

    result_mem = load_controls(
        mem_adata,
        perturbation_key="perturbation",
        control_selector="ctrl",
        max_control_cells=None,
    )
    result_backed = load_controls(
        backed_adata,
        perturbation_key="perturbation",
        control_selector="ctrl",
        max_control_cells=None,
    )

    np.testing.assert_array_equal(np.asarray(result_mem.counts), np.asarray(result_backed.counts))
    assert result_mem.gene_names == result_backed.gene_names
    assert result_mem.pert_names == result_backed.pert_names


def test_load_controls_backed_with_cell_keep_mask(backed_and_memory_adata) -> None:
    mem_adata, backed_adata = backed_and_memory_adata
    mask = np.zeros(mem_adata.n_obs, dtype=bool)
    mask[:30] = True

    result_mem = load_controls(
        mem_adata,
        perturbation_key="perturbation",
        control_selector="ctrl",
        cell_keep_mask=mask,
        max_control_cells=None,
    )
    result_backed = load_controls(
        backed_adata,
        perturbation_key="perturbation",
        control_selector="ctrl",
        cell_keep_mask=mask,
        max_control_cells=None,
    )

    np.testing.assert_array_equal(np.asarray(result_mem.counts), np.asarray(result_backed.counts))


def test_load_controls_backed_with_subsampling(backed_and_memory_adata) -> None:
    mem_adata, backed_adata = backed_and_memory_adata

    result_mem = load_controls(
        mem_adata,
        perturbation_key="perturbation",
        control_selector="ctrl",
        max_control_cells=5,
    )
    result_backed = load_controls(
        backed_adata,
        perturbation_key="perturbation",
        control_selector="ctrl",
        max_control_cells=5,
    )

    np.testing.assert_array_equal(np.asarray(result_mem.counts), np.asarray(result_backed.counts))
    assert np.asarray(result_mem.counts).shape[0] == 5


def test_load_analysis_cells_backed_matches_memory(backed_and_memory_adata) -> None:
    mem_adata, backed_adata = backed_and_memory_adata

    result_mem = load_analysis_cells(
        mem_adata,
        perturbation_key="perturbation",
    )
    result_backed = load_analysis_cells(
        backed_adata,
        perturbation_key="perturbation",
    )

    np.testing.assert_array_equal(np.asarray(result_mem.counts), np.asarray(result_backed.counts))
    assert result_mem.gene_names == result_backed.gene_names
    assert result_mem.pert_names == result_backed.pert_names


def test_load_analysis_cells_backed_with_cell_keep_mask(backed_and_memory_adata) -> None:
    mem_adata, backed_adata = backed_and_memory_adata
    mask = np.zeros(mem_adata.n_obs, dtype=bool)
    mask[10:40] = True

    result_mem = load_analysis_cells(
        mem_adata,
        perturbation_key="perturbation",
        cell_keep_mask=mask,
    )
    result_backed = load_analysis_cells(
        backed_adata,
        perturbation_key="perturbation",
        cell_keep_mask=mask,
    )

    np.testing.assert_array_equal(np.asarray(result_mem.counts), np.asarray(result_backed.counts))


def test_compute_gene_clip_thresholds_row_chunked_matches_default() -> None:
    rng = np.random.default_rng(99)
    counts = sp.random(100, 8, density=0.5, format="csr", dtype=np.float32, random_state=rng)
    counts.data[:] = rng.integers(1, 50, size=counts.data.shape).astype(np.float32)

    thresholds_default = compute_gene_clip_thresholds(counts, percentile=95.0)
    thresholds_chunked = compute_gene_clip_thresholds(counts, percentile=95.0, row_chunk_size=20)

    np.testing.assert_array_equal(thresholds_default, thresholds_chunked)


def test_count_gene_outliers_row_chunked_matches_default() -> None:
    rng = np.random.default_rng(77)
    counts = sp.random(60, 5, density=0.6, format="csr", dtype=np.float32, random_state=rng)
    counts.data[:] = rng.integers(1, 30, size=counts.data.shape).astype(np.float32)

    thresholds = np.array([10, 5, 15, 8, 20], dtype=np.int32)

    outliers_default = count_gene_outliers_per_cell(counts, thresholds)
    outliers_chunked = count_gene_outliers_per_cell(counts, thresholds, row_chunk_size=15)

    np.testing.assert_array_equal(outliers_default, outliers_chunked)
