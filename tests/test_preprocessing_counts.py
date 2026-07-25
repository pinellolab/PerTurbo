from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from perturbo.preprocessing import (
    compute_gene_clip_thresholds,
    count_gene_outliers_per_cell,
    to_dense_array,
    winsorize_counts_to_gene_thresholds,
)


def test_preprocessing_public_exports_exist() -> None:
    assert callable(to_dense_array)
    assert callable(compute_gene_clip_thresholds)
    assert callable(count_gene_outliers_per_cell)
    assert callable(winsorize_counts_to_gene_thresholds)


def test_to_dense_array_supports_dense_sparse_and_matrix_like_inputs() -> None:
    class _ArrayLikeWithA:
        def __init__(self, arr: np.ndarray) -> None:
            self.A = arr

    dense = np.array([[1, 2], [3, 4]], dtype=np.int32)
    csr = sp.csr_matrix(dense)
    csc = sp.csc_matrix(dense)
    matrix_like = _ArrayLikeWithA(dense)

    np.testing.assert_array_equal(to_dense_array(dense), dense)
    np.testing.assert_array_equal(to_dense_array(csr), dense)
    np.testing.assert_array_equal(to_dense_array(csc), dense)
    np.testing.assert_array_equal(to_dense_array(matrix_like), dense)


def test_compute_gene_clip_thresholds_handles_empty_gene_axis() -> None:
    matrix = sp.csr_matrix((3, 0), dtype=np.int32)
    thresholds = compute_gene_clip_thresholds(matrix, percentile=99.0)
    assert thresholds.dtype == np.int32
    assert thresholds.shape == (0,)


@pytest.mark.parametrize("percentile", [0.0, 101.0, np.inf, np.nan])
def test_compute_gene_clip_thresholds_validates_percentile(percentile: float) -> None:
    counts = np.array([[0, 1], [2, 3]], dtype=np.int32)
    with pytest.raises(ValueError, match="clip_gene_expression_percentile"):
        compute_gene_clip_thresholds(counts, percentile=percentile)


def test_compute_gene_clip_thresholds_applies_floor_and_matches_dense_quantiles_blockwise() -> None:
    counts = np.array(
        [
            [0, 5, 1, 2, 0],
            [1, 2, 9, 4, 3],
            [0, 8, 3, 6, 1],
            [2, 1, 7, 5, 0],
        ],
        dtype=np.int32,
    )
    expected = np.maximum(
        np.ceil(np.quantile(counts, 0.75, axis=0)).astype(np.int32),
        4,
    )
    thresholds = compute_gene_clip_thresholds(
        sp.csr_matrix(counts),
        percentile=75.0,
        threshold_floor=4,
        target_dense_bytes=32,
    )
    np.testing.assert_array_equal(thresholds, expected)


def test_count_gene_outliers_per_cell_supports_dense_and_sparse_inputs() -> None:
    counts = np.array(
        [
            [0, 5, 1],
            [4, 1, 8],
            [2, 3, 0],
        ],
        dtype=np.int32,
    )
    thresholds = np.array([1, 4, 3], dtype=np.int32)
    expected = np.array([1, 2, 1], dtype=np.int32)

    np.testing.assert_array_equal(count_gene_outliers_per_cell(counts, thresholds), expected)
    np.testing.assert_array_equal(count_gene_outliers_per_cell(sp.csr_matrix(counts), thresholds), expected)


def test_count_gene_outliers_per_cell_rejects_threshold_shape_mismatch() -> None:
    counts = np.array([[0, 1], [2, 3]], dtype=np.int32)
    with pytest.raises(ValueError, match="thresholds must be a 1D array"):
        count_gene_outliers_per_cell(counts, np.array([1, 2, 3], dtype=np.int32))


def test_winsorize_counts_to_gene_thresholds_passthrough_and_clipping() -> None:
    counts = np.array([[0, 5], [7, 2]], dtype=np.int32)

    assert winsorize_counts_to_gene_thresholds(counts, None) is counts
    np.testing.assert_array_equal(
        winsorize_counts_to_gene_thresholds(counts, np.array([4, 3], dtype=np.int32)),
        np.array([[0, 3], [4, 2]], dtype=np.int32),
    )


def test_winsorize_counts_to_gene_thresholds_rejects_threshold_shape_mismatch() -> None:
    counts = np.array([[0, 1], [2, 3]], dtype=np.int32)
    with pytest.raises(ValueError, match="gene_clip_thresholds must be a 1D array"):
        winsorize_counts_to_gene_thresholds(counts, np.array([1, 2, 3], dtype=np.int32))
