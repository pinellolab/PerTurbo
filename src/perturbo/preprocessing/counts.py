from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# Upper bound on integer count values used when building histograms in backed
# (row-chunked) mode. Values above this are capped, so if the true percentile
# threshold for a gene exceeds this value the threshold is conservatively set
# to _HIST_MAX_COUNT (clipping will be slightly more aggressive than requested,
# which is safe for scRNA-seq data where such counts are essentially absent).
_HIST_MAX_COUNT = 10_000

# Default row chunk size for backed/disk-resident matrices.
BACKED_ROW_CHUNK_SIZE = 50_000


def to_dense_array(x) -> np.ndarray:
    """Convert an array or sparse matrix to a dense NumPy array.

    Parameters
    ----------
    x : numpy.ndarray or scipy.sparse.spmatrix
        Input to materialize. Slice backed matrices before calling this helper.

    Returns
    -------
    numpy.ndarray
        Dense array with the input shape. An existing dense array can be
        returned without copying; sparse inputs allocate dense storage.
    """
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    if hasattr(x, "A"):
        return np.asarray(x.A)
    return np.asarray(x)


def _validate_clip_percentile(percentile: float | None) -> float | None:
    if percentile is None:
        return None
    value = float(percentile)
    if not np.isfinite(value):
        raise ValueError("clip_gene_expression_percentile must be finite.")
    if value <= 0.0 or value > 100.0:
        raise ValueError("clip_gene_expression_percentile must be in the interval (0, 100].")
    return value


def _validate_gene_outlier_threshold_floor(floor: int) -> int:
    value = int(floor)
    if value < 0:
        raise ValueError("gene_outlier_threshold_floor must be >= 0.")
    return value


def compute_gene_clip_thresholds(
    matrix,
    *,
    percentile: float,
    threshold_floor: int = 2,
    target_dense_bytes: int = 64 * 1024 * 1024,
    max_histogram_bytes: int = 2 * 1024 * 1024 * 1024,
    row_chunk_size: int | None = None,
) -> np.ndarray:
    """Compute integer gene thresholds from cell-wise count percentiles.

    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse.spmatrix
        Nonnegative integer counts with shape ``(n_cells, n_genes)``.
        A row-sliceable backed matrix is supported with ``row_chunk_size``.
    percentile : float
        Percentile in ``(0, 100]``, computed across all cells including zeros.
        The linear quantile is rounded upward to an integer.
    threshold_floor : int, default: 2
        Nonnegative minimum threshold for every gene.
    target_dense_bytes : int, default: 67108864
        Target byte budget for a dense working gene block. This is not a cap
        on total process memory; at least one gene is processed.
    max_histogram_bytes : int, default: 2147483648
        Histogram budget in row-chunked mode. Genes that cannot be represented
        together require additional passes over the matrix. Other working
        arrays, including cumulative histograms, need additional memory.
    row_chunk_size : int or None, default: None
        Positive number of rows per read. When set, use integer histograms
        suitable for backed input. Otherwise read gene blocks and use NumPy
        quantiles directly.

    Returns
    -------
    numpy.ndarray
        ``int32`` thresholds of shape ``(n_genes,)`` in the input gene order.

    Raises
    ------
    ValueError
        If the percentile is missing, nonfinite, or outside ``(0, 100]``, or
        the threshold floor is negative.

    Notes
    -----
    The row-chunked histogram path caps input counts at 10,000. Thresholds
    above that count are therefore capped before applying the floor. Use
    integer counts; fractional values are cast to integers on this path.
    The input matrix is not modified.
    """
    percentile = _validate_clip_percentile(percentile)
    if percentile is None:
        raise ValueError("percentile must not be None.")
    threshold_floor = _validate_gene_outlier_threshold_floor(threshold_floor)
    n_obs, n_vars = matrix.shape
    if n_vars == 0:
        return np.zeros((0,), dtype=np.int32)

    q = percentile / 100.0
    thresholds = np.empty(n_vars, dtype=np.int32)

    if row_chunk_size is not None:
        # Row-chunked histogram path for backed / disk-resident matrices.
        # Gene blocks are sized so that both the histogram array and the dense
        # working buffer for one row chunk fit within target_dense_bytes.
        _MAX = _HIST_MAX_COUNT
        int32_size = np.dtype(np.int32).itemsize
        # Size of the dense working buffer for one row chunk. This is a memory
        # bound, not an I/O one, so it only decides how a row chunk already in
        # hand is carved up.
        genes_per_block = max(
            1,
            min(n_vars, target_dense_bytes // (row_chunk_size * int32_size)),
        )
        # How many genes' histograms can be held at once. This is what decides
        # how many times the matrix is read, because a backed row slice reads
        # every gene whatever column subset is wanted afterwards. Holding the
        # histograms for one gene block and looping rows inside that - which
        # this did until it was measured - reads the whole matrix once per gene
        # block: 25 passes over 62 GB on the genome-wide screen, to build a
        # histogram that needs one.
        genes_per_pass = max(
            genes_per_block,
            min(n_vars, max_histogram_bytes // ((_MAX + 1) * int32_size)),
        )

        # Replicate numpy's linear quantile formula exactly so results match
        # the column-chunked path: result = x[lo] + (x[hi] - x[lo]) * frac,
        # then ceil.  x[lo] and x[hi] are the lo-th and hi-th order statistics
        # (0-indexed), recoverable from the per-gene cumulative histogram as
        # the first value v where cumsum[v] >= rank.
        h = q * (n_obs - 1)
        lo = int(np.floor(h))
        hi = int(np.ceil(h))
        frac = h - lo

        def _rank_value(cumsum_2d: np.ndarray, rank: int) -> np.ndarray:
            """First histogram bin where cumsum >= rank, capped at _MAX on overflow."""
            idx = np.argmax(cumsum_2d >= rank, axis=0).astype(np.int32)
            return np.where(cumsum_2d[-1] >= rank, idx, np.int32(_MAX))

        for p_start in range(0, n_vars, genes_per_pass):
            p_stop = min(n_vars, p_start + genes_per_pass)
            hist = np.zeros((_MAX + 1, p_stop - p_start), dtype=np.int32)

            for r_start in range(0, n_obs, row_chunk_size):
                r_stop = min(n_obs, r_start + row_chunk_size)
                # The one disk read. Every gene block below slices this in
                # memory rather than going back to the file.
                row_block = matrix[r_start:r_stop]
                for g_start in range(p_start, p_stop, genes_per_block):
                    g_stop = min(p_stop, g_start + genes_per_block)
                    n_block = g_stop - g_start
                    dense = to_dense_array(row_block[:, g_start:g_stop]).astype(np.int32, copy=False)
                    clipped = np.clip(dense, 0, _MAX)
                    # Vectorised 2-D histogram via offset trick: shift each gene's
                    # values into a non-overlapping range then use a single bincount.
                    offsets = (np.arange(n_block, dtype=np.int32) * (_MAX + 1))[None, :]
                    flat = (clipped + offsets).ravel()
                    flat_hist = np.bincount(flat, minlength=(_MAX + 1) * n_block)
                    hist[:, g_start - p_start : g_stop - p_start] += (
                        flat_hist.reshape(n_block, _MAX + 1).T.astype(np.int32)
                    )

            cumsum = np.cumsum(hist, axis=0)
            x_lo = _rank_value(cumsum, lo + 1).astype(np.float64)
            x_hi = _rank_value(cumsum, hi + 1).astype(np.float64)
            result = x_lo + (x_hi - x_lo) * frac
            thresholds[p_start:p_stop] = np.maximum(np.ceil(result).astype(np.int32), threshold_floor)

        return thresholds

    # Original column-chunked in-memory path.
    matrix_dtype = np.dtype(getattr(matrix, "dtype", np.int32))
    itemsize = max(matrix_dtype.itemsize, np.dtype(np.float32).itemsize)
    bytes_per_gene = max(1, n_obs * itemsize)
    genes_per_block = max(1, min(n_vars, target_dense_bytes // bytes_per_gene))

    for start in range(0, n_vars, genes_per_block):
        stop = min(n_vars, start + genes_per_block)
        block = to_dense_array(matrix[:, start:stop])
        block_thresholds = np.ceil(np.quantile(block, q, axis=0)).astype(np.int32, copy=False)
        thresholds[start:stop] = np.maximum(block_thresholds, threshold_floor)

    return thresholds


def count_gene_outliers_per_cell(
    matrix,
    thresholds: np.ndarray | None,
    *,
    row_chunk_size: int | None = None,
) -> np.ndarray:
    """Count genes strictly above their thresholds in each cell.

    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse.spmatrix
        Counts with shape ``(n_cells, n_genes)``. Canonical sparse matrices
        should contain no duplicate entries; implicit zeros are assumed not
        to exceed the nonnegative thresholds.
    thresholds : numpy.ndarray
        Nonnegative thresholds of shape ``(n_genes,)`` in matrix gene order.
        ``None`` is not accepted.
    row_chunk_size : int or None, default: None
        Positive rows per read for backed input; otherwise use the matrix
        directly. Sparse inputs are compared without densifying.

    Returns
    -------
    numpy.ndarray
        ``int32`` counts of shape ``(n_cells,)``. Values equal to a threshold
        are not counted. Neither input is modified.

    Raises
    ------
    ValueError
        If thresholds are missing or do not match the gene axis.
    """
    if thresholds is None:
        raise ValueError("thresholds must not be None when counting gene outliers per cell.")

    threshold_arr = np.asarray(thresholds)
    n_obs, n_vars = matrix.shape
    if threshold_arr.ndim != 1 or threshold_arr.shape[0] != n_vars:
        raise ValueError("thresholds must be a 1D array with length matching the number of genes.")

    if row_chunk_size is not None:
        result = np.zeros(n_obs, dtype=np.int32)
        for r_start in range(0, n_obs, row_chunk_size):
            r_stop = min(n_obs, r_start + row_chunk_size)
            chunk = matrix[r_start:r_stop]
            if sp.issparse(chunk):
                chunk_csr = chunk.tocsr(copy=False)
                flagged = (
                    np.asarray(chunk_csr.data) > threshold_arr[np.asarray(chunk_csr.indices)]
                ).astype(np.int32, copy=False)
                flagged_matrix = sp.csr_matrix(
                    (flagged, chunk_csr.indices, chunk_csr.indptr), shape=chunk_csr.shape
                )
                result[r_start:r_stop] = np.asarray(flagged_matrix.sum(axis=1)).ravel().astype(np.int32)
            else:
                dense = to_dense_array(chunk)
                result[r_start:r_stop] = np.count_nonzero(
                    dense > threshold_arr[None, :], axis=1
                ).astype(np.int32)
        return result

    if sp.issparse(matrix):
        matrix_csr = matrix.tocsr(copy=False)
        flagged = (np.asarray(matrix_csr.data) > threshold_arr[np.asarray(matrix_csr.indices)]).astype(
            np.int32, copy=False
        )
        flagged_matrix = sp.csr_matrix((flagged, matrix_csr.indices, matrix_csr.indptr), shape=matrix_csr.shape)
        return np.asarray(flagged_matrix.sum(axis=1)).reshape(-1).astype(np.int32, copy=False)

    dense = to_dense_array(matrix)
    return np.count_nonzero(dense > threshold_arr[None, :], axis=1).astype(np.int32, copy=False)


def winsorize_counts_to_gene_thresholds(counts: np.ndarray, thresholds: np.ndarray | None) -> np.ndarray:
    """Cap a dense count array at aligned gene-specific thresholds.

    Parameters
    ----------
    counts : numpy.ndarray
        Dense counts of shape ``(n_cells, n_genes)``.
    thresholds : numpy.ndarray or None
        Upper cutoffs of shape ``(n_genes,)`` in count-column order.
        ``None`` disables clipping.

    Returns
    -------
    numpy.ndarray
        Elementwise minimum of counts and thresholds, with the same shape.
        Clipping returns a new array without modifying counts. When thresholds
        are ``None``, the original object is returned unchanged.

    Raises
    ------
    ValueError
        If a provided threshold array does not match the gene axis.
    """
    if thresholds is None:
        return counts
    arr = np.asarray(counts)
    threshold_arr = np.asarray(thresholds)
    if threshold_arr.ndim != 1 or threshold_arr.shape[0] != arr.shape[1]:
        raise ValueError("gene_clip_thresholds must be a 1D array with length matching the number of genes.")
    return np.minimum(arr, threshold_arr[None, :])
