import warnings
from typing import Any

from anndata import AnnData
import jax.numpy as jnp
import numpy as np
import pandas as pd


VALID_SIZE_FACTOR_MODES = ("sum_log", "median_of_ratios")


def compute_size_factors(
    counts: Any,
    library_size: Any | None = None,
    mode: str = "sum_log",
    min_gene_mean: float = 1.0,
    pseudocount: float = 1.0,
) -> jnp.ndarray:
    """Compute centered log-size factors with shape (n_cells, 1).

    Args:
        counts: per-cell gene count matrix (n_cells, n_genes).
        library_size: precomputed total counts per cell (n_cells,). Only used
            for ``mode='sum_log'`` to avoid subsetting bias.
        mode: which size-factor estimator to use.

            * ``'sum_log'`` (default): ``log1p(library_size)`` per cell, centered.
              Equivalent to total-UMI normalization. Cheap but sensitive to
              perturbations on highly-expressed genes inflating the per-cell
              total.
            * ``'median_of_ratios'``: DESeq2-style robust normalization
              (Anders & Huber 2010). Each cell's size factor is the median
              across genes of ``count[j,g] / geomean(count[:,g])``, with a
              ``pseudocount`` added to all counts to keep geometric means
              well-defined under sparsity. Restricted to genes whose raw
              mean across cells is at least ``min_gene_mean``. Robust to
              perturbations on any gene subset.

        min_gene_mean: only used for ``mode='median_of_ratios'``; minimum
            cross-cell mean (in raw counts, no pseudocount) for a gene to be
            included in the ratio set.
        pseudocount: only used for ``mode='median_of_ratios'``; added to all
            counts before taking logs/ratios.

    Returns:
        Centered log-size factors, shape ``(n_cells, 1)``.
    """
    counts_np = np.asarray(counts, dtype=np.float32)
    if mode == "sum_log":
        if library_size is None:
            lib = counts_np.sum(axis=1)
        else:
            lib = np.asarray(library_size, dtype=np.float32)
        log_lib = np.log1p(lib)
        size_factors = log_lib - log_lib.mean()
    elif mode == "median_of_ratios":
        if library_size is not None:
            raise ValueError("library_size override is not supported for mode='median_of_ratios'.")
        gene_mask = counts_np.mean(axis=0) >= float(min_gene_mean)
        if not gene_mask.any():
            raise ValueError(
                f"No genes meet min_gene_mean={min_gene_mean!r} for median-of-ratios; "
                "lower the threshold or use a different mode."
            )
        # Add pseudocount, then work in log-space. Median commutes with the log
        # transform (monotonic), so median of log-ratios == log(median of ratios).
        log_adj = np.log(counts_np[:, gene_mask] + float(pseudocount))
        log_geomean_per_gene = log_adj.mean(axis=0)
        log_ratios = log_adj - log_geomean_per_gene[None, :]
        size_factors = np.median(log_ratios, axis=1)
        size_factors = size_factors - size_factors.mean()
    else:
        raise ValueError(
            f"Unknown size-factor mode {mode!r}; valid: {VALID_SIZE_FACTOR_MODES}."
        )
    size_factors = size_factors[:, np.newaxis]
    assert size_factors.shape == (counts_np.shape[0], 1)
    return jnp.asarray(size_factors, dtype=jnp.float32)


def map_gene_ids(
    adata: AnnData,
    gene_ids: list[str] | pd.Series,
    gene_name_col: str | None = None,
) -> list[str] | pd.Series:
    """Map gene IDs using adata.var_names -> adata.var[gene_name_col].

    Args:
        adata: AnnData object with gene names in adata.var_names and optional columns in adata.var.
        gene_ids: list or pd.Series of gene IDs to map.
        gene_name_col: column name in adata.var. If None, returns gene_ids unchanged.
    Returns:
        list[str] | pd.Series: mapped gene names.
    """
    if gene_name_col is None:
        return list(gene_ids)
    if gene_name_col not in adata.var.columns:
        raise KeyError(f"Column '{gene_name_col}' not found in adata.var.")
    gene_series = adata.var[gene_name_col].astype(str)
    if not gene_series.is_unique:
        warnings.warn(
            f"Column '{gene_name_col}' in adata.var is not unique.",
            category=UserWarning,
            stacklevel=2,
        )
    mapping = dict(
        zip(
            adata.var_names.astype(str).tolist(),
            gene_series.tolist(),
            strict=True,
        )
    )
    if isinstance(gene_ids, pd.Series):
        return gene_ids.astype(str).map(mapping)
    return [mapping[gene_id] for gene_id in gene_ids]
