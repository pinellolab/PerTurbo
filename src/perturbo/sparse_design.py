"""Compact row-indexed design matrices for sparse perturbation assignments."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp


class IndexedDesignMatrix(NamedTuple):
    """A fixed-width list of active columns for every design row.

    ``indices`` and ``values`` have shape ``(num_rows, max_active)``. An index of
    ``-1`` is padding and contributes zero. This representation is a JAX pytree,
    can be cell-minibatched with ordinary gathers, and stores high-MOI designs in
    memory proportional to observed assignments rather than rows times columns.
    """

    indices: jnp.ndarray
    values: jnp.ndarray
    num_columns: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.indices.shape[0], self.num_columns

    @property
    def ndim(self) -> int:
        return 2

    def take_rows(self, row_indices) -> IndexedDesignMatrix:
        return IndexedDesignMatrix(
            indices=self.indices[row_indices],
            values=self.values[row_indices],
            num_columns=self.num_columns,
        )


def indexed_design_from_matrix(matrix) -> IndexedDesignMatrix:
    """Convert a dense or scipy sparse matrix to padded active-column rows."""
    if sp.issparse(matrix) or hasattr(matrix, "tocsr"):
        csr = matrix.tocsr(copy=True)
        csr.eliminate_zeros()
        num_rows, num_columns = csr.shape
        row_widths = np.diff(csr.indptr)
        max_active = max(1, int(row_widths.max(initial=0)))
        indices = np.full((num_rows, max_active), -1, dtype=np.int32)
        values = np.zeros((num_rows, max_active), dtype=np.float32)
        if csr.nnz:
            rows = np.repeat(np.arange(num_rows, dtype=np.int64), row_widths)
            slots = np.arange(csr.nnz, dtype=np.int64) - np.repeat(csr.indptr[:-1], row_widths)
            indices[rows, slots] = csr.indices
            values[rows, slots] = csr.data
    else:
        dense = np.asarray(matrix)
        if dense.ndim != 2:
            raise ValueError("An indexed design source must be a 2D matrix.")
        num_rows, num_columns = dense.shape
        active = dense != 0
        row_widths = active.sum(axis=1)
        max_active = max(1, int(row_widths.max(initial=0)))
        indices = np.full((num_rows, max_active), -1, dtype=np.int32)
        values = np.zeros((num_rows, max_active), dtype=np.float32)
        rows, columns = np.nonzero(active)
        if rows.size:
            row_starts = np.repeat(np.cumsum(row_widths) - row_widths, row_widths)
            slots = np.arange(rows.size, dtype=np.int64) - row_starts
            indices[rows, slots] = columns
            values[rows, slots] = dense[rows, columns]
    return IndexedDesignMatrix(
        indices=jnp.asarray(indices, dtype=jnp.int32),
        values=jnp.asarray(values, dtype=jnp.float32),
        num_columns=int(num_columns),
    )


def indexed_design_to_dense(design: IndexedDesignMatrix) -> jnp.ndarray:
    """Materialize an indexed design, primarily for validation and tests."""
    safe_indices = jnp.maximum(design.indices, 0)
    mask = design.indices >= 0
    rows = jnp.arange(design.indices.shape[0], dtype=jnp.int32)[:, None]
    dense = jnp.zeros(design.shape, dtype=design.values.dtype)
    return dense.at[rows, safe_indices].add(jnp.where(mask, design.values, 0))


def design_matrix_product(design, coefficients: jnp.ndarray) -> jnp.ndarray:
    """Return ``design @ coefficients`` without densifying an indexed design."""
    if not isinstance(design, IndexedDesignMatrix):
        return jnp.asarray(design, dtype=coefficients.dtype) @ coefficients
    valid = design.indices >= 0
    # Keep padding out of both the gather and its transpose. Clamping ``-1`` to
    # zero makes every padded slot a (zero-weighted) gradient destination for
    # column zero, which can dominate sparse high-MOI reverse-mode work.
    indices = jnp.where(valid, design.indices, coefficients.shape[0])
    gathered = coefficients.at[indices].get(mode="fill", fill_value=0)
    weights = jnp.where(valid, design.values, 0).astype(coefficients.dtype)
    return jnp.sum(gathered * weights[..., None], axis=1)


def design_values_are_finite_nonnegative(design) -> bool:
    if isinstance(design, IndexedDesignMatrix):
        values = np.asarray(design.values)
    else:
        values = np.asarray(design)
    return bool(np.all(np.isfinite(values)) and np.all(values >= 0))
