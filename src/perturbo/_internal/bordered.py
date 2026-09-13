"""Weighted nuisance algebra that preserves the original reference coding.

Mutually exclusive indicator columns form a diagonal information block. All
other columns, including the intercept, remain in a small dense border. The
Schur complement changes only the linear algebra, never the parameterization
or its diagonal penalty. A missing/reference level has no indicator column.

Detection is a host operation. The returned named tuples are JAX pytrees;
NumPy leaves retain their precision for the matching host diagnostic routines.
No routine here constructs the full nuisance-by-nuisance information matrix.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class BorderedDesign(NamedTuple):
    border: object
    codes: object
    border_indices: object
    group_indices: object

    @property
    def num_border(self) -> int:
        return int(self.border_indices.shape[0])

    @property
    def num_groups(self) -> int:
        return int(self.group_indices.shape[0])

    @property
    def num_columns(self) -> int:
        return self.num_border + self.num_groups

    def take(self, rows) -> BorderedDesign:
        """Subset rows, preserving column identities and reference coding."""
        # Within jit/vmap, leaves are tracers and ordinary indexing is traced.
        # Outside a trace, NumPy indexing preserves host diagnostic precision.
        if isinstance(rows, jax.core.Tracer):
            return self._replace(border=jnp.asarray(self.border)[rows], codes=jnp.asarray(self.codes)[rows])
        return self._replace(border=self.border[rows], codes=self.codes[rows])

    def pad_rows(self, count: int = 1) -> BorderedDesign:
        """Append zero-contribution rows for padded gathers, not intercepts."""
        if count < 0:
            raise ValueError("The number of padded rows must be nonnegative.")
        xp = np if isinstance(self.border, np.ndarray) and isinstance(self.codes, np.ndarray) else jnp
        return self._replace(
            border=xp.concatenate([self.border, xp.zeros((count, self.num_border), dtype=self.border.dtype)]),
            codes=xp.concatenate([self.codes, xp.full((count,), self.num_groups, dtype=self.codes.dtype)]),
        )

    def astype(self, dtype) -> BorderedDesign:
        """Convert only numeric design values; indices remain integral."""
        return self._replace(border=self.border.astype(dtype))


class BorderedInfo(NamedTuple):
    """Information blocks in shapes ``(..., G, d, d/d,K/K)``."""

    border: object
    cross: object
    diagonal: object


def detect_bordered_design(nuisance_design) -> BorderedDesign | None:
    """Recognize at least two mutually exclusive binary indicator columns.

    Detection is intentionally conservative: overlapping binary columns,
    nonfinite/non-numeric values, and fewer than two indicators return None.
    All-zero columns retain their original penalties in the diagonal block,
    so absent batch levels do not enlarge the dense border. All-one columns
    stay in the border; the original intercept is never replaced by a full
    set of batch coefficients.
    """
    design = np.asarray(nuisance_design)
    if (
        design.ndim != 2
        or design.shape[0] == 0
        or not np.issubdtype(design.dtype, np.number)
        or np.iscomplexobj(design)
        or not np.isfinite(design).all()
    ):
        return None
    binary = np.all((design == 0) | (design == 1), axis=0)
    all_one = np.all(design == 1, axis=0)
    group_indices = np.flatnonzero(binary & ~all_one).astype(np.int32)
    if group_indices.size < 2:
        return None
    indicator_block = design[:, group_indices]
    if np.any(indicator_block.sum(axis=1) > 1):
        return None
    border_indices = np.setdiff1d(np.arange(design.shape[1]), group_indices).astype(np.int32)
    codes = np.full(design.shape[0], group_indices.size, dtype=np.int32)
    rows, groups = np.nonzero(indicator_block)
    codes[rows] = groups
    return BorderedDesign(design[:, border_indices].copy(), codes, border_indices, group_indices)


def has_reference_dependency(design: BorderedDesign) -> bool:
    """Host check for an intercept aliased with all observed indicator levels.

    If the reference level is absent, the group indicators sum to one. Any
    nonzero constant border column then lies in their span. In float32, a tiny
    existing jitter need not resolve that dependency, so callers can retain
    their dense fallback instead of accepting a different finite solution.
    Apply this check after host row subsetting, before entering a JAX trace.
    """
    codes = np.asarray(design.codes)
    border = np.asarray(design.border)
    if codes.size == 0 or np.any(codes == design.num_groups) or design.num_border == 0:
        return False
    return bool(np.any((border[0] != 0) & np.all(border == border[0], axis=0)))


def to_dense_numpy(design: BorderedDesign) -> np.ndarray:
    """Reconstruct original column coding for an explicitly selected fallback."""
    border = np.asarray(design.border)
    codes = np.asarray(design.codes)
    dense = np.zeros((border.shape[0], design.num_columns), dtype=border.dtype)
    dense[:, np.asarray(design.border_indices)] = border
    rows = np.flatnonzero(codes < design.num_groups)
    dense[rows, np.asarray(design.group_indices)[codes[rows]]] = 1
    return dense


def _segment_sum(data, codes, num_groups, xp):
    """Sum an array with cells on axis zero; discard the reference sentinel."""
    if xp is np:
        out = np.zeros((num_groups + 1,) + data.shape[1:], dtype=data.dtype)
        np.add.at(out, np.asarray(codes), data)
    else:
        out = jax.ops.segment_sum(data, jnp.asarray(codes), num_segments=num_groups + 1)
    return out[:num_groups]


def _weighted_information(design, weights, ridge, xp):
    border = xp.asarray(design.border)
    weights = xp.asarray(weights)
    if weights.ndim < 2 or weights.shape[-2] != border.shape[0]:
        raise ValueError("Weights must have shape (..., cells, genes) matching the design.")
    ridge = xp.asarray(ridge)
    if ridge.ndim and ridge.shape[-1] != design.num_columns:
        raise ValueError("A diagonal ridge must use the original nuisance column order.")
    dtype = xp.result_type(border, weights, ridge, xp.float32)
    border, weights, ridge = (xp.asarray(value, dtype=dtype) for value in (border, weights, ridge))
    block = xp.einsum("nd,...ng,ne->...gde", border, weights, border)
    diagonal = xp.moveaxis(
        _segment_sum(xp.moveaxis(weights, -2, 0), design.codes, design.num_groups, xp), 0, -1
    )
    weighted_border = weights[..., :, :, None] * border[:, None, :]
    cross = xp.moveaxis(
        _segment_sum(xp.moveaxis(weighted_border, -3, 0), design.codes, design.num_groups, xp), 0, -1
    )
    if ridge.ndim == 0:
        border_ridge = ridge
        group_ridge = ridge
    else:
        border_ridge = xp.take(ridge, design.border_indices, axis=-1)[..., :, None]
        group_ridge = xp.take(ridge, design.group_indices, axis=-1)
    block = block + border_ridge * xp.eye(design.num_border, dtype=dtype)
    diagonal = diagonal + group_ridge
    return BorderedInfo(block, cross, diagonal)


def weighted_information(design: BorderedDesign, weights, ridge=0.0) -> BorderedInfo:
    """Build ``Z'WZ + diag(ridge)`` blocks with no dense one-hot products.

    Weights have shape (..., cells, genes). Ridge is scalar or ends in the
    original column dimension; a (..., genes, columns) ridge is also accepted.
    Add a shared penalty only once when combining information from cell pools.
    """
    return _weighted_information(design, weights, ridge, jnp)


def weighted_information_numpy(design: BorderedDesign, weights, ridge=0.0) -> BorderedInfo:
    """NumPy equivalent; float64 stays float64 regardless of JAX settings."""
    return _weighted_information(design, weights, ridge, np)


def add_information(*infos: BorderedInfo) -> BorderedInfo:
    """Add information for disjoint row sets with identical column metadata."""
    if not infos:
        raise ValueError("At least one information block is required.")
    result = infos[0]
    for other in infos[1:]:
        result = BorderedInfo(*(a + b for a, b in zip(result, other)))
    return result


def _assemble_original(design, border_values, group_values, xp):
    shape = xp.broadcast_shapes(border_values.shape[:-1], group_values.shape[:-1])
    dtype = xp.result_type(border_values, group_values)
    out = xp.zeros(shape + (design.num_columns,), dtype=dtype)
    if xp is np:
        out[..., np.asarray(design.border_indices)] = border_values
        out[..., np.asarray(design.group_indices)] = group_values
        return out
    out = out.at[..., jnp.asarray(design.border_indices)].set(border_values)
    return out.at[..., jnp.asarray(design.group_indices)].set(group_values)


def _solve(design, info, rhs, xp):
    rhs = xp.asarray(rhs)
    if rhs.ndim < 2 or rhs.shape[-1] != design.num_columns:
        raise ValueError("The right-hand side must have shape (..., genes, original_columns).")
    border, cross, diagonal = (xp.asarray(value) for value in info)
    border_rhs = xp.take(rhs, design.border_indices, axis=-1)
    group_rhs = xp.take(rhs, design.group_indices, axis=-1)
    inverse_diagonal = 1.0 / diagonal
    scaled_cross = cross * inverse_diagonal[..., None, :]
    if design.num_border:
        schur = border - xp.einsum("...gdk,...gek->...gde", scaled_cross, cross)
        schur_rhs = border_rhs - xp.einsum("...gdk,...gk->...gd", scaled_cross, group_rhs)
        border_solution = xp.linalg.solve(schur, schur_rhs[..., None])[..., 0]
    else:
        shape = xp.broadcast_shapes(border_rhs.shape[:-1], diagonal.shape[:-1])
        border_solution = xp.zeros(shape + (0,), dtype=xp.result_type(rhs, diagonal))
    group_solution = inverse_diagonal * (
        group_rhs - xp.einsum("...gdk,...gd->...gk", cross, border_solution)
    )
    return _assemble_original(design, border_solution, group_solution, xp)


def solve(design: BorderedDesign, info: BorderedInfo, rhs):
    """Solve in original coordinates, broadcasting leading target/resample axes.

    An information shape (targets, 1, genes, ...) broadcasts against a RHS
    (targets, resamples, genes, columns). No additional regularizer is added.
    Singular unregularized blocks can return nonfinite values; callers choose
    whether their established dense fallback or a failure status is required.
    """
    return _solve(design, info, rhs, jnp)


def solve_numpy(design: BorderedDesign, info: BorderedInfo, rhs):
    """NumPy equivalent; NumPy's singular-matrix exception is not suppressed."""
    return _solve(design, info, rhs, np)


def _matmul(design, coefficients, xp):
    coefficients = xp.asarray(coefficients)
    if coefficients.ndim < 2 or coefficients.shape[-2] != design.num_columns:
        raise ValueError("Coefficients must have shape (..., original_columns, genes).")
    border_coefficients = xp.take(coefficients, design.border_indices, axis=-2)
    group_coefficients = xp.take(coefficients, design.group_indices, axis=-2)
    dummy = xp.zeros(group_coefficients.shape[:-2] + (1, coefficients.shape[-1]), dtype=coefficients.dtype)
    groups = xp.concatenate([group_coefficients, dummy], axis=-2)
    return xp.einsum("nd,...dg->...ng", xp.asarray(design.border), border_coefficients) + xp.take(
        groups, design.codes, axis=-2
    )


def matmul(design: BorderedDesign, coefficients):
    """Compute Z @ coefficients without materializing indicator columns."""
    return _matmul(design, coefficients, jnp)


def matmul_numpy(design: BorderedDesign, coefficients):
    return _matmul(design, coefficients, np)


def _transpose_dot(design, values, xp):
    values = xp.asarray(values)
    if values.ndim < 2 or values.shape[-2] != design.border.shape[0]:
        raise ValueError("Values must have shape (..., cells, genes) matching the design.")
    border_values = xp.einsum("nd,...ng->...gd", xp.asarray(design.border), values)
    group_values = xp.moveaxis(
        _segment_sum(xp.moveaxis(values, -2, 0), design.codes, design.num_groups, xp), 0, -1
    )
    return _assemble_original(design, border_values, group_values, xp)


def transpose_dot(design: BorderedDesign, values):
    """Compute (Z' @ values)' in shape (..., genes, original_columns)."""
    return _transpose_dot(design, values, jnp)


def transpose_dot_numpy(design: BorderedDesign, values):
    return _transpose_dot(design, values, np)
