"""Score-test algebra for a ``[intercept, one-hot batch]`` nuisance design.

``prepare_control_fitted_batch_covariate`` emits reference-coded indicators, so
a cell belongs to exactly one batch and the nuisance design is

    Z = [1, onehot(K)]

which makes the per-gene curvature an *arrow* matrix::

    A = Z'WZ = [[a, s'],
                [s, diag(s)]]

with ``s_k`` the weight sum of batch ``k`` and ``a`` the total. The diagonal
block is ``s`` itself rather than a separate quantity, because a cell
contributes to exactly one batch: its group-``k`` diagonal entry and its
group-``k`` intercept cross-term are the same sum. Every off-diagonal entry
among the batch columns is *structurally* zero, not merely small.

Two consequences, and they are what this module exists for.

``A x = b`` has a closed form. From rows ``2..K+1``, ``s_k x_0 + s_k x_k = b_k``,
so ``x_k = b_k / s_k - x_0``; substituting into row one and using
``a - sum_k s_k = s_ref`` (the dropped level's weight sum) gives

    x_0 = (b_0 - sum_k b_k) / s_ref

so no factorization and no ``(genes, K, K)`` tensor is ever needed.

The selected-cell cross term ``Z_S' W_S 1`` is a *segment sum*. The dense path
gathers a width-``genes * K`` row per selected cell to compute it; here only the
weight is gathered and scattered into ``K`` bins. At genome-wide scale - 8,248
genes and 267 columns from ``gem_group`` - that is the difference between
gathering ``batch x cells x genes x 267`` floats and ``batch x cells x genes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "ArrowNuisance",
    "batch_codes_from_one_hot",
    "bordered_efficient_score",
    "control_arrow_segments",
    "prepare_arrow_target_scores",
    "split_bordered_design",
    "as_arrow_nuisance",
    "arrow_efficient_score",
    "is_intercept_and_one_hot",
]


def is_intercept_and_one_hot(nuisance_design: np.ndarray, *, tolerance: float = 0.0) -> bool:
    """Whether ``nuisance_design`` is ``[intercept, reference-coded one-hot]``.

    Checked rather than assumed: the structured path is only equivalent to the
    dense one for this exact shape, and a design that merely looks batch-like -
    an interaction, a continuous covariate appended - would silently give a
    different answer.
    """

    design = np.asarray(nuisance_design)
    if design.ndim != 2 or design.shape[1] < 2:
        return False
    if not np.all(np.abs(design[:, 0] - 1.0) <= tolerance):
        return False
    batch_block = design[:, 1:]
    if not np.all((np.abs(batch_block) <= tolerance) | (np.abs(batch_block - 1.0) <= tolerance)):
        return False
    return bool(np.all((batch_block > 0.5).sum(axis=1) <= 1))


@dataclass(frozen=True)
class ArrowNuisance:
    """A one-hot batch design reduced to what the score test actually needs.

    ``codes`` maps each cell to ``0`` for the dropped reference level or
    ``1..K`` for a batch column. ``group_weight_sums`` and
    ``reference_weight_sum`` are the null curvature's segment sums over *every*
    design cell, per gene - the arrow matrix in full, held as ``K + 1`` numbers
    per gene instead of ``(K + 1)^2``.
    """

    codes: np.ndarray
    group_weight_sums: np.ndarray
    reference_weight_sum: np.ndarray
    num_groups: int

    @property
    def num_nuisance(self) -> int:
        return self.num_groups + 1


def as_arrow_nuisance(
    nuisance_design: np.ndarray,
    observation_weight: np.ndarray,
) -> ArrowNuisance | None:
    """Reduce a one-hot design plus weights to :class:`ArrowNuisance`.

    Returns ``None`` when the design is not ``[intercept, one-hot]``, so callers
    can fall back to the dense path rather than silently changing the model.
    """

    design = np.asarray(nuisance_design)
    if not is_intercept_and_one_hot(design):
        return None
    weight = np.asarray(observation_weight, dtype=np.float64)
    batch_block = design[:, 1:] > 0.5
    num_groups = int(batch_block.shape[1])
    # 0 for the reference level, 1..K for the indicator columns.
    codes = np.zeros(design.shape[0], dtype=np.int32)
    rows, columns = np.nonzero(batch_block)
    codes[rows] = columns + 1

    segments = np.zeros((num_groups + 1, weight.shape[1]), dtype=np.float64)
    np.add.at(segments, codes, weight)
    return ArrowNuisance(
        codes=codes,
        group_weight_sums=segments[1:],
        reference_weight_sum=segments[0],
        num_groups=num_groups,
    )


@jax.jit
def arrow_efficient_score(
    selected_indices: jnp.ndarray,
    score_residual: jnp.ndarray,
    observation_weight: jnp.ndarray,
    codes: jnp.ndarray,
    group_weight_sums: jnp.ndarray,
    reference_weight_sum: jnp.ndarray,
    nuisance_score: jnp.ndarray,
) -> jnp.ndarray:
    """Efficient score z-statistics, arrow-structured.

    Mirrors :func:`batched_efficient_score_from_indices` exactly - the same
    ``(targets, assignments, selections)`` indices in, the same
    ``(targets, assignments, genes)`` statistics out - but never forms the
    nuisance cross tensor or the ``(targets, genes, K, K)`` inverse.

    The curvature is target-specific, as in the dense path: a target adds its
    own cells to the control pool, so ``group_weight_sums`` is
    ``(targets, K, genes)`` and ``reference_weight_sum`` is
    ``(targets, genes)``. ``nuisance_score`` is ``(targets, K + 1, genes)``
    with row 0 the intercept.
    """

    num_targets, num_assignments = selected_indices.shape[0], selected_indices.shape[1]
    num_genes = score_residual.shape[1]
    num_groups = group_weight_sums.shape[1]

    score = jnp.take(score_residual, selected_indices, axis=0).sum(axis=2)
    selected_weight = jnp.take(observation_weight, selected_indices, axis=0)
    raw_information = selected_weight.sum(axis=2)

    # The cross term as a segment sum. Only the weight is gathered - shape
    # (targets, assignments, selections, genes) - and scattered into K + 1 bins
    # by each cell's batch. The dense path instead gathers that same tensor
    # widened by a factor of K + 1, which is the entire cost being removed.
    selected_codes = jnp.take(codes, selected_indices)
    bins = jnp.zeros(
        (num_targets, num_assignments, num_groups + 1, num_genes),
        dtype=selected_weight.dtype,
    )
    bins = bins.at[
        jnp.arange(num_targets)[:, None, None],
        jnp.arange(num_assignments)[None, :, None],
        selected_codes,
    ].add(selected_weight)
    cross_groups = bins[:, :, 1:, :]

    # A x = c in closed form. Guard both denominators: a batch with no weight
    # for a gene contributes nothing to the projection and must not divide.
    safe_reference = jnp.where(reference_weight_sum > 0, reference_weight_sum, 1.0)
    safe_groups = jnp.where(group_weight_sums > 0, group_weight_sums, 1.0)
    x_intercept = (raw_information - cross_groups.sum(axis=2)) / safe_reference[:, None, :]
    x_groups = jnp.where(
        group_weight_sums[:, None] > 0,
        cross_groups / safe_groups[:, None] - x_intercept[:, :, None, :],
        0.0,
    )

    projected = raw_information * x_intercept + (cross_groups * x_groups).sum(axis=2)
    correction = nuisance_score[:, 0][:, None, :] * x_intercept + (
        nuisance_score[:, 1:][:, None] * x_groups
    ).sum(axis=2)

    efficient_information = raw_information - projected
    valid = efficient_information > 1e-12
    return jnp.where(
        valid,
        (score - correction) / jnp.sqrt(jnp.where(valid, efficient_information, 1.0)),
        jnp.nan,
    )


def batch_codes_from_one_hot(covariates: np.ndarray | None) -> np.ndarray | None:
    """Recover full one-hot batch codes from a reference-coded covariate block.

    ``prepare_control_fitted_batch_covariate`` drops one level, so a K-level
    batch arrives as K - 1 indicator columns with the reference level encoded as
    all zeros. The categorical score kernel wants the *undropped* coding - one
    code per level, including the reference - because with every level present
    ``Z'WZ`` is diagonal rather than merely arrow-shaped, and the two
    parameterizations span the same column space so the projection, and
    therefore the score statistic, is identical.

    Returns ``None`` when the block is not reference-coded indicators, so the
    caller keeps the dense path rather than silently changing the model.
    """

    if covariates is None:
        return None
    block = np.asarray(covariates)
    if block.ndim != 2 or block.shape[1] < 1:
        return None
    if not np.all((block == 0.0) | (block == 1.0)):
        return None
    active = (block > 0.5).sum(axis=1)
    if np.any(active > 1):
        return None
    codes = np.zeros(block.shape[0], dtype=np.int32)
    rows, columns = np.nonzero(block > 0.5)
    codes[rows] = columns + 1
    return codes


def split_bordered_design(nuisance_design: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Split a design into ``(border columns, one-hot codes)``, or ``None``.

    The high-MOI nuisance is a continuous border - MOI, library size - alongside
    a categorical batch, so its curvature is *bordered diagonal* rather than
    diagonal: only the batch block is diagonal, and the border couples to it.

    The batch block is identified as a maximal set of 0/1 columns that are
    mutually exclusive across rows, which is exactly the condition making
    ``H'WH`` diagonal. Everything else becomes the border.

    Returns ``None`` unless at least two such columns exist, since with fewer
    there is nothing to exploit.
    """

    design = np.asarray(nuisance_design, dtype=np.float64)
    if design.ndim != 2:
        return None
    binary = np.all((design == 0.0) | (design == 1.0), axis=0)
    constant = np.all(design == 1.0, axis=0)
    candidates = np.flatnonzero(binary & ~constant)
    if candidates.size < 2:
        return None
    block = design[:, candidates]
    if np.any(block.sum(axis=1) > 1.0):
        return None

    # Undropped coding: give the rows in no candidate column their own level, so
    # the block's columns sum to the intercept and span it. The border must then
    # NOT carry an intercept of its own, or the design is collinear.
    codes = np.full(design.shape[0], candidates.size, dtype=np.int32)
    rows, columns = np.nonzero(block > 0.5)
    codes[rows] = columns
    border_columns = np.setdiff1d(np.arange(design.shape[1]), candidates)
    border = design[:, border_columns]
    keep = ~np.all(border == 1.0, axis=0) if border.size else np.zeros(0, dtype=bool)
    return border[:, keep].astype(np.float32), codes


@partial(jax.jit, static_argnames=("num_groups",))
def bordered_efficient_score(
    selected_indices: jnp.ndarray,
    score_residual: jnp.ndarray,
    observation_weight: jnp.ndarray,
    border_design: jnp.ndarray,
    group_codes: jnp.ndarray,
    schur_inverse: jnp.ndarray,
    border_group_cross: jnp.ndarray,
    group_weight_sums: jnp.ndarray,
    nuisance_score_border: jnp.ndarray,
    nuisance_score_groups: jnp.ndarray,
    *,
    num_groups: int,
) -> jnp.ndarray:
    """Efficient score for a continuous border plus a categorical batch.

    ``A = [[D'WD, D'WH], [H'WD, diag(s)]]`` is solved by Schur complement on the
    border, which is ``d x d`` and shared across every element and resample, so
    the per-statistic work is ``O(d^2 + K)`` rather than a ``(d + K)^3`` solve
    against a ``(genes, d + K, d + K)`` tensor.

    ``schur_inverse`` is ``(genes, d, d)``, the inverse of
    ``D'WD - (D'WH) diag(s)^-1 (H'WD)``; the remaining shared pieces are
    ``border_group_cross`` ``(genes, d, K)`` and ``group_weight_sums``
    ``(genes, K)``. Nuisance scores are ``(d, genes)`` and ``(K, genes)``.
    """

    num_targets, num_assignments = selected_indices.shape[0], selected_indices.shape[1]
    num_genes = score_residual.shape[1]

    selected_weight = jnp.take(observation_weight, selected_indices, axis=0)
    score = jnp.take(score_residual, selected_indices, axis=0).sum(axis=2)
    raw_information = selected_weight.sum(axis=2)

    selected_border = jnp.take(border_design, selected_indices, axis=0)
    cross_border = jnp.einsum("tbsg,tbsj->tbjg", selected_weight, selected_border)

    selected_codes = jnp.take(group_codes, selected_indices)
    bins = jnp.zeros((num_targets, num_assignments, num_groups, num_genes), dtype=selected_weight.dtype)
    cross_groups = bins.at[
        jnp.arange(num_targets)[:, None, None],
        jnp.arange(num_assignments)[None, :, None],
        selected_codes,
    ].add(selected_weight)

    inverse_groups = jnp.where(group_weight_sums > 0, 1.0 / jnp.where(group_weight_sums > 0, group_weight_sums, 1.0), 0.0)
    scaled_cross = border_group_cross * inverse_groups[:, None, :]

    rhs = cross_border - jnp.einsum("gjk,tbkg->tbjg", scaled_cross, cross_groups)
    x_border = jnp.einsum("gij,tbjg->tbig", schur_inverse, rhs)
    x_groups = inverse_groups.T[None, None] * (
        cross_groups - jnp.einsum("gjk,tbjg->tbkg", border_group_cross, x_border)
    )

    projected = jnp.einsum("tbjg,tbjg->tbg", cross_border, x_border) + jnp.einsum(
        "tbkg,tbkg->tbg", cross_groups, x_groups
    )
    correction = jnp.einsum("jg,tbjg->tbg", nuisance_score_border, x_border) + jnp.einsum(
        "kg,tbkg->tbg", nuisance_score_groups, x_groups
    )

    efficient_information = raw_information - projected
    valid = efficient_information > 1e-12
    return jnp.where(
        valid,
        (score - correction) / jnp.sqrt(jnp.where(valid, efficient_information, 1.0)),
        jnp.nan,
    )


@partial(jax.jit, static_argnames=("num_groups",))
def prepare_arrow_target_scores(
    target_indices: jnp.ndarray,
    score_residual: jnp.ndarray,
    observation_weight: jnp.ndarray,
    codes: jnp.ndarray,
    control_group_sums: jnp.ndarray,
    control_reference_sum: jnp.ndarray,
    control_nuisance_score: jnp.ndarray,
    *,
    num_groups: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-target arrow curvature and score, adding a target's cells to the pool.

    The arrow analogue of
    :func:`~perturbo._internal.jax_kernels.prepare_control_only_target_scores`.
    It exists because the categorical kernel cannot serve a caller that supplies
    its own nuisance coefficients: that kernel parametrizes the nuisance as one
    intercept per batch and fits it, whereas the production CRT reuses a stage
    one fit on a reference-coded design. The arrow form needs no
    reparameterization - it takes the curvature as segment sums and leaves the
    supplied coefficients, and hence the residual and weight, untouched.

    Returns ``(group_weight_sums, reference_weight_sum, nuisance_score,
    observed)`` shaped ``(targets, K, genes)``, ``(targets, genes)``,
    ``(targets, K + 1, genes)`` and ``(targets, genes)``.
    """

    num_targets = target_indices.shape[0]
    num_genes = score_residual.shape[1]

    target_weight = jnp.take(observation_weight, target_indices, axis=0)
    target_residual = jnp.take(score_residual, target_indices, axis=0)
    target_codes = jnp.take(codes, target_indices)
    rows = jnp.arange(num_targets)[:, None]

    binned_weight = jnp.zeros((num_targets, num_groups + 1, num_genes), dtype=target_weight.dtype)
    binned_weight = binned_weight.at[rows, target_codes].add(target_weight)
    binned_residual = jnp.zeros((num_targets, num_groups + 1, num_genes), dtype=target_residual.dtype)
    binned_residual = binned_residual.at[rows, target_codes].add(target_residual)

    group_sums = control_group_sums[None] + binned_weight[:, 1:]
    reference_sum = control_reference_sum[None] + binned_weight[:, 0]
    # Row 0 is the intercept, which spans every level including the dropped
    # one, so it takes the total rather than a single bin.
    target_score = jnp.concatenate(
        [binned_residual.sum(axis=1)[:, None, :], binned_residual[:, 1:]], axis=1
    )
    nuisance_score = control_nuisance_score[None] + target_score

    observed = arrow_efficient_score(
        target_indices[:, None, :],
        score_residual,
        observation_weight,
        codes,
        group_sums,
        reference_sum,
        nuisance_score,
    )[:, 0, :]
    return group_sums, reference_sum, nuisance_score, observed


def control_arrow_segments(
    codes: np.ndarray,
    observation_weight: np.ndarray,
    score_residual: np.ndarray,
    control_mask: np.ndarray,
    *,
    num_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Control-pool arrow segments: group sums, reference sum, nuisance score."""

    weight = np.asarray(observation_weight)
    residual = np.asarray(score_residual)
    selected = np.asarray(control_mask, dtype=bool)
    codes_selected = np.asarray(codes)[selected]

    binned_weight = np.zeros((num_groups + 1, weight.shape[1]), dtype=np.float64)
    np.add.at(binned_weight, codes_selected, weight[selected])
    binned_residual = np.zeros((num_groups + 1, residual.shape[1]), dtype=np.float64)
    np.add.at(binned_residual, codes_selected, residual[selected])
    nuisance_score = np.concatenate(
        [binned_residual.sum(axis=0)[None, :], binned_residual[1:]], axis=0
    )
    return binned_weight[1:], binned_weight[0], nuisance_score
