"""Structured nuisance projections for the existing low-MOI score tests."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from perturbo._internal.bordered import (
    BorderedInfo,
    matmul,
    solve,
    transpose_dot,
    weighted_information,
)

_CUMULANT_WORKING_ELEMENTS = 4_000_000


@jax.jit
def score_from_indices(indices, residual, weight, design, information, nuisance_score):
    """Score assignments using target-specific bordered information blocks.

    Indices are (targets, assignments, selected cells); the coefficient axis
    remains in the original reference-coded order throughout the projection.
    """
    selected_weight = weight[indices]
    cross = jax.vmap(jax.vmap(lambda rows, w: transpose_dot(design.take(rows), w)))(
        indices, selected_weight
    )
    expanded = jax.tree.map(lambda block: block[:, None], information)
    direction = solve(design, expanded, cross)
    raw = selected_weight.sum(axis=2)
    efficient = raw - jnp.sum(cross * direction, axis=-1)
    corrected = residual[indices].sum(axis=2) - jnp.sum(
        nuisance_score[:, None] * direction, axis=-1
    )
    valid = efficient > 1e-12
    return jnp.where(valid, corrected / jnp.sqrt(jnp.where(valid, efficient, 1.0)), jnp.nan)


@jax.jit
def prepare_target_scores(indices, residual, weight, design, control_information, control_score):
    """Add each target's own rows to the shared control-side information."""
    own = jax.vmap(lambda rows: weighted_information(design.take(rows), weight[rows]))(indices)
    information = jax.tree.map(lambda own_block, control: own_block + control, own, control_information)
    nuisance_score = jax.vmap(lambda rows: transpose_dot(design.take(rows), residual[rows]))(indices)
    nuisance_score = nuisance_score + control_score.T
    observed = score_from_indices(
        indices[:, None], residual, weight, design, information, nuisance_score
    )[:, 0]
    return information, nuisance_score, observed


@partial(jax.jit, static_argnames=("num_targets",))
def segmented_information(design, weight, target_codes, *, num_targets):
    """Information from disjoint target rows, without target-by-q-by-q tensors."""
    border = jnp.asarray(design.border, dtype=weight.dtype)
    groups = design.num_groups
    joint = target_codes * (groups + 1) + design.codes

    def group_sum(values):
        return jax.ops.segment_sum(values, joint, num_segments=num_targets * (groups + 1)).reshape(
            num_targets, groups + 1, weight.shape[1]
        )[:, :groups].transpose(0, 2, 1)

    diagonal = group_sum(weight)
    cross = (
        jnp.stack([group_sum(weight * border[:, k, None]) for k in range(design.num_border)], axis=-2)
        if design.num_border else jnp.zeros((num_targets, weight.shape[1], 0, groups), dtype=weight.dtype)
    )
    curvature = jnp.stack([
        jnp.stack([
            jax.ops.segment_sum(
                weight * (border[:, k] * border[:, ell])[:, None],
                target_codes,
                num_segments=num_targets,
            )
            for ell in range(design.num_border)
        ], axis=-1)
        for k in range(design.num_border)
    ], axis=-2) if design.num_border else jnp.zeros((num_targets, weight.shape[1], 0, 0), dtype=weight.dtype)
    return BorderedInfo(curvature, cross, diagonal)


@partial(jax.jit, static_argnames=("num_targets",))
def segmented_transpose_dot(design, values, target_codes, *, num_targets):
    """Compute each target's original-coordinate nuisance score by grouping."""
    groups = design.num_groups
    joint = target_codes * (groups + 1) + design.codes
    group_score = jax.ops.segment_sum(
        values, joint, num_segments=num_targets * (groups + 1)
    ).reshape(num_targets, groups + 1, values.shape[1])[:, :groups].transpose(0, 2, 1)
    border_score = jnp.stack([
        jax.ops.segment_sum(
            values * design.border[:, k, None], target_codes, num_segments=num_targets
        )
        for k in range(design.num_border)
    ], axis=-1) if design.num_border else jnp.zeros((num_targets, values.shape[1], 0), dtype=values.dtype)
    out = jnp.zeros((num_targets, values.shape[1], design.num_columns), dtype=values.dtype)
    out = out.at[..., design.border_indices].set(border_score)
    return out.at[..., design.group_indices].set(group_score)


@jax.jit
def _row_shifts(design, coefficients, targets):
    """Evaluate a different target's coefficient vector for each design row."""
    columns = jnp.concatenate([jnp.asarray(design.group_indices), jnp.zeros(1, dtype=jnp.int32)])
    shift = coefficients[targets[:, None], jnp.arange(coefficients.shape[1])[None], columns[design.codes, None]]
    shift = jnp.where(design.codes[:, None] < design.num_groups, shift, 0.0)
    for k in range(design.num_border):
        shift = shift + coefficients[targets, :, design.border_indices[k]] * design.border[:, k, None]
    return shift


@jax.jit
def _correct_bordered_cumulants(design, coefficients, selection, bernoulli, third_weight,
                               weight, contribution, mean, variance, third):
    """Bound the target-by-cell-by-gene workspace and keep the existing screen."""
    n = weight.shape[0]
    if n == 0:
        return mean, variance, third
    width = min(n, max(1, _CUMULANT_WORKING_ELEMENTS // max(coefficients.shape[0] * weight.shape[1], 1)))
    blocks = (n + width - 1) // width

    def step(block, moments):
        mean, variance, third = moments
        start = jnp.minimum(block * width, n - width)
        rows = start + jnp.arange(width)
        valid = rows >= block * width
        shift = matmul(design.take(rows), coefficients.transpose(0, 2, 1))
        w = weight[rows].astype(coefficients.dtype)
        c = contribution[rows]
        delta = w[None] * shift
        p = selection[:, rows] * valid[None]
        v = bernoulli[:, rows] * valid[None]
        t = third_weight[:, rows] * valid[None]
        mean = mean - jnp.einsum("tn,tng->tg", p, delta)
        variance = variance + jnp.einsum("tn,tng->tg", v, delta * delta - 2.0 * c[None] * delta)
        # Match the dense screen's second-order correction. The exact SPA
        # subsequently uses fully corrected contributions, without truncation.
        third = third + jnp.einsum(
            "tn,tng->tg", t, 3.0 * c[None] * delta * delta - 3.0 * c[None] ** 2 * delta
        )
        return mean, variance, third

    return jax.lax.fori_loop(0, blocks, step, (mean, variance, third))


class BorderedPoolProjection:
    """Pool correction with K-1 batch coefficients and a small dense border."""

    @classmethod
    def build(cls, *, design, information, weight, num_targets, control_rows, own_rows,
              own_codes, all_rows, codes_all, own_contribution, target_valid):
        self = cls()
        self.design = design.astype(jnp.float64)
        self.control_design = self.design.take(np.asarray(control_rows))
        self.design_ext = self.design.pad_rows()
        weight = jnp.asarray(weight)
        genes = weight.shape[1]
        self.weight_ext = jnp.concatenate([weight, jnp.zeros((1, genes), dtype=weight.dtype)])
        self.control_weight = weight[jnp.asarray(control_rows)]
        info = jax.tree.map(lambda a: jnp.asarray(a, dtype=jnp.float64), information)
        self.e = jnp.zeros((num_targets, genes, design.num_columns), dtype=jnp.float64)
        order = np.argsort(own_codes, kind="stable")
        sorted_codes = np.asarray(own_codes)[order]
        starts = np.searchsorted(sorted_codes, np.arange(num_targets + 1))
        own_rows = np.asarray(own_rows)
        # Bound the target-specific information; its large dimension is K,
        # never K squared. Cell intermediates are independently row-chunked.
        per_target = genes * (design.num_border ** 2 + design.num_border * design.num_groups + design.num_groups) * 8
        batch = min(max(num_targets, 1), max(1, 128_000_000 // max(per_target, 1)))
        row_chunk = max(1, 4_000_000 // max(genes * max(design.num_border, 1), 1))
        for first in range(0, num_targets, batch):
            last = min(first + batch, num_targets)
            positions = order[starts[first]:starts[last]]
            accumulated = jax.tree.map(lambda a: jnp.zeros((last - first,) + a.shape, dtype=a.dtype), info)
            rhs = jnp.zeros((last - first, genes, design.num_columns), dtype=jnp.float64)
            for begin in range(0, len(positions), row_chunk):
                part = positions[begin:begin + row_chunk]
                rows = own_rows[part]
                codes = jnp.asarray(np.asarray(own_codes)[part] - first)
                subset = self.design.take(rows)
                addition = segmented_information(subset, weight[rows].astype(jnp.float64), codes, num_targets=last-first)
                accumulated = jax.tree.map(jnp.add, accumulated, addition)
                rhs = rhs + segmented_transpose_dot(subset, own_contribution[part], codes, num_targets=last-first)
            pooled = jax.tree.map(jnp.add, accumulated, info)
            fitted = solve(self.design, pooled, rhs)
            fitted = jnp.where(jnp.asarray(target_valid[first:last])[:, None, None], fitted, 0.0)
            self.e = self.e.at[first:last].set(jnp.nan_to_num(fitted, nan=0.0, posinf=0.0, neginf=0.0))
        corrected = []
        for first in range(0, len(own_rows), row_chunk):
            last = min(first + row_chunk, len(own_rows))
            rows = own_rows[first:last]
            shift = _row_shifts(self.design.take(rows), self.e, jnp.asarray(own_codes[first:last]))
            corrected.append(own_contribution[first:last] - weight[rows] * shift)
        self.corrected_own = jnp.concatenate(corrected) if corrected else jnp.zeros((0, genes), dtype=jnp.float64)
        self.observed_shift = jnp.zeros((num_targets, genes), dtype=jnp.float64)
        for first in range(0, len(all_rows), row_chunk):
            rows = np.asarray(all_rows)[first:first + row_chunk]
            targets = jnp.asarray(codes_all[first:first + row_chunk])
            shift = _row_shifts(self.design.take(rows), self.e, targets)
            self.observed_shift = self.observed_shift + jax.ops.segment_sum(
                weight[rows] * shift, targets, num_segments=num_targets
            )
        return self

    def correct_control_cumulants(self, targets, selection, bernoulli, third_weight, control_contribution,
                                  control_square, mean, variance, third):
        del control_square
        coefficients = self.e[targets]
        parts = []
        for start in range(0, len(coefficients), 32):
            stop = min(start + 32, len(coefficients))
            parts.append(_correct_bordered_cumulants(
                self.control_design, coefficients[start:stop], selection[start:stop],
                bernoulli[start:stop], third_weight[start:stop], self.control_weight,
                control_contribution, mean[start:stop], variance[start:stop], third[start:stop],
            ))
        return tuple(jnp.concatenate([part[k] for part in parts]) for k in range(3))

    def correct_blocks(self, t_dev, g_dev, rows, block_controls, block_own):
        coefficients = self.e[t_dev, g_dev]
        control_shift = matmul(self.control_design, coefficients.T)
        columns = jnp.concatenate([jnp.asarray(self.design.group_indices), jnp.zeros(1, dtype=jnp.int32)])
        codes = self.design_ext.codes[rows]
        own_shift = coefficients[jnp.arange(len(t_dev))[:, None], columns[codes]]
        own_shift = jnp.where(codes < self.design.num_groups, own_shift, 0.0)
        for k in range(self.design.num_border):
            own_shift = own_shift + self.design_ext.border[rows, k] * coefficients[:, self.design.border_indices[k], None]
        control_weight = self.control_weight[:, g_dev]
        own_weight = self.weight_ext[rows, g_dev[:, None]].T
        return block_controls - control_weight * control_shift, block_own - own_weight * own_shift.T
