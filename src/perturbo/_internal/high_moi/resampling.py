"""Exact stratified subset sampling for the high-MOI CRT.

The low-MOI resampler ranks random keys over the whole candidate pool, which is
cheap there because a pair's pool is the controls plus one target's cells. In
high MOI the pool is every cell, so that approach costs
``resamples * cells`` work per element - on the order of 10^10 draws for a
pilot screen and far worse at scale - purely to select the one percent of cells
an element occupies.

Floyd's algorithm draws a uniform ``k``-subset in ``O(k)`` regardless of pool
size. Vectorizing it across resamples turns the per-element cost into
``resamples * k``, and the only structure it needs is a seen-flag per candidate
which is cleared by unsetting the drawn entries rather than by rezeroing.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def sample_distinct_indices(
    candidates: np.ndarray,
    *,
    num_selected: int,
    num_resamples: int,
    rng: np.random.Generator,
    row_chunk_size: int = 512,
) -> np.ndarray:
    """Draw uniform ``num_selected``-subsets of ``candidates`` per resample.

    Returns ``(num_resamples, num_selected)`` values taken from ``candidates``.
    Each row is a uniformly random subset without replacement, drawn by Floyd's
    algorithm so the cost does not scale with the candidate-pool size.
    """

    pool = np.asarray(candidates).reshape(-1)
    pool_size = pool.size
    if num_selected < 0 or num_selected > pool_size:
        raise ValueError("num_selected must be between 0 and the candidate count.")
    if num_resamples < 1:
        raise ValueError("num_resamples must be positive.")
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be positive.")
    if num_selected == 0:
        return np.empty((num_resamples, 0), dtype=pool.dtype)
    if num_selected == pool_size:
        return np.broadcast_to(pool, (num_resamples, pool_size)).copy()

    out = np.empty((num_resamples, num_selected), dtype=np.int64)
    seen = np.zeros((min(row_chunk_size, num_resamples), pool_size), dtype=bool)
    for start in range(0, num_resamples, row_chunk_size):
        stop = min(start + row_chunk_size, num_resamples)
        rows = stop - start
        row_ids = np.arange(rows)
        block = out[start:stop]
        for column, bound in enumerate(range(pool_size - num_selected, pool_size)):
            # Floyd: propose uniformly from the growing prefix; on a collision
            # take the newly exposed slot, which is provably still uniform.
            proposal = rng.integers(0, bound + 1, size=rows)
            collided = seen[row_ids, proposal]
            chosen = np.where(collided, bound, proposal)
            seen[row_ids, chosen] = True
            block[:, column] = chosen
        # Clear only what was set, so the next chunk does not pay for the pool.
        seen[row_ids[:, None], block] = False
    return pool[out]


def sample_stratified_indices(
    observed_assignment: np.ndarray,
    *,
    num_resamples: int,
    strata: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    row_chunk_size: int = 512,
) -> np.ndarray:
    """Resample an indicator, preserving the selected count within each stratum.

    Equivalent in distribution to permuting ``observed_assignment`` inside every
    stratum and reporting the selected positions, but it never materializes the
    permuted vector.
    """

    assignment = np.asarray(observed_assignment, dtype=np.int8).reshape(-1)
    if assignment.size == 0 or np.any((assignment != 0) & (assignment != 1)):
        raise ValueError("observed_assignment must be a non-empty binary vector.")
    if rng is None:
        rng = np.random.default_rng(0)
    if strata is None:
        stratum_values = np.zeros(assignment.size, dtype=np.int64)
    else:
        stratum_values = np.asarray(strata).reshape(-1)
        if stratum_values.shape != assignment.shape:
            raise ValueError("strata must contain one value per assignment.")

    unique_strata, stratum_codes = np.unique(stratum_values, return_inverse=True)
    blocks: list[np.ndarray] = []
    for stratum in range(unique_strata.size):
        members = np.flatnonzero(stratum_codes == stratum)
        selected = int(assignment[members].sum())
        if selected == 0:
            continue
        blocks.append(
            sample_distinct_indices(
                members,
                num_selected=selected,
                num_resamples=num_resamples,
                rng=rng,
                row_chunk_size=row_chunk_size,
            )
        )
    if not blocks:
        return np.empty((num_resamples, 0), dtype=np.int64)
    return np.concatenate(blocks, axis=1)


@partial(jax.jit, static_argnames=("max_iterations",))
def _logistic_irls_orthonormal(
    indicators: jnp.ndarray,
    basis: jnp.ndarray,
    jitter: jnp.ndarray,
    eta_clip: jnp.ndarray,
    *,
    max_iterations: int,
) -> jnp.ndarray:
    """IRLS for a batch of logistic fits sharing one orthonormal design basis.

    Returns coefficients in that basis rather than fitted probabilities.
    """

    num_features = basis.shape[1]
    eye = jnp.eye(num_features, dtype=basis.dtype)

    def step(coef, _):
        probability = jax.nn.sigmoid(jnp.clip(coef @ basis.T, -eta_clip, eta_clip))
        weight = jnp.clip(probability * (1.0 - probability), 1e-9, None)
        gradient = (indicators - probability) @ basis
        curvature = jnp.einsum("nk,en,nl->ekl", basis, weight, basis) + jitter * eye
        return coef + jnp.linalg.solve(curvature, gradient[..., None])[..., 0], None

    coef, _ = jax.lax.scan(
        step,
        jnp.zeros((indicators.shape[0], num_features), dtype=basis.dtype),
        None,
        length=max_iterations,
    )
    # Coefficients in the orthonormal basis, not fitted values. The linear
    # predictor is ``clip(coef @ basis.T, +/-eta_clip)``; callers that want it
    # for every cell form it themselves, and callers that only need it one
    # element at a time - the propensity saddlepoint - keep the ``(elements, q)``
    # coefficients instead, which is 5.4 GB smaller at genome-wide scale.
    return coef


def fit_propensity_probabilities(
    indicators: np.ndarray | jnp.ndarray,
    design: np.ndarray | jnp.ndarray,
    *,
    max_iterations: int = 25,
    jitter: float = 1e-6,
    eta_clip: float = 30.0,
) -> jnp.ndarray:
    """Fitted values of a logistic regression of each indicator on ``design``.

    One unpenalized MLE per row of ``indicators``, all sharing the same
    covariate matrix. ``jitter`` only conditions the linear solve; it is not a
    penalty, so this is the plain MLE ``glm.fit`` would return rather than a
    MAP under some prior. That distinction matters: the intercept's score
    equation is what makes ``sum(p)`` reproduce the observed selected count,
    and a penalty would quietly break it.

    The fit runs in an orthonormal basis for the design's column space, taken
    by QR. Fitted probabilities are invariant to any full-rank linear
    reparameterization, so this changes nothing statistically while making
    ``Q'WQ`` well conditioned - which is what lets the whole thing run in
    float32 without the Hessian solve degrading.

    ``eta_clip`` bounds the linear predictor, standing in for the step-halving
    a production GLM routine does; without it a separable element sends the
    coefficients to infinity and the IRLS weights underflow.
    """

    return jax.nn.sigmoid(
        fit_propensity_logits(
            indicators,
            design,
            max_iterations=max_iterations,
            jitter=jitter,
            eta_clip=eta_clip,
        )
    )


def fit_propensity_logits(
    indicators: np.ndarray | jnp.ndarray,
    design: np.ndarray | jnp.ndarray,
    *,
    max_iterations: int = 25,
    jitter: float = 1e-6,
    eta_clip: float = 30.0,
) -> jnp.ndarray:
    """Linear predictors of the same fits :func:`fit_propensity_probabilities` returns.

    ``sigmoid`` of this is that function exactly. It exists because the
    propensity saddlepoint's cumulant generating function is naturally written
    in the logit,

        K(t) = sum_i [softplus(eta_i + t c_i) - softplus(eta_i)],

    and a probability that has saturated to 1.0 in float32 carries no eta at
    all. Working in the logit keeps the whole clipped range representable.
    """

    coefficients, basis = fit_propensity_coefficients(
        indicators,
        design,
        max_iterations=max_iterations,
        jitter=jitter,
        eta_clip=eta_clip,
    )
    return propensity_logits_from_coefficients(coefficients, basis, eta_clip=eta_clip)


def fit_propensity_coefficients(
    indicators: np.ndarray | jnp.ndarray,
    design: np.ndarray | jnp.ndarray,
    *,
    max_iterations: int = 25,
    jitter: float = 1e-6,
    eta_clip: float = 30.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The same fits, returned as ``(coefficients, basis)`` in an orthonormal basis.

    ``propensity_logits_from_coefficients`` expands these back to per-cell
    linear predictors, and ``sigmoid`` of that is
    :func:`fit_propensity_probabilities` exactly.

    This form exists for the propensity saddlepoint, whose cumulant generating
    function is naturally written in the logit,

        K(t) = sum_i [softplus(eta_i + t c_i) - softplus(eta_i)],

    and which needs one element's ``eta`` at a time. Holding the expanded
    ``(elements, cells)`` predictors instead would be 5.4 GB on the
    genome-wide screen, against about 160 kB for the coefficients.

    The logit is also the only representable form at the clip: ``eta_clip``
    runs to 30, and ``sigmoid(30) = 1 - 9.4e-14`` rounds to exactly 1.0 in
    float32, whose logit is infinite.
    """

    y = jnp.atleast_2d(jnp.asarray(indicators, dtype=jnp.float32))
    Z = jnp.asarray(design, dtype=jnp.float32)
    if y.shape[1] != Z.shape[0]:
        raise ValueError("indicators and design disagree on the cell count.")
    basis, _ = jnp.linalg.qr(Z)
    coefficients = _logistic_irls_orthonormal(
        y,
        basis,
        jnp.asarray(jitter, dtype=jnp.float32),
        jnp.asarray(eta_clip, dtype=jnp.float32),
        max_iterations=max_iterations,
    )
    return coefficients, basis


def propensity_logits_from_coefficients(
    coefficients: np.ndarray | jnp.ndarray,
    basis: np.ndarray | jnp.ndarray,
    *,
    eta_clip: float = 30.0,
) -> jnp.ndarray:
    """Per-cell linear predictors from :func:`fit_propensity_coefficients`."""

    coefficients = jnp.atleast_2d(jnp.asarray(coefficients))
    basis = jnp.asarray(basis, dtype=coefficients.dtype)
    return jnp.clip(coefficients @ basis.T, -eta_clip, eta_clip)


WIDTH_BUCKET = 64
"""Granularity of the padded resample width.

Padding to a bucket rather than to the exact maximum keeps the number of
distinct compiled shapes small. The extra columns hold ``pad_value``, which
points at a zero-weight, zero-residual cell, so a wider pad costs a little
gather bandwidth and changes no result.
"""


@partial(jax.jit, static_argnames=("width", "pad_value"))
def _extract_selected(mask: jnp.ndarray, width: int, pad_value: int) -> jnp.ndarray:
    """Positions of the set entries in each row, padded to ``width``."""

    return jax.vmap(lambda row: jnp.nonzero(row, size=width, fill_value=pad_value)[0])(mask)


def sample_propensity_indices(
    propensity: np.ndarray | jnp.ndarray,
    *,
    num_resamples: int,
    key: jax.Array,
    pad_value: int,
    resample_chunk_size: int = 256,
) -> np.ndarray:
    """Resample an indicator as independent Bernoulli draws from ``propensity``.

    This is the model-X conditional randomization test as SCEPTRE runs it: each
    cell's synthetic assignment is drawn from its own fitted selection
    probability, so the *number* of selected cells varies between resamples
    rather than being held at the observed count.

    That is the substantive difference from
    :func:`sample_stratified_indices`, which fixes the count within each
    stratum. Fixing it is exchangeable only if the selection probability is
    constant inside a stratum; where it varies continuously - with guide load
    or sequencing depth - the permutation null is too narrow and the p-values
    run anti-conservative.

    Rows are padded to the widest draw with ``pad_value``, which the caller
    should point at a zero-weight, zero-residual dummy cell so the padding
    contributes nothing to the score.
    """

    p = jnp.asarray(propensity, dtype=jnp.float32).reshape(-1)
    if num_resamples < 1:
        raise ValueError("num_resamples must be positive.")
    if resample_chunk_size < 1:
        raise ValueError("resample_chunk_size must be positive.")

    # Draw every mask first so the padded width is the true maximum over the
    # whole set rather than a per-chunk one; jnp.nonzero needs that width to be
    # a compile-time constant, so it cannot be discovered lazily.
    masks: list[jnp.ndarray] = []
    counts: list[np.ndarray] = []
    for index, start in enumerate(range(0, num_resamples, resample_chunk_size)):
        rows = min(resample_chunk_size, num_resamples - start)
        drawn = jax.random.bernoulli(jax.random.fold_in(key, index), p, (rows, p.size))
        masks.append(drawn)
        counts.append(np.asarray(drawn.sum(axis=1)))

    # Round the width up so elements of similar size share one compiled
    # extract. The width is a static argument - it has to be, jnp.nonzero needs
    # a concrete output size - so an exact per-element width would recompile
    # once per element, which on a GPU costs seconds each and dwarfs the draw.
    exact = int(np.concatenate(counts).max())
    width = int(np.ceil(exact / WIDTH_BUCKET) * WIDTH_BUCKET)
    return np.concatenate(
        [np.asarray(_extract_selected(mask, width, pad_value), dtype=np.int32) for mask in masks]
    )


def sample_propensity_indices_transposed(
    propensity: np.ndarray | jnp.ndarray,
    *,
    num_resamples: int,
    key: jax.Array,
    pad_value: int,
    max_repair_rounds: int = 12,
) -> np.ndarray:
    """Bernoulli propensity draws, generated per cell instead of per resample.

    Same distribution as :func:`sample_propensity_indices`, reached by
    transposing the loop. Drawing a mask over every cell for every resample
    costs ``O(resamples * cells)`` to produce ``O(resamples * selected)``
    successes, and an element occupies well under one percent of cells - so
    almost all of that work lands on failures.

    Instead, ask each cell how many of the resamples select it:
    ``M_j ~ Binomial(resamples, p_j)``, which is one draw per cell rather than
    ``resamples`` of them. Cell ``j`` is then scattered into ``M_j`` distinct
    resamples. Total work falls to ``O(cells + resamples * selected)``.

    The scatter draws slots with replacement and redraws collisions, because a
    variable-size without-replacement draw per cell does not vectorize. A cell
    landing in ``M_j`` of ``resamples`` slots collides with probability about
    ``M_j (M_j - 1) / (2 * resamples)`` per round, so a handful of rounds drives
    the residual far below Monte Carlo error; ``max_repair_rounds`` bounds the
    work, and any duplicate still standing would cost that cell one selection.
    """

    p = jnp.asarray(propensity, dtype=jnp.float32).reshape(-1)
    num_cells = int(p.size)
    if num_resamples < 1:
        raise ValueError("num_resamples must be positive.")

    count_key, slot_key = jax.random.split(key)
    counts = jax.random.binomial(count_key, num_resamples, p).astype(jnp.int32)
    total = int(counts.sum())
    if total == 0:
        return np.full((num_resamples, 0), pad_value, dtype=np.int32)

    # Static length is required for jnp.repeat under tracing; `total` is a host
    # scalar precisely so the rest of the routine can have concrete shapes.
    cell_of_entry = jnp.repeat(jnp.arange(num_cells, dtype=jnp.int32), counts,
                               total_repeat_length=total)

    slots = jax.random.randint(slot_key, (total,), 0, num_resamples, dtype=jnp.int32)
    for round_index in range(max_repair_rounds):
        pair = cell_of_entry.astype(jnp.int64) * num_resamples + slots
        order = jnp.argsort(pair)
        ranked = pair[order]
        duplicated = jnp.zeros(total, dtype=bool).at[order[1:]].set(ranked[1:] == ranked[:-1])
        num_duplicated = int(duplicated.sum())
        if num_duplicated == 0:
            break
        redraw = jax.random.randint(
            jax.random.fold_in(slot_key, round_index + 1), (total,), 0, num_resamples,
            dtype=jnp.int32,
        )
        slots = jnp.where(duplicated, redraw, slots)

    per_slot = np.asarray(jnp.bincount(slots, length=num_resamples))
    width = int(per_slot.max())

    order = jnp.argsort(slots)
    sorted_slots = slots[order]
    sorted_cells = cell_of_entry[order]
    starts = jnp.searchsorted(sorted_slots, jnp.arange(num_resamples, dtype=jnp.int32), side="left")
    position = jnp.arange(total, dtype=jnp.int32) - starts[sorted_slots]
    out = jnp.full((num_resamples, width), pad_value, dtype=jnp.int32)
    out = out.at[sorted_slots, position].set(sorted_cells, mode="drop")
    return np.asarray(out, dtype=np.int32)
