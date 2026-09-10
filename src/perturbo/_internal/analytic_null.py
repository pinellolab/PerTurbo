"""Closed-form moments of the low-MOI CRT null, for the categorical-batch path.

The resampled statistic in
:func:`~perturbo._internal.jax_kernels.batched_efficient_score_from_indices_categorical_batch`
is, for a draw ``S`` from the target's pair pool,

    numerator    A = sum_{i in S} c_i,     c_i = r_i - w_i u_{s(i)} / I_{s(i)}
    denominator  J = sum_s B_s (1 - B_s / I_s),   B_s = sum_{i in S cap s} w_i
    statistic    T = A / sqrt(J)

with ``I_s`` and ``u_s`` the pool weight and residual sums in stratum ``s`` --
both fixed while the assignment is resampled. So ``A`` is exactly a stratified
sample sum of the fixed per-cell vector ``c``, and ``J`` is a sample sum plus a
small quadratic. Writing ``J = nu + L - Q``,

    T = nu^(-1/2) (A) (1 + (L - Q)/nu)^(-1/2)

and ``(L - Q)/nu = O(k^-1/2)``, so a delta expansion gives the moments of ``T``
from joint moments of ``(A, L)``. Those are stratified-SRSWOR sample sums of
fixed vectors, whose joint moments are exact and closed form.

This replaces the Monte Carlo estimate of the null moments that feeds
``parametric_null``; its cost does not depend on the resample count ``B`` and
involves no per-resample gather over cells.

Accuracy is controlled by ``jmax``, the highest retained power of ``L``: a
monomial ``A^i L^j`` enters at order ``k^(-j/2)``, so the moments carry relative
error ``O(k^-(jmax+1)/2)``. ``jmax=2`` matches 999-draw Monte Carlo noise at a
few hundred cells per target; small targets want ``jmax=3``.

Two exact facts keep the expansion short:

* ``sum_pool c_i = u_s - I_s (u_s / I_s) = 0`` in every stratum, so ``E[A] = 0``
  exactly and the expansion has no constant term.
* The delta expansion only ever reaches joint moments with ``a <= mmax`` and
  ``b <= jmax``, and every set-partition block of such an index stays inside
  that rectangle -- so the moment/cumulant recursion closes on it, and the rest
  of the ``a + b <= order`` triangle can be skipped.
"""

from __future__ import annotations

import math
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

__all__ = [
    "categorical_batch_null_raw_moments",
    "intercept_only_null_raw_moments",
    "control_stratum_means",
    "DEFAULT_JMAX",
    "DEFAULT_MMAX",
]

DEFAULT_JMAX = 2
DEFAULT_MMAX = 4


# ---------------------------------------------------------------------------
# set-partition machinery (cached; depends only on the truncation orders)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _set_partitions(n: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if n == 0:
        return ((),)
    if n == 1:
        return (((0,),),)
    out: list[tuple[tuple[int, ...], ...]] = []
    for smaller in _set_partitions(n - 1):
        for i in range(len(smaller)):
            out.append(smaller[:i] + (smaller[i] + (n - 1,),) + smaller[i + 1 :])
        out.append(smaller + ((n - 1,),))
    return tuple(out)


@lru_cache(maxsize=None)
def _srswor_terms(a: int, b: int) -> tuple[tuple[int, float, tuple[tuple[int, int], ...]], ...]:
    """Symbolic E[X^a Y^b] for centred SRSWOR sample sums X, Y.

    ``E[prod delta]`` depends only on how many of the ``a + b`` factor indices
    are distinct, which is the outer partition; the inner partition is the
    Mobius inversion turning "sum over distinct indices" into power sums.
    Returns ``(nblocks, coeff, powers)`` so the moment is

        sum coeff * (k)_nblocks / (N)_nblocks * prod S[p, q].
    """
    labels = (0,) * a + (1,) * b
    terms: dict[tuple[int, tuple[tuple[int, int], ...]], float] = {}
    for pi in _set_partitions(a + b):
        sig = [
            (
                sum(1 for j in block if labels[j] == 0),
                sum(1 for j in block if labels[j] == 1),
            )
            for block in pi
        ]
        nblocks = len(pi)
        for sigma in _set_partitions(nblocks):
            mu = 1.0
            merged = []
            for group in sigma:
                mu *= (-1) ** (len(group) - 1) * math.factorial(len(group) - 1)
                merged.append(
                    (sum(sig[t][0] for t in group), sum(sig[t][1] for t in group))
                )
            key = (nblocks, tuple(sorted(merged)))
            terms[key] = terms.get(key, 0.0) + mu
    return tuple((nb, c, p) for (nb, p), c in terms.items() if c != 0.0)


@lru_cache(maxsize=None)
def _partition_signatures(a: int, b: int) -> tuple[tuple[tuple[tuple[int, int], ...], int], ...]:
    labels = (0,) * a + (1,) * b
    return tuple(
        (
            tuple(
                sorted(
                    (
                        sum(1 for j in block if labels[j] == 0),
                        sum(1 for j in block if labels[j] == 1),
                    )
                    for block in pi
                )
            ),
            len(pi),
        )
        for pi in _set_partitions(a + b)
    )


@lru_cache(maxsize=None)
def _needed_pairs(mmax: int, jmax: int) -> tuple[tuple[int, int], ...]:
    return tuple((a, b) for a in range(mmax + 1) for b in range(jmax + 1))


@lru_cache(maxsize=None)
def _raw_pairs(mmax: int, order: int) -> tuple[tuple[int, int], ...]:
    return tuple((a, b) for a in range(mmax + 1) for b in range(order - a + 1))


@lru_cache(maxsize=None)
def _moment_program(mmax: int, jmax: int):
    """SRSWOR joint moments collapsed onto distinct power-sum products."""
    pairs = _needed_pairs(mmax, jmax)
    slot = {pq: i for i, pq in enumerate(pairs)}
    plan = []
    for a, b in pairs:
        if a + b == 0:
            plan.append([])
            continue
        grouped: dict[tuple[int, ...], dict[int, float]] = {}
        for nb, coeff, powers in _srswor_terms(a, b):
            key = tuple(slot[pq] for pq in powers)
            by_nb = grouped.setdefault(key, {})
            by_nb[nb] = by_nb.get(nb, 0.0) + coeff
        plan.append(
            [
                (slots, tuple((nb, c) for nb, c in sorted(by_nb.items()) if c != 0.0))
                for slots, by_nb in grouped.items()
            ]
        )
    return pairs, plan


@lru_cache(maxsize=None)
def _cumulant_program(mmax: int, jmax: int):
    """Moments<->cumulants collapsed onto distinct block signatures."""
    pairs = _needed_pairs(mmax, jmax)
    slot = {pq: i for i, pq in enumerate(pairs)}
    to_cum, to_mom = [], []
    for a, b in pairs:
        if a + b == 0:
            to_cum.append([])
            to_mom.append([])
            continue
        cw: dict[tuple[int, ...], float] = {}
        mw: dict[tuple[int, ...], float] = {}
        for sig, nblocks in _partition_signatures(a, b):
            slots = tuple(sorted(slot[pq] for pq in sig))
            cw[slots] = cw.get(slots, 0.0) + (-1) ** (nblocks - 1) * math.factorial(nblocks - 1)
            mw[slots] = mw.get(slots, 0.0) + 1.0
        to_cum.append([(v, k) for k, v in cw.items() if v != 0.0])
        to_mom.append([(v, k) for k, v in mw.items() if v != 0.0])
    return pairs, to_cum, to_mom


def _falling(n, r: int):
    out = jnp.ones_like(n, dtype=jnp.float64)
    for i in range(r):
        out = out * (n - i)
    return out


# ---------------------------------------------------------------------------
# raw tables
# ---------------------------------------------------------------------------


def control_stratum_means(
    observation_weight: np.ndarray,
    batch_codes: np.ndarray,
    control_mask: np.ndarray,
    num_batches: int,
) -> np.ndarray:
    """Per-stratum control mean of ``w``, used as the power-sum shift.

    Power sums are accumulated for ``w - shift`` rather than ``w`` so that
    forming central moments never differences large like-sized quantities. A
    high-expression gene can have ``w`` two orders of magnitude above its own
    spread, where the unshifted route loses most of its significant digits by
    the sixth power.
    """
    idx = np.flatnonzero(control_mask)
    codes = batch_codes[idx]
    total = np.zeros((num_batches, observation_weight.shape[1]), dtype=np.float64)
    np.add.at(total, codes, observation_weight[idx].astype(np.float64))
    count = np.bincount(codes, minlength=num_batches).astype(np.float64)
    return total / np.maximum(count, 1.0)[:, None]


# ---------------------------------------------------------------------------
# raw pool moment tables, batched over targets
# ---------------------------------------------------------------------------

# Cap on float64 elements in one intermediate of the moment assembly. Bounds
# peak memory only; it never changes the result.
# Sized to keep one moment-assembly intermediate roughly L3-resident. Larger
# blocks amortize NumPy call overhead but lose cache; smaller ones invert that.
_TARGET_BLOCK_ELEMENTS = 4_000_000
_CELL_BLOCK = 40_000
# Below this many (cells x genes) the float64 NumPy reduction is used: the JAX
# dispatch does not pay for itself, and small problems are where the exactness
# tests live.
_JAX_MIN_ELEMENTS = 1_000_000


def _target_block_size(n_strata: int, n_genes: int, n_terms: int) -> int:
    return max(1, int(_TARGET_BLOCK_ELEMENTS // max(n_strata * n_genes * n_terms, 1)))


@partial(jax.jit, static_argnames=("num_segments", "order", "mmax"))
def _segment_raw_tables_jax(residual, weight, segment, num_segments, order, mmax):
    """Fused powers-into-segment-sum for one block of cells.

    XLA folds each ``r^a w^b`` into its reduction, so the (cells x genes)
    intermediate is never materialized - which is the whole cost on the NumPy
    path.

    Runs in whatever precision the caller hands over. With the package's x64
    setting that is float64, which is what makes this usable for the control
    pool: that table spans every control cell and includes ``u = sum_pool r``,
    a near-total cancellation held to ~0 by the null fit. Under the old float32
    default, accumulation over that many terms would have swamped it, so the
    control side was confined to NumPy; it no longer needs to be.
    """
    pairs = _raw_pairs(mmax, order)
    max_a = max(a for a, _ in pairs)
    max_b = max(b for _, b in pairs)
    rp = [jnp.ones_like(residual)]
    for _ in range(max_a):
        rp.append(rp[-1] * residual)
    wp = [jnp.ones_like(weight)]
    for _ in range(max_b):
        wp.append(wp[-1] * weight)
    return jnp.stack([
        jax.ops.segment_sum(
            rp[a] if b == 0 else (wp[b] if a == 0 else rp[a] * wp[b]),
            segment, num_segments=num_segments, indices_are_sorted=True,
        )
        for a, b in pairs
    ])


def _segment_raw_tables(
    score_residual: np.ndarray,
    observation_weight: np.ndarray,
    shift: np.ndarray,
    cells: np.ndarray,
    segment: np.ndarray,
    segment_stratum: np.ndarray,
    num_segments: int,
    order: int,
    mmax: int,
    use_jax: bool = False,
) -> np.ndarray:
    """``raw[t, seg, g] = sum over the cells of one segment of r^a (w - shift)^b``.

    One reduction per ``(a, b)`` over every cell at once, rather than one per
    target. The reduction is a sparse indicator matmul: ``np.add.reduceat`` is
    the obvious primitive and is ~30x slower here, because it degrades badly
    when the reduced axis is long relative to the segment count.

    Cells are blocked so the cached ``r^a`` and ``w^b`` powers stay a few
    hundred MB at transcriptome scale, and accumulated in float64 throughout:
    the delta expansion differences these sums against each other, and
    ``u = sum_pool r`` in particular is a near-total cancellation.
    """
    pairs = _raw_pairs(mmax, order)
    n_genes = score_residual.shape[1]
    out = np.zeros((len(pairs), num_segments, n_genes), dtype=np.float64)
    if cells.size == 0:
        return out
    max_a = max(a for a, _ in pairs)
    max_b = max(b for _, b in pairs)

    for start in range(0, cells.size, _CELL_BLOCK):
        block = cells[start : start + _CELL_BLOCK]
        seg = segment[start : start + _CELL_BLOCK]
        if use_jax and block.size * score_residual.shape[1] >= _JAX_MIN_ELEMENTS:
            out += np.asarray(
                _segment_raw_tables_jax(
                    jnp.asarray(score_residual[block]),
                    jnp.asarray(observation_weight[block]) - jnp.asarray(shift[segment_stratum[seg]]),
                    jnp.asarray(seg), num_segments, order, mmax,
                ),
                dtype=np.float64,
            )
            continue
        indicator = sparse.csr_matrix(
            (np.ones(block.size), (seg, np.arange(block.size))),
            shape=(num_segments, block.size),
        )
        rc = np.asarray(score_residual[block], dtype=np.float64)
        wc = np.asarray(observation_weight[block], dtype=np.float64) - shift[segment_stratum[seg]]
        rp = [np.ones_like(rc)]
        for _ in range(max_a):
            rp.append(rp[-1] * rc)
        wp = [np.ones_like(wc)]
        for _ in range(max_b):
            wp.append(wp[-1] * wc)
        for index, (a, b) in enumerate(pairs):
            value = rp[a] if b == 0 else (wp[b] if a == 0 else rp[a] * wp[b])
            out[index] += indicator @ value
    return out


def _order_targeting_cells(
    batch_codes: np.ndarray,
    target_cells: dict[int, np.ndarray],
    num_batches: int,
):
    """Concatenate every target's cells in (target block index, stratum) order."""
    indices = sorted(target_cells)
    cells, segment, counts = [], [], []
    for position, target_index in enumerate(indices):
        own = np.asarray(target_cells[target_index], dtype=np.int64)
        codes = batch_codes[own]
        ordering = np.argsort(codes, kind="stable")
        cells.append(own[ordering])
        segment.append(position * num_batches + codes[ordering])
        counts.append(np.bincount(codes, minlength=num_batches))
    if not cells:
        empty = np.zeros(0, dtype=np.int64)
        return np.asarray(indices, dtype=np.int64), empty, empty, np.zeros((0, num_batches), np.int64)
    return (
        np.asarray(indices, dtype=np.int64),
        np.concatenate(cells),
        np.concatenate(segment),
        np.stack(counts),
    )


# ---------------------------------------------------------------------------
# stratified (categorical batch) moments
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("order", "jmax", "mmax"))
def _target_raw_moments(table, shift, n_pool, counts, order, jmax, mmax):
    """Raw moments ``E[T^m]`` for a block of targets across a gene chunk.

    ``table`` is ``(terms, targets, strata, genes)``, ``n_pool`` and ``counts``
    are ``(targets, strata)``. Empty strata need no masking: their falling
    factorial ``(0)_p`` vanishes, so they contribute nothing to any cumulant.

    Jitted so XLA fuses the whole chain. Every intermediate here - the shifted
    table, the power sums, the moments, the cumulants - is
    ``(terms, targets, strata, genes)``, and materializing each one separately
    was most of this function's cost; there are roughly a hundred full-array
    passes over tens of megabytes. The truncation orders are static because
    they drive the Python loops, which unroll into the trace, so each distinct
    ``(order, jmax, mmax)`` compiles once.
    """
    raw_pairs = _raw_pairs(mmax, order)
    raw_slot = {pq: i for i, pq in enumerate(raw_pairs)}
    pairs = _needed_pairs(mmax, jmax)
    slot = {pq: i for i, pq in enumerate(pairs)}
    n_genes = table.shape[3]
    shape = table.shape[1:]                                    # (targets, strata, genes)

    kf = counts.astype(jnp.float64)[..., None]
    npool = n_pool.astype(jnp.float64)[..., None]
    safe_pool = jnp.where(npool > 0, npool, 1.0)
    shift_b = shift[None, :, :]

    information = table[raw_slot[(0, 1)]] + npool * shift_b    # I_s = sum_pool w
    safe_information = jnp.where(information > 0, information, 1.0)
    u = table[raw_slot[(1, 0)]]                                # u_s = sum_pool r
    phi = u / safe_information
    omega_bar = table[raw_slot[(0, 1)]] / safe_pool
    wbar = omega_bar + shift_b
    beta = kf * wbar                                           # E[B_s]
    lam = 1.0 - 2.0 * beta / safe_information

    binom = [[math.comb(n, k) for k in range(n + 1)] for n in range(order + 1)]

    def _powers(base, n):
        out = [jnp.ones(shape)]
        for _ in range(n):
            out.append(out[-1] * base)
        return out

    p_ps = _powers(-phi * shift_b, mmax)
    p_p = _powers(-phi, mmax)
    p_ob = _powers(-omega_bar, jmax)

    # Two binomial passes rather than one trinomial sum over
    # (r, phi*shift, omega): same result, a third of the array traffic.
    shifted = []
    for a, b in raw_pairs:
        acc = jnp.zeros(shape)
        for m in range(a + 1):
            acc = acc + binom[a][m] * p_ps[m] * table[raw_slot[(a - m, b)]]
        shifted.append(acc)

    # power sums of (c, w - wbar) with c = r - phi*w; note sum_pool c = 0 exactly
    S = []
    for p, q in pairs:
        if p + q == 0:
            S.append(jnp.zeros(shape))
            continue
        acc = jnp.zeros(shape)
        for j in range(p + 1):
            for l in range(q + 1):
                b_idx = j + q - l
                if (p - j) + b_idx > order:
                    continue
                acc = acc + (
                    (binom[p][j] * binom[q][l])
                    * p_p[j] * p_ob[l]
                    * shifted[raw_slot[(p - j, b_idx)]]
                )
        S.append(acc)

    # exact SRSWOR joint central moments of (A_s, B_s)
    _, plan = _moment_program(mmax, jmax)
    max_nb = max((nb for terms in plan for _, by in terms for nb, _ in by), default=0)
    fall = [None] * (max_nb + 1)
    for nb in range(1, max_nb + 1):
        fall[nb] = _falling(kf, nb) / _falling(safe_pool, nb)

    mom = []
    for i, (a, b) in enumerate(pairs):
        if a + b == 0:
            mom.append(jnp.ones(shape))
            continue
        acc = jnp.zeros(shape)
        for slots, by_nb in plan[i]:
            weight = sum(c * fall[nb] for nb, c in by_nb)
            term = S[slots[0]] * weight
            for sl in slots[1:]:
                term = term * S[sl]
            acc = acc + term
        mom.append(acc)

    # cumulants add over independent strata; lam scales the L slot
    _, to_cum, to_mom = _cumulant_program(mmax, jmax)
    total_cum = []
    for i, (a, b) in enumerate(pairs):
        if a + b == 0:
            total_cum.append(jnp.zeros((shape[0], n_genes)))
            continue
        acc = jnp.zeros(shape)
        for coeff, slots in to_cum[i]:
            term = mom[slots[0]] * coeff
            for sl in slots[1:]:
                term = term * mom[sl]
            acc = acc + term
        total_cum.append((acc * lam**b).sum(axis=1))

    joint = []
    for i, (a, b) in enumerate(pairs):
        if a + b == 0:
            joint.append(jnp.ones((shape[0], n_genes)))
            continue
        acc = jnp.zeros((shape[0], n_genes))
        for coeff, slots in to_mom[i]:
            term = total_cum[slots[0]] * coeff
            for sl in slots[1:]:
                term = term * total_cum[sl]
            acc = acc + term
        joint.append(acc)

    # nu = E[J], alpha = E[A] = 0 exactly (sum_pool c = 0 in every stratum)
    nu = (beta - (beta**2 + mom[slot[(0, 2)]]) / safe_information).sum(axis=1)
    return _delta_expansion(joint, slot, nu, order, jmax, mmax)


def _delta_expansion(joint, slot, nu, order, jmax, mmax):
    """``E[T^m]`` from the joint moments of ``(A, L)`` and ``nu = E[J]``.

    ``joint`` is a list indexed by ``slot``, not a stacked array: the entries
    are only ever read by key, so stacking them would materialize a tensor for
    nothing. Non-positive ``nu`` deliberately produces NaN rather than an
    error - a degenerate pool has no statistic, and the caller filters on
    finiteness.
    """
    inv_sqrt = jnp.where(nu > 0, nu, jnp.nan) ** -0.5
    series = [1.0, -0.5, 3.0 / 8.0, -5.0 / 16.0][: jmax + 1]
    poly: dict[tuple[int, int], object] = {}
    for j, scale in enumerate(series):
        poly[(1, j)] = poly.get((1, j), 0.0) + inv_sqrt * scale / nu**j

    out = []
    for m in range(1, mmax + 1):
        power: dict[tuple[int, int], object] = {(0, 0): jnp.ones(nu.shape)}
        for _ in range(m):
            nxt: dict[tuple[int, int], object] = {}
            for (i1, j1), v1 in power.items():
                for (i2, j2), v2 in poly.items():
                    i, j = i1 + i2, j1 + j2
                    if j > jmax or i > mmax:
                        continue
                    nxt[(i, j)] = nxt.get((i, j), 0.0) + v1 * v2
            power = nxt
        out.append(sum(v * joint[slot[key]] for key, v in power.items()))
    return jnp.stack(out)


def categorical_batch_null_raw_moments(
    *,
    score_residual: np.ndarray,
    observation_weight: np.ndarray,
    batch_codes: np.ndarray,
    control_mask: np.ndarray,
    target_cells: dict[int, np.ndarray],
    num_targets: int,
    num_batches: int,
    jmax: int = DEFAULT_JMAX,
    mmax: int = DEFAULT_MMAX,
) -> np.ndarray:
    """Raw null moments ``E[T^m]`` for every target over one gene chunk.

    Returns ``(num_targets, num_genes, mmax)``; targets absent from
    ``target_cells`` are NaN. ``score_residual`` and ``observation_weight`` are
    the control-only null fit's per-cell arrays, without any padding row.
    """
    order = mmax + jmax
    n_genes = score_residual.shape[1]
    shift = control_stratum_means(observation_weight, batch_codes, control_mask, num_batches)
    control_cells = np.flatnonzero(control_mask)
    control_table = _segment_raw_tables(
        score_residual, observation_weight, shift,
        control_cells[np.argsort(batch_codes[control_cells], kind="stable")],
        np.sort(batch_codes[control_cells]), np.arange(num_batches),
        num_batches, order, mmax, use_jax=True,
    )
    control_counts = np.bincount(batch_codes[control_cells], minlength=num_batches)

    indices, cells, segment, counts = _order_targeting_cells(batch_codes, target_cells, num_batches)
    out = np.full((num_targets, n_genes, mmax), np.nan)
    if indices.size == 0:
        return out

    n_terms = len(_raw_pairs(mmax, order))
    block = _target_block_size(num_batches, n_genes, n_terms)
    segment_stratum = np.arange(num_batches * block) % num_batches
    for start in range(0, indices.size, block):
        stop = min(start + block, indices.size)
        take = (segment >= start * num_batches) & (segment < stop * num_batches)
        table = _segment_raw_tables(
            score_residual, observation_weight, shift,
            cells[take], segment[take] - start * num_batches,
            segment_stratum, (stop - start) * num_batches, order, mmax, use_jax=True,
        ).reshape(n_terms, stop - start, num_batches, n_genes)
        table += control_table[:, None, :, :]
        moments = _target_raw_moments(
            table, shift, control_counts[None, :] + counts[start:stop],
            counts[start:stop], order, jmax, mmax,
        )
        # The moment assembly runs on device; land it back in the host array
        # the caller owns. Explicit rather than relying on __array__.
        out[indices[start:stop]] = np.moveaxis(np.asarray(moments), 0, -1)
    return out


# ---------------------------------------------------------------------------
# intercept-only (no batch covariate) specialization
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("order", "jmax", "mmax"))
def _intercept_raw_moments(table, shift, n_pool, counts, order, jmax, mmax):
    """Raw moments ``E[T^m]`` for a block of targets, intercept-only nuisance.

    ``table`` is ``(terms, targets, genes)``. Deliberately independent of the
    stratified path rather than a call into it with one batch: with a single
    stratum the moment/cumulant round trip is the identity, and the scaling
    ``E[A^a (lam L)^b] = lam^b E[A^a L^b]`` feeds the delta expansion directly.
    Agreement between the two is therefore a real cross-check of the stratified
    machinery, not a tautology.
    """
    raw_pairs = _raw_pairs(mmax, order)
    raw_slot = {pq: i for i, pq in enumerate(raw_pairs)}
    pairs = _needed_pairs(mmax, jmax)
    slot = {pq: i for i, pq in enumerate(pairs)}
    n_genes = table.shape[2]
    shape = table.shape[1:]                                    # (targets, genes)

    kf = counts.astype(jnp.float64)[:, None]
    npool = n_pool.astype(jnp.float64)[:, None]
    shift0 = shift[0][None, :]

    information = table[raw_slot[(0, 1)]] + npool * shift0     # I = sum_pool w
    safe_information = jnp.where(information > 0, information, 1.0)
    u = table[raw_slot[(1, 0)]]                                # u = sum_pool r
    phi = u / safe_information
    omega_bar = table[raw_slot[(0, 1)]] / npool
    wbar = omega_bar + shift0
    beta = kf * wbar
    lam = 1.0 - 2.0 * beta / safe_information

    binom = [[math.comb(n, k) for k in range(n + 1)] for n in range(order + 1)]

    def _powers(base, n):
        out = [jnp.ones(shape)]
        for _ in range(n):
            out.append(out[-1] * base)
        return out

    p_ps = _powers(-phi * shift0, mmax)
    p_p = _powers(-phi, mmax)
    p_ob = _powers(-omega_bar, jmax)

    shifted = []
    for a, b in raw_pairs:
        acc = jnp.zeros(shape)
        for m in range(a + 1):
            acc = acc + binom[a][m] * p_ps[m] * table[raw_slot[(a - m, b)]]
        shifted.append(acc)

    S = []
    for p, q in pairs:
        if p + q == 0:
            S.append(jnp.zeros(shape))
            continue
        acc = jnp.zeros(shape)
        for j in range(p + 1):
            for l in range(q + 1):
                b_idx = j + q - l
                if (p - j) + b_idx > order:
                    continue
                acc = acc + (
                    (binom[p][j] * binom[q][l])
                    * p_p[j] * p_ob[l]
                    * shifted[raw_slot[(p - j, b_idx)]]
                )
        S.append(acc)

    _, plan = _moment_program(mmax, jmax)
    max_nb = max((nb for terms in plan for _, by in terms for nb, _ in by), default=0)
    fall = [None] * (max_nb + 1)
    for nb in range(1, max_nb + 1):
        fall[nb] = _falling(kf, nb) / _falling(npool, nb)

    joint = []
    var_l = jnp.zeros(shape)
    for i, (a, b) in enumerate(pairs):
        if a + b == 0:
            joint.append(jnp.ones(shape))
            continue
        acc = jnp.zeros(shape)
        for slots, by_nb in plan[i]:
            weight = sum(c * fall[nb] for nb, c in by_nb)
            term = S[slots[0]] * weight
            for sl in slots[1:]:
                term = term * S[sl]
            acc = acc + term
        if (a, b) == (0, 2):
            var_l = acc                                        # Var(L), unscaled
        joint.append(acc * lam**b)                             # E[A^a (lam L)^b]

    nu = beta - (beta**2 + var_l) / safe_information
    return _delta_expansion(joint, slot, nu, order, jmax, mmax)


def intercept_only_null_raw_moments(
    *,
    score_residual: np.ndarray,
    observation_weight: np.ndarray,
    control_mask: np.ndarray,
    target_cells: dict[int, np.ndarray],
    num_targets: int,
    jmax: int = DEFAULT_JMAX,
    mmax: int = DEFAULT_MMAX,
) -> np.ndarray:
    """Raw null moments ``E[T^m]`` for every target, no batch covariate.

    The no-batch kernel projects out a global intercept, so the per-cell
    correction is the single scalar ``u / I`` and the resampling law is one
    unstratified SRSWOR draw over the pair pool. One quantified departure from
    that kernel: its nuisance information carries a curvature jitter of 1e-8,
    which this closed form omits; the relative effect is of order ``1e-8 / I``,
    far below both float32 kernel precision and the expansion's own truncation.
    """
    order = mmax + jmax
    n_cells, n_genes = score_residual.shape
    zeros = np.zeros(n_cells, dtype=np.int64)
    shift = control_stratum_means(observation_weight, zeros, control_mask, 1)
    control_cells = np.flatnonzero(control_mask)
    control_table = _segment_raw_tables(
        score_residual, observation_weight, shift, control_cells,
        np.zeros(control_cells.size, dtype=np.int64), np.zeros(1, dtype=np.int64),
        1, order, mmax, use_jax=True,
    )[:, 0, :]
    control_count = control_cells.size

    indices, cells, segment, counts = _order_targeting_cells(zeros, target_cells, 1)
    out = np.full((num_targets, n_genes, mmax), np.nan)
    if indices.size == 0:
        return out

    n_terms = len(_raw_pairs(mmax, order))
    block = _target_block_size(1, n_genes, n_terms)
    for start in range(0, indices.size, block):
        stop = min(start + block, indices.size)
        take = (segment >= start) & (segment < stop)
        table = _segment_raw_tables(
            score_residual, observation_weight, shift,
            cells[take], segment[take] - start, np.zeros(stop - start, dtype=np.int64),
            stop - start, order, mmax, use_jax=True,
        )
        table += control_table[:, None, :]
        own = counts[start:stop, 0]
        moments = _intercept_raw_moments(
            table, shift, control_count + own, own, order, jmax, mmax,
        )
        # The moment assembly runs on device; land it back in the host array
        # the caller owns. Explicit rather than relying on __array__.
        out[indices[start:stop]] = np.moveaxis(np.asarray(moments), 0, -1)
    return out
