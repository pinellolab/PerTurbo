"""Saddlepoint tail probabilities for permutation-style score nulls.

The score a CRT computes for one (target, gene) pair is a sum over the cells
assigned to that target of a per-cell contribution, and its null is generated
by reassigning which cells those are. So the null is the distribution of a sum
of ``m`` values drawn from a pool of ``n``, and its cumulant generating
function is available in closed form from the pool itself - no parametric
family need be assumed at all.

Why this rather than matching three or four moments:

  Edgeworth-type expansions - Cornish-Fisher, and in spirit a moment-matched
  skew-normal - control *absolute* error. Out where a p-value is 1e-50 that is
  worth nothing. With the correct CGF, a saddlepoint approximation controls
  *relative* error and does so uniformly into the tail. It is also evaluated in
  the exponent, so the log tail is native rather than the log of a number that
  already underflowed. The with-replacement surrogate below adds a separate
  finite-population modeling error which must be measured empirically.

  Concretely it removes both failure modes the skew-normal has here: the tail
  pinning at 2.2e-308 past |z| ~ 38, and the hard ceiling at |skewness| <
  0.995272 that leaves a genuinely more skewed null with no representable fit.

The CRT samples a fixed number of cells without replacement inside every
stratum.  The exact treatment needs a conditional/double saddlepoint (Booth &
Butler 1990).  This prototype instead adds the CGFs of independent
with-replacement draws, one per stratum.  An optional power-CGF correction
matches the exact finite-population variance while retaining the original
support; it is a heuristic, not the without-replacement CGF.  The maximum
within-stratum sampling fraction is returned with every fit so callers can
measure where this approximation is being leaned on.

References: Lugannani & Rice (1980); Daniels (1987); Robinson (1982) for
permutation distributions specifically; Butler, *Saddlepoint Approximations
with Applications* (2007).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import stats as jstats
from jax.scipy.special import gammainc, gammaincc, logsumexp

# Below this |t| the saddlepoint is effectively at the mean, where the
# Lugannani-Rice correction term 1/w - 1/u is 0/0. Daniels' limiting form is
# used instead.
_NEAR_MEAN = 1e-6


@dataclass(frozen=True)
class SaddlepointTailFit:
    """Per-pair numerator-SPA results on both linear and log scales."""

    p_value: np.ndarray
    log_p_value: np.ndarray
    null_mean: np.ndarray
    null_variance: np.ndarray
    null_skewness: np.ndarray
    observed_sum: np.ndarray
    max_sampling_fraction: np.ndarray
    valid: np.ndarray
    used_fallback: np.ndarray


def _cgf_terms(t: jnp.ndarray, pool: jnp.ndarray, m: jnp.ndarray):
    """``K``, ``K'`` and ``K''`` of a sum of ``m`` draws from ``pool`` at ``t``.

    Everything goes through ``logsumexp`` so a large ``t`` cannot overflow:
    the tilted weights are normalised in log space before any moment is taken.
    """
    weighted = t * pool                        # (pool,)
    log_norm = logsumexp(weighted)
    probability = jnp.exp(weighted - log_norm)  # tilted distribution
    mean_tilted = jnp.sum(probability * pool)
    var_tilted = jnp.sum(probability * jnp.square(pool)) - jnp.square(mean_tilted)
    n = pool.shape[0]
    cgf = m * (log_norm - jnp.log(n))
    return cgf, m * mean_tilted, m * var_tilted


def _stratified_cgf_terms(
    t: jnp.ndarray,
    pool: jnp.ndarray,
    mask: jnp.ndarray,
    selected: jnp.ndarray,
    finite_population_factor: jnp.ndarray,
):
    """CGF derivatives for independent stratum-specific sample sums.

    ``pool`` is ``(strata, padded_cells, genes)`` and ``mask`` marks real
    cells.  For stratum ``s`` the uncorrected contribution is

        m_s log mean_i exp(t c_si).

    The optional finite-population adjustment is the power-CGF transform
    ``K_s(F_s t) / F_s``.  It preserves the mean and support and multiplies the
    variance by ``F_s = (N_s-m_s)/(N_s-1)``.  It does not make the draws
    without-replacement; that requires the conditional SPA deliberately left
    for a later implementation.
    """

    scaled_t = finite_population_factor[:, None] * t[None, :]
    logits = scaled_t[:, None, :] * pool
    logits = jnp.where(mask[:, :, None], logits, -jnp.inf)
    log_norm = logsumexp(logits, axis=1)
    population = mask.sum(axis=1).astype(pool.dtype)
    probability = jnp.exp(logits - log_norm[:, None, :])
    probability = jnp.where(mask[:, :, None], probability, 0.0)
    tilted_mean = jnp.sum(probability * pool, axis=1)
    tilted_variance = (
        jnp.sum(probability * jnp.square(pool), axis=1) - jnp.square(tilted_mean)
    )
    multiplier = selected[:, None] / finite_population_factor[:, None]
    cgf = jnp.sum(multiplier * (log_norm - jnp.log(population)[:, None]), axis=0)
    first = jnp.sum(selected[:, None] * tilted_mean, axis=0)
    second = jnp.sum(
        selected[:, None] * finite_population_factor[:, None] * tilted_variance,
        axis=0,
    )
    return cgf, first, second


def _solve_stratified_saddlepoint(
    target: jnp.ndarray,
    pool: jnp.ndarray,
    mask: jnp.ndarray,
    selected: jnp.ndarray,
    finite_population_factor: jnp.ndarray,
    *,
    iterations: int,
    expansions: int = 40,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the vector of gene-wise stratified saddlepoint equations."""

    dtype = pool.dtype
    low = -jnp.ones_like(target)
    high = jnp.ones_like(target)

    def widen(_, bounds):
        low, high = bounds
        _, at_low, _ = _stratified_cgf_terms(
            low, pool, mask, selected, finite_population_factor
        )
        _, at_high, _ = _stratified_cgf_terms(
            high, pool, mask, selected, finite_population_factor
        )
        return (
            jnp.where(at_low > target, low * 2.0, low),
            jnp.where(at_high < target, high * 2.0, high),
        )

    low, high = jax.lax.fori_loop(0, expansions, widen, (low, high))
    _, at_low, _ = _stratified_cgf_terms(
        low, pool, mask, selected, finite_population_factor
    )
    _, at_high, _ = _stratified_cgf_terms(
        high, pool, mask, selected, finite_population_factor
    )
    bracketed = (at_low <= target) & (at_high >= target)

    def refine(_, state):
        t, low, high = state
        _, first, second = _stratified_cgf_terms(
            t, pool, mask, selected, finite_population_factor
        )
        low = jnp.where(first < target, t, low)
        high = jnp.where(first < target, high, t)
        newton = t + (target - first) / jnp.maximum(second, jnp.finfo(dtype).tiny)
        inside = (newton > low) & (newton < high) & jnp.isfinite(newton)
        return jnp.where(inside, newton, 0.5 * (low + high)), low, high

    initial = jnp.zeros_like(target)
    t_hat, _, _ = jax.lax.fori_loop(0, iterations, refine, (initial, low, high))
    return t_hat, bracketed


def _solve_saddlepoint(target: jnp.ndarray, pool: jnp.ndarray, m: jnp.ndarray,
                       *, iterations: int, expansions: int = 40) -> jnp.ndarray:
    """Solve ``K'(t) = target`` by Newton safeguarded inside a bracket.

    ``K'`` is strictly increasing - ``K''`` is a variance - so the root is
    unique and bracketing is reliable. Plain Newton is not: the pool is a
    finite empirical sample, so a large ``t`` tilts it onto its extreme value,
    ``K''`` collapses toward zero, and the Newton step diverges. Validated
    against a Poisson sum, unsafeguarded Newton silently returned a *negative*
    saddlepoint for any target past about 17 standard deviations - exactly the
    regime this exists to serve - and the tail came back as a constant.

    So each iteration takes the Newton step only when it stays inside the
    current bracket and otherwise bisects, and the bracket is contracted either
    way. That keeps Newton's convergence rate where it works while inheriting
    bisection's guarantee where it does not.
    """
    dtype = pool.dtype
    zero = jnp.asarray(0.0, dtype)

    def widen(_, bounds):
        low, high = bounds
        _, at_low, _ = _cgf_terms(low, pool, m)
        _, at_high, _ = _cgf_terms(high, pool, m)
        return (jnp.where(at_low > target, low * 2.0, low),
                jnp.where(at_high < target, high * 2.0, high))

    low, high = jax.lax.fori_loop(
        0, expansions, widen, (jnp.asarray(-1.0, dtype), jnp.asarray(1.0, dtype))
    )

    def refine(_, state):
        t, low, high = state
        _, first, second = _cgf_terms(t, pool, m)
        low = jnp.where(first < target, t, low)
        high = jnp.where(first < target, high, t)
        newton = t + (target - first) / jnp.maximum(second, 1e-300)
        inside = (newton > low) & (newton < high) & jnp.isfinite(newton)
        return (jnp.where(inside, newton, 0.5 * (low + high)), low, high)

    t_hat, low, high = jax.lax.fori_loop(
        0, iterations, refine, (zero, low, high)
    )
    return t_hat


def _lugannani_rice_log_sf(t_hat, cgf, second, observed):
    """log P(S >= observed), in the exponent throughout.

    The correction is applied as ``log1p`` of a Mills-ratio term rather than by
    forming the linear tail and taking its log, which is what keeps this
    meaningful once the tail is far below the smallest representable double.
    """
    w = jnp.sign(t_hat) * jnp.sqrt(jnp.maximum(2.0 * (t_hat * observed - cgf), 0.0))
    u = t_hat * jnp.sqrt(jnp.maximum(second, 1e-300))
    log_sf = jstats.norm.logsf(w)
    log_pdf = jstats.norm.logpdf(w)
    # phi(w)/Phibar(w) * (1/u - 1/w), guarded where w or u touch zero.
    safe_w = jnp.where(jnp.abs(w) < 1e-12, 1e-12, w)
    safe_u = jnp.where(jnp.abs(u) < 1e-12, 1e-12, u)
    ratio = jnp.exp(log_pdf - log_sf) * (1.0 / safe_u - 1.0 / safe_w)
    corrected = log_sf + jnp.log1p(jnp.maximum(ratio, -1.0 + 1e-12))
    # Daniels' limit at the mean: the correction is 0/0 there.
    return jnp.where(jnp.abs(t_hat) < _NEAR_MEAN, jnp.log(0.5), corrected)


@partial(jax.jit, static_argnames=("iterations",))
def saddlepoint_log_tail(
    observed: jnp.ndarray, pool: jnp.ndarray, m: jnp.ndarray, *, iterations: int = 30
) -> jnp.ndarray:
    """log P(S >= observed) for one pair, S a sum of ``m`` draws from ``pool``."""
    t_hat = _solve_saddlepoint(observed, pool, m, iterations=iterations)
    cgf, _, second = _cgf_terms(t_hat, pool, m)
    return _lugannani_rice_log_sf(t_hat, cgf, second, observed)


def saddlepoint_log_two_sided(
    observed: jnp.ndarray, pool: jnp.ndarray, m: jnp.ndarray, *, iterations: int = 30
) -> jnp.ndarray:
    """log P(|S - E S| >= |observed - E S|), the CRT's own two-sided event.

    Both tails are evaluated at their own saddlepoints rather than one being
    doubled: the null is skewed, so the two are not mirror images, and it was
    exactly that asymmetry that made the skew-normal's absolute two-sided event
    lose ranking at low expression.
    """
    n = pool.shape[0]
    mean = m * jnp.mean(pool)
    deviation = jnp.abs(observed - mean)
    upper = saddlepoint_log_tail(mean + deviation, pool, m, iterations=iterations)
    lower_flipped = saddlepoint_log_tail(-(mean - deviation), -pool, m, iterations=iterations)
    _ = n
    return jnp.logaddexp(upper, lower_flipped)


@partial(jax.jit, static_argnames=("iterations",))
def _stratified_log_two_sided_padded(
    observed: jnp.ndarray,
    pool: jnp.ndarray,
    mask: jnp.ndarray,
    selected: jnp.ndarray,
    finite_population_factor: jnp.ndarray,
    *,
    iterations: int = 30,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Two-sided numerator tail for padded, mean-centred stratum pools."""

    threshold = jnp.abs(observed)

    def one_tail(values):
        t_hat, bracketed = _solve_stratified_saddlepoint(
            threshold,
            values,
            mask,
            selected,
            finite_population_factor,
            iterations=iterations,
        )
        cgf, _, second = _stratified_cgf_terms(
            t_hat, values, mask, selected, finite_population_factor
        )
        log_tail = _lugannani_rice_log_sf(t_hat, cgf, second, threshold)
        return log_tail, bracketed & jnp.isfinite(log_tail)

    upper, upper_valid = one_tail(pool)
    lower, lower_valid = one_tail(-pool)
    log_p = jnp.minimum(jnp.logaddexp(upper, lower), 0.0)
    valid = upper_valid & lower_valid & jnp.isfinite(observed)
    return jnp.where(valid, log_p, jnp.nan), valid


def stratified_saddlepoint_log_two_sided(
    observed: np.ndarray,
    pools: tuple[np.ndarray, ...] | list[np.ndarray],
    selected: np.ndarray,
    *,
    finite_population: bool = True,
    iterations: int = 30,
    pad_width: int | None = None,
    require_centered: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the stratified with-replacement SPA for one target.

    Each entry of ``pools`` is ``(cells_in_stratum, genes)`` and must already
    contain the efficient contributions ``c_i``. Empty strata are invalid;
    zero-selection strata are allowed so callers can keep a static padded shape
    across targets. Returns ``(log_p, valid, variance, skewness)``.

    The pools are required to be centred because the current experimental
    integration supports only intercept-only or categorical-stratum nuisance
    designs, where that identity is exact.  Refusing a visibly uncentred pool
    is preferable to silently changing the CRT's two-sided event.
    """

    values = [np.asarray(value, dtype=np.float64) for value in pools]
    chosen = np.asarray(selected, dtype=np.int64).reshape(-1)
    observed_array = np.asarray(observed, dtype=np.float64).reshape(-1)
    if not values or len(values) != chosen.size:
        raise ValueError("pools and selected must describe at least one matching stratum.")
    num_genes = observed_array.size
    if any(value.ndim != 2 or value.shape[1] != num_genes for value in values):
        raise ValueError("every pool must have shape (cells_in_stratum, genes).")
    population = np.asarray([value.shape[0] for value in values], dtype=np.int64)
    if np.any(chosen < 0) or np.any(chosen >= population):
        raise ValueError("each selected count must lie between zero and its pool size (exclusive at the top).")
    if require_centered:
        for value in values:
            scale = np.maximum(np.abs(value).sum(axis=0), 1.0)
            if np.any(np.abs(value.sum(axis=0)) > 1e-8 * scale):
                raise ValueError("efficient-contribution pools must sum to zero within each stratum.")

    width = int(population.max()) if pad_width is None else int(pad_width)
    if width < int(population.max()):
        raise ValueError("pad_width cannot be smaller than the largest stratum pool.")
    padded = np.zeros((len(values), width, num_genes), dtype=np.float64)
    mask = np.zeros((len(values), width), dtype=bool)
    for index, value in enumerate(values):
        padded[index, : value.shape[0]] = value
        mask[index, : value.shape[0]] = True

    fpc = np.ones(chosen.shape, dtype=np.float64)
    if finite_population:
        positive = chosen > 0
        fpc[positive] = (population[positive] - chosen[positive]) / (population[positive] - 1.0)
    log_p, valid = _stratified_log_two_sided_padded(
        jnp.asarray(observed_array),
        jnp.asarray(padded),
        jnp.asarray(mask),
        jnp.asarray(chosen, dtype=jnp.float64),
        jnp.asarray(fpc),
        iterations=iterations,
    )

    second = np.zeros(num_genes, dtype=np.float64)
    third = np.zeros(num_genes, dtype=np.float64)
    for value, count, correction in zip(values, chosen, fpc):
        second += count * correction * np.mean(np.square(value), axis=0)
        third += count * correction**2 * np.mean(np.power(value, 3), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        skewness = third / np.power(second, 1.5)
    return np.asarray(log_p), np.asarray(valid), second, skewness


def fit_stratified_saddlepoint_from_components(
    *,
    score_residual: np.ndarray,
    observation_weight: np.ndarray,
    strata: np.ndarray,
    control_mask: np.ndarray,
    target_cells: dict[int, np.ndarray],
    num_targets: int,
    observed_score: np.ndarray | None = None,
    screen_p_value: float = 0.01,
    gene_block_size: int = 64,
    finite_population: bool = True,
    iterations: int = 30,
) -> SaddlepointTailFit:
    """Fit screened numerator SPAs for categorical-stratum score tests.

    This is intentionally an experimental companion to ``analytic_null``.  It
    uses the exact efficient numerator ``A = sum c_i`` but approximates the CRT
    denominator as fixed, so its tail event is ``P(|A| >= |A_obs|)`` rather than
    the resampled path's ``P(|T| >= |T_obs|)`` with assignment-varying
    information. The observed score's standard-normal tail screens the full
    matrix. Pairs outside ``screen_p_value`` retain that normal approximation
    as an explicit fallback; only candidates enter the tilted-CGF solve, in
    fixed-width gene blocks. This is load-bearing at transcriptome scale: it
    avoids both an all-pairs root find and JAX recompilation for every target's
    distinct candidate count.
    """

    residual = np.asarray(score_residual, dtype=np.float64)
    weight = np.asarray(observation_weight, dtype=np.float64)
    labels = np.asarray(strata).reshape(-1)
    controls = np.asarray(control_mask, dtype=bool).reshape(-1)
    if residual.ndim != 2 or weight.shape != residual.shape:
        raise ValueError("score_residual and observation_weight must share (cells, genes) shape.")
    if labels.shape != (residual.shape[0],) or controls.shape != labels.shape:
        raise ValueError("strata and control_mask must contain one entry per cell.")
    if not 0.0 < screen_p_value <= 1.0:
        raise ValueError("screen_p_value must lie in (0, 1].")
    if gene_block_size < 1:
        raise ValueError("gene_block_size must be positive.")
    _, codes = np.unique(labels, return_inverse=True)
    num_strata = int(codes.max()) + 1
    control_counts = np.bincount(codes[controls], minlength=num_strata)
    maximum_target_counts = np.zeros(num_strata, dtype=np.int64)
    for raw_cells in target_cells.values():
        cells = np.asarray(raw_cells, dtype=np.int64).reshape(-1)
        maximum_target_counts = np.maximum(
            maximum_target_counts,
            np.bincount(codes[cells], minlength=num_strata),
        )
    shared_pad_width = int(np.max(control_counts + maximum_target_counts))
    num_genes = residual.shape[1]
    shape = (num_targets, num_genes)
    if observed_score is None:
        normal_log_p = np.full(shape, np.nan, dtype=np.float64)
        normal_valid = np.zeros(shape, dtype=bool)
        evaluate = np.ones(shape, dtype=bool)
    else:
        score = jnp.asarray(observed_score, dtype=jnp.float64)
        if score.shape != shape:
            raise ValueError(f"observed_score must have shape {shape}; got {score.shape}.")
        normal_log_p = np.asarray(
            jnp.minimum(jnp.log(2.0) + jstats.norm.logsf(jnp.abs(score)), 0.0)
        )
        normal_valid = np.isfinite(normal_log_p)
        evaluate = normal_valid & (normal_log_p <= np.log(screen_p_value))
    log_p = normal_log_p.copy()
    variance = np.where(normal_valid, 1.0, np.nan)
    skewness = np.where(normal_valid, 0.0, np.nan)
    observed_sum = np.full(shape, np.nan, dtype=np.float64)
    max_fraction = np.full(shape, np.nan, dtype=np.float64)
    valid = normal_valid.copy()
    used_fallback = normal_valid.copy()

    for target, raw_cells in target_cells.items():
        cells = np.asarray(raw_cells, dtype=np.int64).reshape(-1)
        if cells.size == 0:
            continue
        pair = controls.copy()
        pair[cells] = True
        pair_rows = np.flatnonzero(pair)
        stratum_rows: list[np.ndarray] = []
        stratum_selected_rows: list[np.ndarray] = []
        fractions: list[float] = []
        for stratum in range(num_strata):
            pool_rows = pair_rows[codes[pair_rows] == stratum]
            selected_rows = cells[codes[cells] == stratum]
            count = int(selected_rows.size)
            if pool_rows.size == 0:
                stratum_rows = []
                break
            if count >= pool_rows.size:
                stratum_rows = []
                break
            stratum_rows.append(pool_rows)
            stratum_selected_rows.append(selected_rows)
            fractions.append(count / pool_rows.size)
        if not stratum_rows:
            continue
        max_fraction[target] = max(fractions)
        candidate = np.flatnonzero(evaluate[target])
        for start in range(0, candidate.size, gene_block_size):
            genes = candidate[start : start + gene_block_size]
            actual = genes.size
            padded_genes = np.zeros(gene_block_size, dtype=np.int64)
            padded_genes[:actual] = genes
            target_pools: list[np.ndarray] = []
            target_contribution = np.zeros(gene_block_size, dtype=np.float64)
            selected_counts: list[int] = []
            for pool_rows, selected_rows in zip(
                stratum_rows, stratum_selected_rows, strict=True
            ):
                information = weight[np.ix_(pool_rows, padded_genes)].sum(axis=0)
                nuisance_score = residual[np.ix_(pool_rows, padded_genes)].sum(axis=0)
                safe_information = np.where(information > 0.0, information, np.nan)
                contribution = residual[np.ix_(pool_rows, padded_genes)] - weight[
                    np.ix_(pool_rows, padded_genes)
                ] * (nuisance_score / safe_information)[None, :]
                target_pools.append(contribution)
                selected_counts.append(int(selected_rows.size))
                positions = np.searchsorted(pool_rows, selected_rows)
                target_contribution += contribution[positions].sum(axis=0)
            fitted_log_p, fitted_valid, fitted_variance, fitted_skewness = (
                stratified_saddlepoint_log_two_sided(
                    target_contribution,
                    target_pools,
                    np.asarray(selected_counts),
                    finite_population=finite_population,
                    iterations=iterations,
                    pad_width=shared_pad_width,
                )
            )
            fitted_valid = (
                fitted_valid[:actual]
                & np.isfinite(fitted_variance[:actual])
                & (fitted_variance[:actual] > 0.0)
            )
            successful_genes = genes[fitted_valid]
            successful_positions = np.flatnonzero(fitted_valid)
            log_p[target, successful_genes] = fitted_log_p[successful_positions]
            variance[target, successful_genes] = fitted_variance[successful_positions]
            skewness[target, successful_genes] = fitted_skewness[successful_positions]
            observed_sum[target, successful_genes] = target_contribution[successful_positions]
            valid[target, successful_genes] = True
            used_fallback[target, successful_genes] = False

    linear = np.exp(np.maximum(log_p, np.log(np.finfo(float).tiny)))
    linear[~valid] = np.nan
    log_p[~valid] = np.nan
    return SaddlepointTailFit(
        p_value=linear,
        log_p_value=log_p,
        null_mean=np.zeros(shape, dtype=np.float64),
        null_variance=variance,
        null_skewness=skewness,
        observed_sum=observed_sum,
        max_sampling_fraction=max_fraction,
        valid=valid,
        used_fallback=used_fallback,
    )



TWO_SIDED_CONVENTIONS = ("symmetric", "equal-tail")


def _check_two_sided(two_sided: str) -> str:
    if two_sided not in TWO_SIDED_CONVENTIONS:
        raise ValueError(f"two_sided must be one of {TWO_SIDED_CONVENTIONS}; got {two_sided!r}.")
    return two_sided


def pearson3_log_two_sided(
    observed: jnp.ndarray,
    mean: jnp.ndarray,
    variance: jnp.ndarray,
    skewness: jnp.ndarray,
    *,
    normal_below: float = 1e-3,
    two_sided: str = "equal-tail",
) -> jnp.ndarray:
    """Two-sided log p-value from the first three cumulants, no draws.

    ``two_sided="equal-tail"`` (the default) is twice the tail on the observed
    side, ``2 min(P(S >= observed), P(S <= observed))``. Under a skewed
    null the two differ: the symmetric event charges the short tail's mass at
    the long tail's threshold and vice versa, so it rejects more often on the
    long (right, for count residuals) side and less often on the short side,
    while the equal-tail p-value rejects each tail exactly half the time.

    A shifted gamma (Pearson type III) matched to mean, variance and skewness:
    with ``k = 4/g^2``, ``theta = sd*g/2`` and ``shift = mean - 2*sd/g`` the
    variable ``shift + Gamma(k, theta)`` reproduces all three exactly, and its
    tail is a regularized incomplete gamma - elementary, and available in JAX.

    This exists as a *screen*. The normal tail it replaces is exact in mean and
    variance and wrong only in shape, which sounds mild until the shape is the
    whole problem: measured against the saddlepoint on the at-scale screen, the
    normal understates the tail by a median 1.06x at |skew| < 0.5 but 29,400x
    at |skew| > 2. Ignoring the third cumulant is what costs that, and the
    third cumulant is already computed.

    Cornish-Fisher is the usual reach here and is the wrong tool: its
    correction ``z - g(z^2-1)/6`` stops being monotone at ``|z| > 3/g``, which
    at the skewness of 7.95 these nulls reach is ``|z| > 0.38``. Pearson III
    has no such breakdown and no family limit - ``k = 4/g^2`` is defined for
    every non-zero skewness.

    Below ``normal_below`` skewness the gamma degenerates (``k`` diverges) and
    the normal tail is used, which is what the gamma converges to anyway.
    """

    _check_two_sided(two_sided)
    sd = jnp.sqrt(jnp.maximum(variance, jnp.finfo(jnp.float64).tiny))
    threshold = jnp.abs(observed)
    if two_sided == "symmetric":
        upper_normal = jstats.norm.logsf((threshold - mean) / sd)
        lower_normal = jstats.norm.logcdf((-threshold - mean) / sd)
        normal = jnp.minimum(jnp.logaddexp(upper_normal, lower_normal), 0.0)
    else:
        standardized = (observed - mean) / sd
        normal = jnp.minimum(
            jnp.log(2.0) + jnp.minimum(jstats.norm.logsf(standardized), jstats.norm.logcdf(standardized)), 0.0
        )

    # Work with a positive skewness by reflecting; the two-sided event is
    # symmetric under it provided the thresholds are reflected too.
    flip = skewness < 0.0
    g = jnp.where(flip, -skewness, skewness)
    g = jnp.maximum(g, normal_below)
    mu = jnp.where(flip, -mean, mean)
    k = 4.0 / jnp.square(g)
    theta = sd * g / 2.0
    shift = mu - 2.0 * sd / g

    def upper(x):
        # P(X >= x) for X = shift + Gamma(k, theta); 1 below the support.
        z = (x - shift) / theta
        return jnp.where(z > 0.0, jnp.log(jnp.clip(gammaincc(k, jnp.maximum(z, 1e-300)), 1e-300, 1.0)), 0.0)

    def lower(x):
        # P(X <= x), which is zero below the support.
        z = (x - shift) / theta
        return jnp.where(z > 0.0, jnp.log(jnp.clip(gammainc(k, jnp.maximum(z, 1e-300)), 1e-300, 1.0)), -jnp.inf)

    if two_sided == "symmetric":
        gamma_log_p = jnp.minimum(jnp.logaddexp(upper(threshold), lower(-threshold)), 0.0)
    else:
        # The observed value reflects with the distribution; the smaller of the
        # two tails at that point is the observed side.
        x = jnp.where(flip, -observed, observed)
        gamma_log_p = jnp.minimum(jnp.log(2.0) + jnp.minimum(upper(x), lower(x)), 0.0)
    usable = (
        jnp.abs(skewness) >= normal_below
    ) & jnp.isfinite(gamma_log_p) & jnp.isfinite(variance) & (variance > 0.0)
    return jnp.where(usable, gamma_log_p, normal)


def _segment_sum_in_pair_blocks(
    contribution: jnp.ndarray,
    cell_index: jnp.ndarray,
    element_index: jnp.ndarray,
    num_elements: int,
    *,
    target_bytes: int = 512 * 1024**2,
) -> jnp.ndarray:
    """Sum each element's cells without materializing the whole gather.

    ``contribution[cell_index]`` is the obvious way to write this and is what
    the segment sum wants, but it builds a ``(pairs, genes)`` tensor first. On
    the Gasperini at-scale screen that is 6,234,344 cell-element pairs by a
    400-gene chunk in float64 - 19.9 GB, against 158 MB on the control subset.
    The shape of the design, not the size of the data, is what decides whether
    it fits, so the control run gives no warning that the full screen will not.

    Accumulating over blocks of pairs gives the identical result at a bounded
    peak. Only the gather is blocked; the sum itself is still exact.
    """

    num_pairs = int(cell_index.shape[0])
    per_pair = int(contribution.shape[1]) * contribution.dtype.itemsize
    block = max(1, min(num_pairs, target_bytes // max(per_pair, 1)))
    total = jnp.zeros((num_elements, contribution.shape[1]), dtype=contribution.dtype)
    for start in range(0, num_pairs, block):
        stop = min(start + block, num_pairs)
        total = total + jax.ops.segment_sum(
            contribution[cell_index[start:stop]],
            element_index[start:stop],
            num_segments=num_elements,
        )
    return total


@partial(jax.jit, static_argnames=("iterations",))
def _high_moi_spa_gene_block(
    observed: jnp.ndarray,
    padded_pool: jnp.ndarray,
    pool_mask: jnp.ndarray,
    selected: jnp.ndarray,
    finite_population_factor: jnp.ndarray,
    gene_indices: jnp.ndarray,
    *,
    iterations: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One element's fixed-width candidate block, entirely on device."""

    block_observed = jnp.take(observed, gene_indices)
    block_pool = jnp.take(padded_pool, gene_indices, axis=2)
    return _stratified_log_two_sided_padded(
        block_observed,
        block_pool,
        pool_mask,
        selected,
        finite_population_factor,
        iterations=iterations,
    )


def fit_high_moi_stratified_saddlepoint(
    *,
    score_residual: np.ndarray,
    observation_weight: np.ndarray,
    nuisance_design: np.ndarray,
    nuisance_information_inverse: np.ndarray,
    nuisance_score: np.ndarray,
    cell_index: np.ndarray,
    element_index: np.ndarray,
    num_elements: int,
    strata: np.ndarray,
    empirical_p_value: np.ndarray | None = None,
    screen_p_value: float = 0.01,
    gene_block_size: int = 64,
    finite_population: bool = True,
    iterations: int = 30,
) -> SaddlepointTailFit:
    """Screened numerator SPA for the arbitrary-nuisance high-MOI CRT.

    The efficient numerator remains a fixed-vector sample sum even with a
    general nuisance design:

        A(a) = sum_i a_i [r_i - w_i z_i' H^-1 (Z'r)].

    Every element draws from the same all-cell stratum pools; only its observed
    sum and per-stratum selected counts differ. A normal approximation with the
    same finite-population variance screens the full matrix. Pairs outside
    ``screen_p_value`` retain that normal tail as an explicit fallback, while
    candidate genes are evaluated in fixed-width blocks so no
    ``elements x cells x genes`` tensor is materialized.

    All numerical kernels run in JAX float64. This is intentional: the method
    exists to distinguish extreme log tails, where a silent float32 downcast
    would erase the accuracy it is meant to add.
    """

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("high-MOI saddlepoint tails require jax_enable_x64=True.")

    residual = jnp.asarray(score_residual, dtype=jnp.float64)
    weight = jnp.asarray(observation_weight, dtype=jnp.float64)
    nuisance = jnp.asarray(nuisance_design, dtype=jnp.float64)
    inverse = jnp.asarray(nuisance_information_inverse, dtype=jnp.float64)
    nuisance_gradient = jnp.asarray(nuisance_score, dtype=jnp.float64)
    cell_device = jnp.asarray(cell_index, dtype=jnp.int32).reshape(-1)
    element_device = jnp.asarray(element_index, dtype=jnp.int32).reshape(-1)
    labels = np.asarray(strata).reshape(-1)
    if residual.ndim != 2 or weight.shape != residual.shape:
        raise ValueError("score_residual and observation_weight must share (cells, genes) shape.")
    num_cells, num_genes = residual.shape
    num_nuisance = nuisance.shape[1]
    if nuisance.shape != (num_cells, num_nuisance) or labels.shape != (num_cells,):
        raise ValueError("nuisance_design and strata must align with the cell axis.")
    if inverse.shape != (num_genes, num_nuisance, num_nuisance):
        raise ValueError("nuisance_information_inverse has the wrong shape.")
    if nuisance_gradient.shape != (num_nuisance, num_genes):
        raise ValueError("nuisance_score has the wrong shape.")
    if cell_device.shape != element_device.shape or bool(
        jnp.any((element_device < 0) | (element_device >= num_elements))
    ):
        raise ValueError("cell_index and element_index must be aligned valid COO coordinates.")
    if not 0.0 < screen_p_value <= 1.0:
        raise ValueError("screen_p_value must lie in (0, 1].")
    if gene_block_size < 1:
        raise ValueError("gene_block_size must be positive.")

    direction = jnp.einsum("gqr,rg->gq", inverse, nuisance_gradient)
    # As a matmul then an elementwise product, not one einsum. The einsum form
    # sums over the nuisance axis last, which lets XLA materialize an
    # (cells, nuisance, genes) intermediate - 4 GB at 205,797 cells, six
    # covariates and a 400-gene chunk. Contracting first keeps every array
    # (cells, genes).
    contribution = residual - weight * (nuisance @ direction.T)
    observed = _segment_sum_in_pair_blocks(
        contribution, cell_device, element_device, num_elements
    )

    _, stratum_codes = np.unique(labels, return_inverse=True)
    num_strata = int(stratum_codes.max()) + 1
    population = np.bincount(stratum_codes, minlength=num_strata).astype(np.int64)
    stratum_device = jnp.asarray(stratum_codes, dtype=jnp.int32)
    population_device = jnp.asarray(population, dtype=jnp.float64)
    selected = (
        jnp.zeros((num_elements, num_strata), dtype=jnp.int32)
        .at[element_device, stratum_device[cell_device]]
        .add(1)
    )
    selected_host = np.asarray(selected)
    if np.any(selected_host >= population[None, :]):
        raise ValueError("every tested element must leave at least one unselected cell per stratum.")
    fraction = selected_host / population[None, :]
    max_fraction_element = np.max(fraction, axis=1)
    max_fraction = np.broadcast_to(max_fraction_element[:, None], (num_elements, num_genes)).copy()

    raw_first = jax.ops.segment_sum(contribution, stratum_device, num_segments=num_strata)
    raw_second = jax.ops.segment_sum(jnp.square(contribution), stratum_device, num_segments=num_strata)
    raw_third = jax.ops.segment_sum(jnp.power(contribution, 3), stratum_device, num_segments=num_strata)
    stratum_mean = raw_first / population_device[:, None]
    stratum_raw_second = raw_second / population_device[:, None]
    stratum_variance = stratum_raw_second - jnp.square(stratum_mean)
    stratum_third = (
        raw_third / population_device[:, None]
        - 3.0 * stratum_mean * stratum_raw_second
        + 2.0 * jnp.power(stratum_mean, 3)
    )
    selected_float = selected.astype(jnp.float64)
    correction = jnp.ones_like(selected_float)
    if finite_population:
        positive = selected > 0
        candidate_correction = (
            population_device[None, :] - selected_float
        ) / jnp.maximum(population_device[None, :] - 1.0, 1.0)
        correction = jnp.where(positive, candidate_correction, correction)
    null_mean = selected_float @ stratum_mean
    null_variance = jnp.einsum(
        "es,es,sg->eg", selected_float, correction, stratum_variance
    )
    null_third = jnp.einsum(
        "es,es,sg->eg", selected_float, jnp.square(correction), stratum_third
    )
    null_skewness = null_third / jnp.power(null_variance, 1.5)
    standard_deviation = jnp.sqrt(null_variance)
    threshold = jnp.abs(observed)
    upper = jstats.norm.logsf((threshold - null_mean) / standard_deviation)
    lower = jstats.norm.logcdf((-threshold - null_mean) / standard_deviation)
    normal_log_p = jnp.minimum(jnp.logaddexp(upper, lower), 0.0)
    normal_valid = jnp.isfinite(normal_log_p) & jnp.isfinite(null_variance) & (null_variance > 0.0)
    # The screen is a union of two tests with complementary failure modes.
    #
    # The normal tail is free but biased by exactly the thing this method
    # exists to handle: measured on the at-scale screen, the saddlepoint sits
    # a median 1.06x above the normal at |skew| < 0.5 and 29,400x above it at
    # |skew| > 2. That bias mostly over-includes, which is safe, but it is
    # systematic rather than bounded.
    #
    # The empirical p carries no such bias - it is a direct Monte Carlo
    # estimate of the same tail - but it is noisy and only exists when the
    # caller resampled. So it is added when offered rather than relied upon:
    # the saddlepoint's whole point is needing no draws, and a screen that
    # required them would hand that property back.
    selected = normal_log_p <= jnp.log(screen_p_value)
    if empirical_p_value is not None:
        empirical = jnp.asarray(empirical_p_value, dtype=jnp.float64)
        if empirical.shape != normal_log_p.shape:
            raise ValueError("empirical_p_value must match the (elements, genes) grid.")
        selected = selected | (jnp.isfinite(empirical) & (empirical <= screen_p_value))
    evaluate_device = normal_valid & selected

    # One padded all-cell pool is shared by every element. Building it once on
    # device costs O(cells * genes), not O(elements * cells * genes). The only
    # host work below is scheduling the sparse candidate blocks selected by the
    # normal screen.
    within_stratum = np.empty(num_cells, dtype=np.int32)
    for stratum in range(num_strata):
        members = np.flatnonzero(stratum_codes == stratum)
        within_stratum[members] = np.arange(members.size, dtype=np.int32)
    shared_pad_width = int(population.max())
    padded_pool = (
        jnp.zeros((num_strata, shared_pad_width, num_genes), dtype=jnp.float64)
        .at[stratum_device, jnp.asarray(within_stratum)]
        .set(contribution)
    )
    pool_mask = (
        jnp.arange(shared_pad_width)[None, :] < jnp.asarray(population)[:, None]
    )

    log_p = np.asarray(normal_log_p).copy()
    valid = np.asarray(normal_valid).copy()
    evaluate = np.asarray(evaluate_device)
    for element in range(num_elements):
        candidate = np.flatnonzero(evaluate[element])
        for start in range(0, candidate.size, gene_block_size):
            genes = candidate[start : start + gene_block_size]
            actual = genes.size
            padded_genes = np.zeros(gene_block_size, dtype=np.int32)
            padded_genes[:actual] = genes
            fitted_log_p, fitted_valid = _high_moi_spa_gene_block(
                observed[element],
                padded_pool,
                pool_mask,
                selected_float[element],
                correction[element],
                jnp.asarray(padded_genes),
                iterations=iterations,
            )
            log_p[element, genes] = np.asarray(fitted_log_p[:actual])
            valid[element, genes] = np.asarray(fitted_valid[:actual])

    null_mean_host = np.asarray(null_mean)
    null_variance_host = np.asarray(null_variance)
    null_skewness_host = np.asarray(null_skewness)
    observed_host = np.asarray(observed)
    used_fallback = valid & ~evaluate
    linear = np.exp(np.maximum(log_p, np.log(np.finfo(float).tiny)))
    linear[~valid] = np.nan
    log_p[~valid] = np.nan
    return SaddlepointTailFit(
        p_value=linear,
        log_p_value=log_p,
        null_mean=null_mean_host,
        null_variance=null_variance_host,
        null_skewness=null_skewness_host,
        observed_sum=observed_host,
        max_sampling_fraction=max_fraction,
        valid=valid,
        used_fallback=used_fallback,
    )


def normal_log_two_sided(observed: jnp.ndarray, pool: jnp.ndarray, m: jnp.ndarray,
                         *, finite_population: bool = True) -> jnp.ndarray:
    """Method-of-moments normal two-sided log tail - the cheap first-line screen.

    Exact in mean and variance including the finite-population correction, and
    wrong only in shape. It is the screen rather than the answer because a
    normal misses the null's skew, but it costs one pass over the pool with no
    root-find, so it can run on every pair.

    Being wrong in shape it is *not* guaranteed anti-conservative - that holds
    where the true null is heavier-tailed than normal, not universally - so a
    screen built on it wants a margin rather than the final threshold.
    """
    n = pool.shape[0]
    mean = m * jnp.mean(pool)
    variance = m * jnp.var(pool)
    if finite_population:
        variance = variance * (n - m) / jnp.maximum(n - 1.0, 1.0)
    z = (observed - mean) / jnp.sqrt(jnp.maximum(variance, 1e-300))
    return jnp.log(2.0) + jstats.norm.logsf(jnp.abs(z))


def sampling_fraction_warning(pool_size: int, m: int, *, limit: float = 0.10) -> str | None:
    """Flag a sampling fraction where the with-replacement CGF stops being safe."""
    fraction = m / max(pool_size, 1)
    if fraction <= limit:
        return None
    return (f"sampling fraction {fraction:.3f} exceeds {limit:g}; the "
            "with-replacement CGF understates the finite-population effect and a "
            "double saddlepoint (Booth & Butler 1990) is the correct treatment.")


# Many targets against one gene's pool: observed and m vary per target, the
# pool is shared, which is how a gene chunk actually arrives.
batched_saddlepoint_log_two_sided = jax.jit(
    jax.vmap(saddlepoint_log_two_sided, in_axes=(0, None, 0))
)
batched_normal_log_two_sided = jax.jit(
    jax.vmap(normal_log_two_sided, in_axes=(0, None, 0))
)


__all__ = [
    "SaddlepointTailFit",
    "batched_normal_log_two_sided",
    "batched_saddlepoint_log_two_sided",
    "fit_high_moi_stratified_saddlepoint",
    "fit_stratified_saddlepoint_from_components",
    "normal_log_two_sided",
    "saddlepoint_log_tail",
    "saddlepoint_log_two_sided",
    "sampling_fraction_warning",
    "stratified_saddlepoint_log_two_sided",
]


def _propensity_cgf_terms(
    t: jnp.ndarray, contribution: jnp.ndarray, logits: jnp.ndarray
):
    """``K``, ``K'`` and ``K''`` at ``t`` for a sum of independent Bernoullis.

    ``contribution`` is ``(cells, columns)`` and ``logits`` is ``(cells,)`` -
    one selection model for every column - or ``(cells, columns)`` with a model
    per column; the returned triple is per column. A column is a gene of one
    element in the per-element layout, or an arbitrary (element, gene) pair in
    the packed layout.

    Under the propensity mechanism cell ``i`` enters the resampled set on its
    own coin, independently of every other cell, so the score is a sum of
    independent terms ``x_i c_i`` with ``x_i ~ Bernoulli(pi_i)`` and

        K(t) = sum_i log(1 - pi_i + pi_i e^{t c_i}).

    Written in the logit ``eta_i = log(pi_i / (1 - pi_i))`` that telescopes to

        K(t) = sum_i [softplus(eta_i + t c_i) - softplus(eta_i)],

    which is what is evaluated here. The two are algebraically identical, but
    this form is bounded for every ``t`` and gives ``K(0) = 0`` exactly rather
    than as a cancellation of large logs, and its derivatives come out as
    plain logistic weights:

        K'(t)  = sum_i c_i sigma(eta_i + t c_i)
        K''(t) = sum_i c_i^2 sigma (1 - sigma).

    At ``t = 0`` those are the exact Bernoulli mean ``sum pi_i c_i`` and
    variance ``sum pi_i (1 - pi_i) c_i^2``.

    Unlike the stratified pool CGF above, this one is *exact*. The stratified
    path samples a fixed count without replacement and models it by adding
    with-replacement CGFs, which is a genuine approximation whose error has to
    be measured. Independent Bernoulli draws have no such gap: this is the
    resampling law the propensity mechanism actually uses, so the only error
    left in the tail is the Lugannani-Rice asymptotic itself.
    """

    logits = _per_pair_logits(logits)
    shifted = logits + t[None, :] * contribution
    weight = jax.nn.sigmoid(shifted)
    cgf = jnp.sum(jax.nn.softplus(shifted) - jax.nn.softplus(logits), axis=0)
    first = jnp.sum(contribution * weight, axis=0)
    second = jnp.sum(jnp.square(contribution) * weight * (1.0 - weight), axis=0)
    return cgf, first, second


def _per_pair_logits(logits: jnp.ndarray) -> jnp.ndarray:
    """Broadcast ``logits`` against a ``(pool, pairs)`` contribution block.

    A ``(pool,)`` vector is one selection model shared by every column - the
    original per-gene layout, where the columns are genes of one element. A
    ``(pool, pairs)`` matrix gives each column its own model, which is what
    lets candidates from different elements or targets share one block.
    """

    return logits if logits.ndim == 2 else logits[:, None]


def _propensity_cgf_derivatives(
    t: jnp.ndarray, contribution: jnp.ndarray, logits: jnp.ndarray
):
    """``K'`` and ``K''`` only - what the root-finder actually consumes.

    The bracket search and every Newton step read the first two derivatives
    and never the CGF value itself, yet the full form pays two softplus passes
    over the whole pool on each call. Only the final Lugannani-Rice evaluation
    needs ``K(t_hat)``, and it calls :func:`_propensity_cgf_terms` once. The
    expressions here are the same ones, so the derivatives are bit-identical.
    """

    shifted = _per_pair_logits(logits) + t[None, :] * contribution
    weight = jax.nn.sigmoid(shifted)
    first = jnp.sum(contribution * weight, axis=0)
    second = jnp.sum(jnp.square(contribution) * weight * (1.0 - weight), axis=0)
    return first, second


def _solve_propensity_saddlepoint(
    target: jnp.ndarray,
    contribution: jnp.ndarray,
    logits: jnp.ndarray,
    *,
    iterations: int,
    expansions: int = 20,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve ``K'(t) = target`` per gene, safeguarded inside a bracket.

    ``K'`` is bounded here in a way the pool version is not: as ``t`` runs to
    infinity every positive contribution is selected with probability one and
    every negative one with probability zero, so

        K'(t) -> sum_{c_i > 0} c_i.

    An observed sum at or beyond that supremum has no saddlepoint at all, and
    the bracket simply fails to close. That is reported rather than papered
    over, because it is not a numerical failure: it means the observed
    statistic sits outside the support the resampling law can produce.
    """

    # The bracket is seeded from the cumulants at the origin rather than from
    # +/-1. Two things follow, and together they cut the search from 80 CGF
    # evaluations to about 20.
    #
    # The origin is one end of the bracket already: K' is increasing and
    # K'(0) is the null mean, so a target above the mean has its saddlepoint
    # at t > 0 and one below it at t < 0. Only the far end has to move, which
    # halves the work the old loop did evaluating both ends every step.
    #
    # And the Newton step from the origin, (target - mean) / variance, is the
    # right *scale* for that end. Doubling from there reaches the root in a
    # handful of steps, where doubling from 1 spent most of its 40 iterations
    # travelling to the correct order of magnitude.
    zero = jnp.zeros_like(target)
    mean, variance = _propensity_cgf_derivatives(zero, contribution, logits)
    step = (target - mean) / jnp.maximum(variance, jnp.finfo(jnp.float64).tiny)
    above = target >= mean
    start = jnp.where(
        jnp.isfinite(step) & (step != 0.0),
        2.0 * step,
        jnp.where(above, 1.0, -1.0),
    )

    def widen(_, bound):
        first, _ = _propensity_cgf_derivatives(bound, contribution, logits)
        short = jnp.where(above, first < target, first > target)
        return jnp.where(short, bound * 2.0, bound)

    bound = jax.lax.fori_loop(0, expansions, widen, start)
    at_bound, _ = _propensity_cgf_derivatives(bound, contribution, logits)
    bracketed = jnp.where(above, at_bound >= target, at_bound <= target)
    low = jnp.minimum(bound, zero)
    high = jnp.maximum(bound, zero)

    def refine(_, state):
        t, low, high = state
        first, second = _propensity_cgf_derivatives(t, contribution, logits)
        low = jnp.where(first < target, t, low)
        high = jnp.where(first < target, high, t)
        newton = t + (target - first) / jnp.maximum(second, jnp.finfo(jnp.float64).tiny)
        inside = (newton > low) & (newton < high) & jnp.isfinite(newton)
        return jnp.where(inside, newton, 0.5 * (low + high)), low, high

    t_hat, _, _ = jax.lax.fori_loop(
        0, iterations, refine, (jnp.zeros_like(target), low, high)
    )
    return t_hat, bracketed


@partial(jax.jit, static_argnames=("iterations", "two_sided"))
def propensity_saddlepoint_log_two_sided(
    observed: jnp.ndarray,
    contribution: jnp.ndarray,
    logits: jnp.ndarray,
    *,
    iterations: int = 30,
    two_sided: str = "equal-tail",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Two-sided log p-value per column under the Bernoulli null.

    ``two_sided="symmetric"`` is ``log P(|S| >= |observed|)``; ``"equal-tail"``
    is twice the tail on the observed side of the null mean,
    ``2 min(P(S >= observed), P(S <= observed))``, each tail at its own
    saddlepoint. The null is skewed, so the two conventions differ: measured on
    Replogle essential's non-targeting pseudo-targets, the symmetric p-value
    rejected the right (up-regulation) tail 4.4 times as often as the left at
    p<0.001 while the total stayed nominal; the equal-tail p-value rejects
    each tail equally by construction.

    ``contribution`` is ``(pool, columns)``; ``logits`` is either ``(pool,)``,
    one selection model for every column, or ``(pool, columns)``, one per
    column. The second layout is how candidates from many elements are packed
    into one block: the per-column arithmetic is identical either way, since
    every reduction runs over the pool axis.

    Both tails are solved at their own saddlepoints rather than one being
    doubled. The null is skewed - that is the entire reason this exists - so
    the two are not mirror images, and the lower tail is obtained by flipping
    the sign of every contribution, which maps ``P(S <= -x)`` to ``P(-S >= x)``
    without touching the selection probabilities.
    """

    _check_two_sided(two_sided)
    threshold = jnp.abs(observed)
    logits = _per_pair_logits(logits)
    log_selected = jax.nn.log_sigmoid(logits)
    log_rejected = jax.nn.log_sigmoid(-logits)

    def one_tail(values, threshold=threshold):
        # The Bernoulli sum has bounded support: every positive contribution
        # selected is the largest value it can take, every negative one
        # selected the smallest. A threshold outside that range has no
        # saddlepoint at all, and the bracket correctly fails to close - which
        # is a statement about the support, not a numerical failure, so it is
        # answered directly rather than invalidating the pair.
        supremum = jnp.sum(jnp.maximum(values, 0.0), axis=0)
        infimum = jnp.sum(jnp.minimum(values, 0.0), axis=0)
        # P(S = supremum): every positive contribution in, every negative out.
        extreme = jnp.sum(
            jnp.where(values > 0.0, log_selected, 0.0)
            + jnp.where(values < 0.0, log_rejected, 0.0),
            axis=0,
        )
        t_hat, bracketed = _solve_propensity_saddlepoint(
            threshold, values, logits, iterations=iterations
        )
        cgf, _, second = _propensity_cgf_terms(t_hat, values, logits)
        solved = _lugannani_rice_log_sf(t_hat, cgf, second, threshold)
        # Strictly past the supremum the event is impossible; exactly at it the
        # event is the single point mass. Conflating the two puts a floor under
        # the tail: for an all-positive null the lower tail's supremum is zero,
        # so every threshold is past it, and charging P(S = 0) there stops the
        # two-sided p-value ever falling below that.
        beyond = threshold > supremum
        at_supremum = threshold >= supremum
        below = threshold <= infimum
        log_tail = jnp.where(
            beyond,
            -jnp.inf,
            jnp.where(at_supremum, extreme, jnp.where(below, 0.0, solved)),
        )
        # -inf is a probability of zero, not a failure, so validity tests for
        # NaN rather than for finiteness.
        return log_tail, (bracketed | at_supremum | below) & ~jnp.isnan(log_tail)

    if two_sided == "symmetric":
        upper, upper_valid = one_tail(contribution)
        lower, lower_valid = one_tail(-contribution)
        log_p = jnp.minimum(jnp.logaddexp(upper, lower), 0.0)
        valid = upper_valid & lower_valid & jnp.isfinite(observed)
        return jnp.where(valid, log_p, jnp.nan), valid
    # Equal-tail: the tail on the observed side of the null mean, doubled. The
    # upper tail is solved at the observed value itself and the lower tail at
    # its negative on the sign-flipped contributions, so each threshold lies on
    # the far side of its own mean and the saddlepoint exists.
    null_mean = jnp.sum(contribution * jax.nn.sigmoid(logits), axis=0)
    upper_side = observed >= null_mean
    upper, upper_valid = one_tail(contribution, observed)
    lower, lower_valid = one_tail(-contribution, -observed)
    log_p = jnp.minimum(jnp.log(2.0) + jnp.where(upper_side, upper, lower), 0.0)
    valid = jnp.where(upper_side, upper_valid, lower_valid) & jnp.isfinite(observed)
    return jnp.where(valid, log_p, jnp.nan), valid


def fit_high_moi_propensity_saddlepoint(
    *,
    score_residual: np.ndarray,
    observation_weight: np.ndarray,
    nuisance_design: np.ndarray,
    nuisance_information_inverse: np.ndarray,
    nuisance_score: np.ndarray,
    cell_index: np.ndarray,
    element_index: np.ndarray,
    num_elements: int,
    propensity_coefficients: np.ndarray,
    propensity_basis: np.ndarray,
    empirical_p_value: np.ndarray | None = None,
    eta_clip: float = 30.0,
    screen_p_value: float = 0.01,
    two_sided: str = "equal-tail",
    gene_block_size: int = 64,
    element_batch_size: int = 64,
    iterations: int = 30,
) -> SaddlepointTailFit:
    """Screened numerator SPA for the propensity high-MOI CRT.

    The efficient numerator is the same fixed-vector sum the stratified path
    uses,

        A(a) = sum_i a_i [r_i - w_i z_i' H^-1 (Z'r)],

    but here the indicator ``a_i`` is an independent Bernoulli draw at the
    element's own fitted selection probability rather than a fixed-count
    without-replacement draw. That makes the CGF exact rather than a
    with-replacement surrogate, so there is no finite-population correction
    and no sampling fraction to report - the field is filled with zeros to
    keep the result type shared.

    The exact Bernoulli cumulants screen the full matrix through a normal
    tail, and only pairs inside ``screen_p_value`` pay for the saddlepoint.
    Screening moments are formed in float32 because they decide candidacy
    rather than any reported number; every candidate's tail is then evaluated
    in float64, which is the accuracy this method exists to provide.
    """
    _check_two_sided(two_sided)

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("high-MOI saddlepoint tails require jax_enable_x64=True.")

    residual = jnp.asarray(score_residual, dtype=jnp.float64)
    weight = jnp.asarray(observation_weight, dtype=jnp.float64)
    nuisance = jnp.asarray(nuisance_design, dtype=jnp.float64)
    inverse = jnp.asarray(nuisance_information_inverse, dtype=jnp.float64)
    nuisance_gradient = jnp.asarray(nuisance_score, dtype=jnp.float64)
    coefficients = jnp.asarray(propensity_coefficients, dtype=jnp.float64)
    basis = jnp.asarray(propensity_basis, dtype=jnp.float64)
    cell_device = jnp.asarray(cell_index, dtype=jnp.int32).reshape(-1)
    element_device = jnp.asarray(element_index, dtype=jnp.int32).reshape(-1)

    if residual.ndim != 2 or weight.shape != residual.shape:
        raise ValueError("score_residual and observation_weight must share (cells, genes) shape.")
    num_cells, num_genes = residual.shape
    num_nuisance = nuisance.shape[1]
    if nuisance.shape != (num_cells, num_nuisance):
        raise ValueError("nuisance_design must align with the cell axis.")
    if inverse.shape != (num_genes, num_nuisance, num_nuisance):
        raise ValueError("nuisance_information_inverse has the wrong shape.")
    if nuisance_gradient.shape != (num_nuisance, num_genes):
        raise ValueError("nuisance_score has the wrong shape.")
    if coefficients.ndim != 2 or coefficients.shape[0] != num_elements:
        raise ValueError("propensity_coefficients must be (elements, basis).")
    if basis.shape != (num_cells, coefficients.shape[1]):
        raise ValueError("propensity_basis must be (cells, basis) matching the coefficients.")
    if not 0.0 < screen_p_value <= 1.0:
        raise ValueError("screen_p_value must lie in (0, 1].")
    if gene_block_size < 1 or element_batch_size < 1:
        raise ValueError("block sizes must be positive.")

    direction = jnp.einsum("gqr,rg->gq", inverse, nuisance_gradient)
    # As a matmul then an elementwise product, not one einsum. The einsum form
    # sums over the nuisance axis last, which lets XLA materialize an
    # (cells, nuisance, genes) intermediate - 4 GB at 205,797 cells, six
    # covariates and a 400-gene chunk. Contracting first keeps every array
    # (cells, genes).
    contribution = residual - weight * (nuisance @ direction.T)
    observed = _segment_sum_in_pair_blocks(
        contribution, cell_device, element_device, num_elements
    )

    # Exact Bernoulli cumulants, in float32 and batched over elements: the
    # expanded (elements, cells) logits are 5.4 GB at genome-wide scale, so
    # they are formed a batch at a time and discarded.
    single = contribution.astype(jnp.float32)
    squared = jnp.square(single)
    cubed = single * squared
    null_mean = np.empty((num_elements, num_genes), dtype=np.float64)
    null_variance = np.empty_like(null_mean)
    null_third = np.empty_like(null_mean)
    for start in range(0, num_elements, element_batch_size):
        stop = min(start + element_batch_size, num_elements)
        logits = jnp.clip(
            coefficients[start:stop] @ basis.T, -eta_clip, eta_clip
        ).astype(jnp.float32)
        selection = jax.nn.sigmoid(logits)
        bernoulli = selection * (1.0 - selection)
        null_mean[start:stop] = np.asarray(selection @ single, dtype=np.float64)
        null_variance[start:stop] = np.asarray(bernoulli @ squared, dtype=np.float64)
        null_third[start:stop] = np.asarray(
            (bernoulli * (1.0 - 2.0 * selection)) @ cubed, dtype=np.float64
        )

    mean_device = jnp.asarray(null_mean)
    variance_device = jnp.asarray(null_variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        skew_host = null_third / np.power(null_variance, 1.5)
    skew_device = jnp.nan_to_num(jnp.asarray(skew_host), nan=0.0, posinf=0.0, neginf=0.0)
    # The screen carries the third cumulant, which is already computed. It also
    # becomes the reported value on every pair the screen does not promote, so
    # this is not only about what gets evaluated: it is the tail those pairs
    # keep.
    normal_log_p = pearson3_log_two_sided(observed, mean_device, variance_device, skew_device, two_sided=two_sided)
    normal_valid = (
        jnp.isfinite(normal_log_p) & jnp.isfinite(variance_device) & (variance_device > 0.0)
    )
    # The screen is a union of two tests with complementary failure modes.
    #
    # The normal tail is free but biased by exactly the thing this method
    # exists to handle: measured on the at-scale screen, the saddlepoint sits
    # a median 1.06x above the normal at |skew| < 0.5 and 29,400x above it at
    # |skew| > 2. That bias mostly over-includes, which is safe, but it is
    # systematic rather than bounded.
    #
    # The empirical p carries no such bias - it is a direct Monte Carlo
    # estimate of the same tail - but it is noisy and only exists when the
    # caller resampled. So it is added when offered rather than relied upon:
    # the saddlepoint's whole point is needing no draws, and a screen that
    # required them would hand that property back.
    selected = normal_log_p <= jnp.log(screen_p_value)
    if empirical_p_value is not None:
        empirical = jnp.asarray(empirical_p_value, dtype=jnp.float64)
        if empirical.shape != normal_log_p.shape:
            raise ValueError("empirical_p_value must match the (elements, genes) grid.")
        selected = selected | (jnp.isfinite(empirical) & (empirical <= screen_p_value))
    evaluate_device = normal_valid & selected

    log_p = np.asarray(normal_log_p).copy()
    valid = np.asarray(normal_valid).copy()
    evaluate = np.asarray(evaluate_device)
    # Candidates are packed across elements: a block holds ``gene_block_size``
    # (element, gene) pairs from wherever the screen promoted them, each column
    # carrying its own element's logits. Blocking per element instead put a
    # floor of one block under every element - a 0.01 screen cost the same as
    # a 0.05 one - and made the block count sum_e ceil(c_e / B) rather than
    # ceil(sum_e c_e / B). The per-column arithmetic is unchanged.
    pair_elements, pair_genes = np.nonzero(evaluate)
    for start in range(0, pair_elements.size, gene_block_size):
        elements = pair_elements[start : start + gene_block_size]
        genes = pair_genes[start : start + gene_block_size]
        count = genes.size
        # A short final block repeats its last pair; duplicates are discarded.
        pad = gene_block_size - count
        padded_elements = jnp.asarray(np.concatenate([elements, np.repeat(elements[-1], pad)]).astype(np.int32))
        padded_genes = jnp.asarray(np.concatenate([genes, np.repeat(genes[-1], pad)]).astype(np.int32))
        block = jnp.take(contribution, padded_genes, axis=1)
        logits = jnp.clip(basis @ jnp.take(coefficients, padded_elements, axis=0).T, -eta_clip, eta_clip)
        fitted_log_p, fitted_valid = propensity_saddlepoint_log_two_sided(
            observed[padded_elements, padded_genes],
            block,
            logits,
            iterations=iterations,
            two_sided=two_sided,
        )
        log_p[elements, genes] = np.asarray(fitted_log_p[:count])
        valid[elements, genes] = np.asarray(fitted_valid[:count])

    used_fallback = valid & ~evaluate
    linear = np.exp(np.maximum(log_p, np.log(np.finfo(float).tiny)))
    linear[~valid] = np.nan
    log_p[~valid] = np.nan
    skewness = skew_host
    return SaddlepointTailFit(
        p_value=linear,
        log_p_value=log_p,
        null_mean=null_mean,
        null_variance=null_variance,
        null_skewness=skewness,
        observed_sum=np.asarray(observed),
        # Independent Bernoulli draws have no finite-population correction to
        # lean on, so there is no sampling fraction to warn about.
        max_sampling_fraction=np.zeros_like(null_mean),
        valid=valid,
        used_fallback=used_fallback,
    )


def fit_low_moi_propensity_saddlepoint(
    *,
    contribution: np.ndarray,
    target_codes: np.ndarray,
    control_mask: np.ndarray,
    shared_logits: np.ndarray,
    intercepts: np.ndarray,
    num_targets: int,
    screen_p_value: float = 0.05,
    two_sided: str = "equal-tail",
    gene_block_size: int = 64,
    iterations: int = 30,
    target_batch_size: int = 1024,
    weight: np.ndarray | None = None,
    nuisance_design: np.ndarray | None = None,
    batch_codes: np.ndarray | None = None,
    control_information: np.ndarray | None = None,
) -> SaddlepointTailFit:
    """Exact-CGF saddlepoint for the low-MOI propensity CRT.

    Pool projection (the ``weight`` group of arguments)
    ---------------------------------------------------
    ``contribution`` is the efficient score under the *control-only* nuisance
    fit. A target's own cells are out of sample for that fit, so their summed
    contribution carries the baseline's estimation error while the resampling
    null, formed from the same contributions over the pool, does not: the
    observed statistic's variance exceeds the null's by a factor that grows with
    ``W_own / W_controls`` (measured: NTC rejection at p<0.05 of 0.074 for 400
    target cells against 2,000 controls, 0.092 against 500). Passing the
    observation ``weight`` (cells, genes) together with the nuisance
    representation - a dense ``nuisance_design`` (cells, q) with
    ``control_information`` (genes, q, q), or categorical ``batch_codes``
    (cells,) with ``control_information`` (batches, genes) - re-projects each
    target's pool as if the nuisance had been fit on the pool:
    ``c_t = c - w Z' e_t`` with ``e_t = (I_controls + I_own)^-1 Z' c_own``. This is
    the first-order refit of the nuisance on the pool, exact for every pool size
    at first order, and it reduces to ``contribution`` unchanged when all of a
    target's cells are themselves controls. The screen's third cumulant keeps
    the correction to second order in ``e_t``; promoted pairs are evaluated
    exactly on the corrected rows. Without ``weight`` the legacy statistic is
    used.

    Each target is tested against its own pool - its cells plus every control -
    and never against cells carrying other perturbations, which have already
    moved and so are not valid counterfactuals for it. The propensity is fitted
    within that pool, so the test stays relative to the controls exactly as the
    permutation version does; only the null changes, from a fixed count within
    coarse strata to independent Bernoulli draws at each cell's own fitted
    probability.

    That change is what makes the CGF exact. The stratified path this replaces
    describes a fixed-count without-replacement draw by summing
    with-replacement stratum CGFs and then correcting the variance by hand, and
    it has to report ``max_sampling_fraction`` so callers can see how hard the
    surrogate is being leaned on. Independent Bernoullis have no such gap: the
    CGF below *is* the resampling law, so the field is filled with zeros and
    the only remaining error is the Lugannani-Rice asymptotic.

    The selection model arrives decomposed - ``shared_logits`` per cell and an
    ``intercepts`` entry per target, NaN where a target has no pool - because
    that decomposition is what lets every target be screened at once. The
    controls are common to every pool, so the three null cumulants over them
    are three matmuls of a ``(targets, controls)`` weight matrix against the
    control contributions, and the few cells a target adds arrive by segment
    sum. Candidates the screen promotes are then packed across targets into
    blocks of ``gene_block_size`` (target, gene) pairs, each column carrying
    its own target's logits, so the block count is ``ceil(candidates / B)``
    rather than one block per target per chunk. Per pair this is the same
    pool, the same logits and the same kernel as evaluating targets one at a
    time; only the order of the pool rows differs (controls first, then the
    target's own cells), which is floating-point rounding, not arithmetic.
    """
    _check_two_sided(two_sided)

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("low-MOI saddlepoint tails require jax_enable_x64=True.")

    # The caller supplies the efficient contributions rather than the pieces to
    # build them, because the low-MOI path carries two incompatible nuisance
    # representations: a dense (genes, q, q) information for a general design,
    # and a (batches, genes) diagonal when the nuisance is a categorical batch.
    # Both reduce to the same (cells, genes) contribution, so that is the
    # interface.
    contribution = jnp.asarray(contribution, dtype=jnp.float64)
    if contribution.ndim != 2:
        raise ValueError("contribution must be (cells, genes).")
    num_cells, num_genes = contribution.shape
    if not 0.0 < screen_p_value <= 1.0:
        raise ValueError("screen_p_value must lie in (0, 1].")
    if gene_block_size < 1 or target_batch_size < 1:
        raise ValueError("block sizes must be positive.")

    codes = np.asarray(target_codes, dtype=np.int64).reshape(-1)
    control = np.asarray(control_mask, dtype=bool).reshape(-1)
    shared = np.asarray(shared_logits, dtype=np.float64).reshape(-1)
    alpha = np.asarray(intercepts, dtype=np.float64).reshape(-1)
    if codes.shape != (num_cells,) or control.shape != (num_cells,) or shared.shape != (num_cells,):
        raise ValueError("target_codes, control_mask and shared_logits must have one entry per cell.")
    if alpha.shape != (num_targets,):
        raise ValueError("intercepts must have one entry per target.")
    if codes.max(initial=-1) >= num_targets:
        raise ValueError("target_codes index beyond num_targets.")

    shape = (num_targets, num_genes)
    target_valid = np.isfinite(alpha)
    # A target with a pool but no cells of its own has no observed sum to test.
    target_valid &= np.bincount(codes[codes >= 0], minlength=num_targets) > 0
    alpha_safe = np.where(target_valid, alpha, 0.0)
    alpha_device = jnp.asarray(alpha_safe)

    # The pool of target t is the controls plus t's own cells. A cell can be
    # both (a control that also carries the target's code); it then enters the
    # pool once, through the control block, and the observed sum still counts
    # it, exactly as the per-target construction did.
    control_rows = np.flatnonzero(control)
    own = (codes >= 0) & ~control
    own_rows = np.flatnonzero(own)
    own_codes = codes[own_rows]
    all_rows = np.flatnonzero(codes >= 0)
    all_codes_device = jnp.asarray(codes[all_rows])
    control_contribution = jnp.take(contribution, jnp.asarray(control_rows), axis=0)
    control_logits = jnp.asarray(shared[control_rows])
    own_contribution = jnp.take(contribution, jnp.asarray(own_rows), axis=0)
    own_logits = jnp.asarray(shared[own_rows]) + alpha_device[jnp.asarray(own_codes)]
    own_codes_device = jnp.asarray(own_codes)

    observed = jax.ops.segment_sum(
        jnp.take(contribution, jnp.asarray(all_rows), axis=0),
        all_codes_device,
        num_segments=num_targets,
    )

    # ---- pool projection -------------------------------------------------
    projection = _LowMoiPoolProjection.build(
        weight=weight,
        nuisance_design=nuisance_design,
        batch_codes=batch_codes,
        control_information=control_information,
        num_cells=num_cells,
        num_genes=num_genes,
        num_targets=num_targets,
        control_rows=control_rows,
        own_rows=own_rows,
        own_codes=own_codes,
        all_rows=all_rows,
        codes_all=codes[all_rows],
        own_contribution=own_contribution,
        target_valid=target_valid,
    )
    if projection is not None:
        own_contribution = projection.corrected_own
        observed = observed - projection.observed_shift

    # Exact Bernoulli cumulants for every target at once. Over the controls
    # they are matmuls; over each target's own cells, segment sums. Batched
    # over targets only to bound the (targets, controls) weight matrices.
    own_selection = jax.nn.sigmoid(own_logits)
    own_bernoulli = own_selection * (1.0 - own_selection)
    own_mean = jax.ops.segment_sum(own_selection[:, None] * own_contribution, own_codes_device, num_segments=num_targets)
    own_variance = jax.ops.segment_sum(
        own_bernoulli[:, None] * jnp.square(own_contribution), own_codes_device, num_segments=num_targets
    )
    own_third = jax.ops.segment_sum(
        (own_bernoulli * (1.0 - 2.0 * own_selection))[:, None] * jnp.power(own_contribution, 3),
        own_codes_device,
        num_segments=num_targets,
    )
    control_square = jnp.square(control_contribution)
    control_cube = control_square * control_contribution
    mean_parts, variance_parts, third_parts = [], [], []
    for start in range(0, num_targets, target_batch_size):
        stop = min(start + target_batch_size, num_targets)
        selection = jax.nn.sigmoid(control_logits[None, :] + alpha_device[start:stop, None])
        bernoulli = selection * (1.0 - selection)
        third_weight = bernoulli * (1.0 - 2.0 * selection)
        batch_mean = selection @ control_contribution
        batch_variance = bernoulli @ control_square
        batch_third = third_weight @ control_cube
        if projection is not None:
            batch_mean, batch_variance, batch_third = projection.correct_control_cumulants(
                slice(start, stop),
                selection,
                bernoulli,
                third_weight,
                control_contribution,
                control_square,
                batch_mean,
                batch_variance,
                batch_third,
            )
        mean_parts.append(batch_mean)
        variance_parts.append(batch_variance)
        third_parts.append(batch_third)
    mean = jnp.concatenate(mean_parts, axis=0) + own_mean
    variance = jnp.concatenate(variance_parts, axis=0) + own_variance
    third = jnp.concatenate(third_parts, axis=0) + own_third

    with np.errstate(divide="ignore", invalid="ignore"):
        skew_host = np.asarray(third) / np.power(np.asarray(variance), 1.5)
    skew_device = jnp.nan_to_num(jnp.asarray(skew_host), nan=0.0, posinf=0.0, neginf=0.0)
    screen_log_p = pearson3_log_two_sided(observed, mean, variance, skew_device, two_sided=two_sided)
    screen_valid = (
        jnp.isfinite(screen_log_p) & jnp.isfinite(variance) & (variance > 0.0)
    ) & jnp.asarray(target_valid)[:, None]
    promote = np.asarray(screen_valid & (screen_log_p <= jnp.log(screen_p_value)))

    log_p = np.where(target_valid[:, None], np.asarray(screen_log_p), np.nan)
    valid = np.array(screen_valid)
    evaluated = np.zeros(shape, dtype=bool)
    observed_host = np.where(target_valid[:, None], np.asarray(observed), np.nan)
    null_mean = np.where(target_valid[:, None], np.asarray(mean), np.nan)
    null_variance = np.where(target_valid[:, None], np.asarray(variance), np.nan)
    null_skewness = np.where(target_valid[:, None], skew_host, np.nan)

    pair_targets, pair_genes = np.nonzero(promote)
    if pair_targets.size:
        # Each target's own cells, padded to a common width with a sentinel row
        # that contributes exactly zero, so one compiled kernel serves every
        # block: pool = controls (shared) followed by the target's own cells.
        own_counts = np.bincount(own_codes, minlength=num_targets)
        width = int(own_counts.max(initial=0))
        sentinel = num_cells
        own_index = np.full((num_targets, max(width, 1)), sentinel, dtype=np.int32)
        order = np.argsort(own_codes, kind="stable")
        starts = np.searchsorted(own_codes[order], np.arange(num_targets), side="left")
        for target in range(num_targets):
            count = own_counts[target]
            if count:
                own_index[target, :count] = own_rows[order[starts[target] : starts[target] + count]]
        own_index_device = jnp.asarray(own_index)
        contribution_ext = jnp.concatenate([contribution, jnp.zeros((1, num_genes), dtype=contribution.dtype)], axis=0)
        shared_ext = jnp.concatenate([jnp.asarray(shared), jnp.zeros((1,), dtype=jnp.float64)])

        for start in range(0, pair_targets.size, gene_block_size):
            targets = pair_targets[start : start + gene_block_size]
            genes = pair_genes[start : start + gene_block_size]
            count = genes.size
            pad = gene_block_size - count
            t_dev = jnp.asarray(np.concatenate([targets, np.repeat(targets[-1], pad)]).astype(np.int32))
            g_dev = jnp.asarray(np.concatenate([genes, np.repeat(genes[-1], pad)]).astype(np.int32))
            block_controls = jnp.take(control_contribution, g_dev, axis=1)
            logits_controls = control_logits[:, None] + alpha_device[t_dev][None, :]
            rows = own_index_device[t_dev]                                   # (pairs, width)
            block_own = contribution_ext[rows, g_dev[:, None]].T             # (width, pairs)
            logits_own = (shared_ext[rows] + alpha_device[t_dev][:, None]).T
            if projection is not None:
                block_controls, block_own = projection.correct_blocks(t_dev, g_dev, rows, block_controls, block_own)
            fitted_log_p, fitted_valid = propensity_saddlepoint_log_two_sided(
                observed[t_dev, g_dev],
                jnp.concatenate([block_controls, block_own], axis=0),
                jnp.concatenate([logits_controls, logits_own], axis=0),
                iterations=iterations,
                two_sided=two_sided,
            )
            log_p[targets, genes] = np.asarray(fitted_log_p[:count])
            valid[targets, genes] = np.asarray(fitted_valid[:count])
            evaluated[targets, genes] = True

    linear = np.exp(np.maximum(log_p, np.log(np.finfo(float).tiny)))
    linear[~valid] = np.nan
    log_p[~valid] = np.nan
    return SaddlepointTailFit(
        p_value=linear,
        log_p_value=log_p,
        null_mean=null_mean,
        null_variance=null_variance,
        null_skewness=null_skewness,
        observed_sum=observed_host,
        # Independent Bernoulli draws carry no finite-population correction, so
        # there is no sampling fraction to warn about.
        max_sampling_fraction=np.zeros(shape, dtype=np.float64),
        valid=valid,
        used_fallback=valid & ~evaluated,
    )


_PROJECTION_ROWS_PER_CHUNK = 64_000_000  # cells x genes per row chunk of per-cell work (~0.5 GB in float64)


class _LowMoiPoolProjection:
    """Per-target re-projection of the efficient score onto the pool's nuisance fit.

    Holds ``e`` (the nuisance shift per target and gene), the resulting shift of
    each own row, the shift of the observed sums, and the control-side moment
    pieces the screen needs. Two representations: dense (``e`` is
    ``(targets, genes, q)`` and a row's shift is ``Z_i' e``) and categorical
    (``e`` is ``(targets, batches, genes)`` and a row's shift is ``e[batch_i]``).
    """

    def __init__(self, *, dense: bool, e, corrected_own, observed_shift, control_weight, control_design,
                 control_batch, weight_ext, design_ext, batch_ext, key_index=None, control_pad_index=None):
        self.dense = dense
        self.corrected_own = corrected_own
        # Dense: ``e`` is (targets, genes, q). Categorical: ``e`` is (K + 1, genes)
        # over the K (target, batch) pairs that own rows occupy, with a zero row
        # at K, and ``key_index`` (targets, batches) maps a pair to its row or K.
        self.e = e
        self.observed_shift = observed_shift
        self.control_weight = control_weight
        self.control_design = control_design
        self.control_batch = control_batch
        self.weight_ext = weight_ext
        self.design_ext = design_ext
        self.batch_ext = batch_ext
        self.key_index = key_index
        # Controls grouped by batch and padded to one width, sentinel = the zero
        # row of ``weight_ext``, so per-batch moments are one batched einsum.
        self.control_pad_index = control_pad_index

    @classmethod
    def build(cls, *, weight, nuisance_design, batch_codes, control_information, num_cells, num_genes, num_targets,
              control_rows, own_rows, own_codes, all_rows, codes_all, own_contribution, target_valid):
        if weight is None:
            if nuisance_design is not None or batch_codes is not None or control_information is not None:
                raise ValueError("The pool projection needs weight together with the nuisance representation.")
            return None
        if control_information is None or (nuisance_design is None) == (batch_codes is None):
            raise ValueError(
                "The pool projection needs control_information and exactly one of nuisance_design or batch_codes."
            )
        # Weights stay in their float32 storage: on GWPS the (cells, genes) chunk is
        # 2.1M x 250, and every float64 copy of it is 4 GB. Per-row work below runs
        # in row chunks and casts each chunk as it goes.
        # Kept in the caller's dtype (float32 from the kernel, float64 in tests);
        # per-row chunks are cast to float64 as they are used.
        weight32 = jnp.asarray(weight)
        if weight32.shape != (num_cells, num_genes):
            raise ValueError("weight must be (cells, genes), matching contribution.")
        own_rows_d = jnp.asarray(own_rows)
        all_rows_d = jnp.asarray(all_rows)
        own_codes_d = jnp.asarray(own_codes)
        all_codes_d = jnp.asarray(codes_all)
        control_weight = jnp.take(weight32, jnp.asarray(control_rows), axis=0).astype(jnp.float64)
        weight_ext = jnp.concatenate([weight32, jnp.zeros((1, num_genes), dtype=weight32.dtype)], axis=0)
        valid = jnp.asarray(target_valid)
        rows_per_chunk = max(1, int(_PROJECTION_ROWS_PER_CHUNK // max(num_genes, 1)))

        def own_weight_chunk(start, stop):
            return jnp.take(weight32, own_rows_d[start:stop], axis=0).astype(jnp.float64)

        def all_weight_chunk(start, stop):
            return jnp.take(weight32, jnp.asarray(all_rows[start:stop]), axis=0).astype(jnp.float64)
        if nuisance_design is not None:
            design = jnp.asarray(nuisance_design, dtype=jnp.float64)
            if design.ndim != 2 or design.shape[0] != num_cells:
                raise ValueError("nuisance_design must be (cells, q).")
            q = int(design.shape[1])
            information = jnp.asarray(control_information, dtype=jnp.float64)
            if information.shape != (num_genes, q, q):
                raise ValueError("control_information must be (genes, q, q) for a dense nuisance design.")
            if q == 0:
                return None
            design_own = jnp.take(design, own_rows_d, axis=0)
            design_all = jnp.take(design, all_rows_d, axis=0)
            # Z' c over each target's own (non-control) rows: (targets, genes, q).
            score = jnp.stack(
                [
                    jax.ops.segment_sum(design_own[:, k, None] * own_contribution, own_codes_d, num_segments=num_targets)
                    for k in range(q)
                ],
                axis=-1,
            )
            # Own-cell information Z' W Z and the solve, in target batches: the
            # (targets, genes, q, q) tensor is 33 GB for 220 targets, 8.5k genes and a
            # 47-level one-hot batch design, so each batch holds at most ~1 GB of it.
            n_own = int(own_rows_d.shape[0])
            own_sorted = np.argsort(np.asarray(own_codes), kind="stable")
            own_codes_sorted = np.asarray(own_codes)[own_sorted]
            target_starts = np.searchsorted(own_codes_sorted, np.arange(num_targets + 1))
            per_target_bytes = num_genes * q * q * 8
            targets_per_batch = max(1, int(1_000_000_000 // max(per_target_bytes, 1)))
            e = jnp.zeros((num_targets, num_genes, q), dtype=jnp.float64)
            for t_start in range(0, num_targets, targets_per_batch):
                t_stop = min(t_start + targets_per_batch, num_targets)
                rows = own_sorted[target_starts[t_start] : target_starts[t_stop]]
                rows_d = jnp.asarray(rows)
                local_codes = jnp.asarray(own_codes_sorted[target_starts[t_start] : target_starts[t_stop]] - t_start)
                n_batch = t_stop - t_start
                own_information = jnp.zeros((n_batch, num_genes, q, q), dtype=jnp.float64)
                for start in range(0, rows.size, rows_per_chunk):
                    stop = min(start + rows_per_chunk, rows.size)
                    w_chunk = jnp.take(weight32, own_rows_d[rows_d[start:stop]], axis=0).astype(jnp.float64)
                    z_chunk = design_own[rows_d[start:stop]]
                    codes_chunk = local_codes[start:stop]
                    for k in range(q):
                        for l in range(k, q):
                            block = jax.ops.segment_sum(
                                (z_chunk[:, k] * z_chunk[:, l])[:, None] * w_chunk, codes_chunk, num_segments=n_batch
                            )
                            own_information = own_information.at[:, :, k, l].add(block)
                            if l != k:
                                own_information = own_information.at[:, :, l, k].add(block)
                pool_information = information[None, :, :, :] + own_information
                e_batch = jnp.linalg.solve(pool_information, score[t_start:t_stop][..., None])[..., 0]
                e = e.at[t_start:t_stop].set(e_batch)
                del own_information, pool_information
            e = jnp.where(valid[:, None, None], e, 0.0)
            e = jnp.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
            corrected_parts = []
            for start in range(0, n_own, rows_per_chunk):
                stop = min(start + rows_per_chunk, n_own)
                shift = jnp.zeros((stop - start, num_genes), dtype=jnp.float64)
                for k in range(q):
                    shift = shift + design_own[start:stop, k, None] * e[own_codes_d[start:stop], :, k]
                corrected_parts.append(own_contribution[start:stop] - own_weight_chunk(start, stop) * shift)
            corrected_own = jnp.concatenate(corrected_parts, axis=0) if corrected_parts else own_contribution
            observed_shift = jnp.zeros((num_targets, num_genes), dtype=jnp.float64)
            n_all = len(all_rows)
            for start in range(0, n_all, rows_per_chunk):
                stop = min(start + rows_per_chunk, n_all)
                w_chunk = all_weight_chunk(start, stop)
                codes_chunk = all_codes_d[start:stop]
                for k in range(q):
                    weighted = jax.ops.segment_sum(design_all[start:stop, k, None] * w_chunk, codes_chunk, num_segments=num_targets)
                    observed_shift = observed_shift + e[:, :, k] * weighted
            design_ext = jnp.concatenate([design, jnp.zeros((1, q), dtype=design.dtype)], axis=0)
            return cls(
                dense=True, e=e, corrected_own=corrected_own, observed_shift=observed_shift,
                control_weight=control_weight, control_design=jnp.take(design, jnp.asarray(control_rows), axis=0),
                control_batch=None, weight_ext=weight_ext, design_ext=design_ext, batch_ext=None,
            )
        codes_b_host = np.asarray(batch_codes, dtype=np.int64).reshape(-1)
        if codes_b_host.shape != (num_cells,):
            raise ValueError("batch_codes must have one entry per cell.")
        information = jnp.asarray(control_information, dtype=jnp.float64)
        if information.ndim != 2 or information.shape[1] != num_genes:
            raise ValueError("control_information must be (batches, genes) for categorical batch codes.")
        num_batches = int(information.shape[0])
        own_batch_host = codes_b_host[own_rows]
        # The (target, batch) pairs that own rows occupy. A dense
        # (targets, batches, genes) array is 5 GB on GWPS (9.8k targets, 267 gem
        # groups); the occupied pairs are a small fraction of it.
        own_key = np.asarray(own_codes, dtype=np.int64) * num_batches + own_batch_host
        unique_keys, inverse = np.unique(own_key, return_inverse=True)
        n_keys = int(unique_keys.size)
        inverse_d = jnp.asarray(inverse.astype(np.int32))
        score = jax.ops.segment_sum(own_contribution, inverse_d, num_segments=max(n_keys, 1))
        n_own = int(own_rows_d.shape[0])
        own_information = jnp.zeros((max(n_keys, 1), num_genes), dtype=jnp.float64)
        for start in range(0, n_own, rows_per_chunk):
            stop = min(start + rows_per_chunk, n_own)
            own_information = own_information + jax.ops.segment_sum(
                own_weight_chunk(start, stop), inverse_d[start:stop], num_segments=max(n_keys, 1)
            )
        key_target = unique_keys // num_batches
        key_batch = unique_keys % num_batches
        pool_information = information[jnp.asarray(key_batch)] + own_information
        e_keys = jnp.where(pool_information > 0.0, score / jnp.where(pool_information > 0.0, pool_information, 1.0), 0.0)
        e_keys = e_keys * valid[jnp.asarray(key_target)][:, None]
        e = jnp.concatenate([e_keys, jnp.zeros((1, num_genes), dtype=jnp.float64)], axis=0)   # row n_keys = absent
        key_index_host = np.full((num_targets, num_batches), n_keys, dtype=np.int32)
        key_index_host[key_target, key_batch] = np.arange(n_keys, dtype=np.int32)
        key_index = jnp.asarray(key_index_host)
        corrected_parts = []
        for start in range(0, n_own, rows_per_chunk):
            stop = min(start + rows_per_chunk, n_own)
            corrected_parts.append(own_contribution[start:stop] - own_weight_chunk(start, stop) * e[inverse_d[start:stop]])
        corrected_own = jnp.concatenate(corrected_parts, axis=0) if corrected_parts else own_contribution
        all_batch_host = codes_b_host[all_rows]
        all_keys_host = key_index_host[np.asarray(codes_all, dtype=np.int64), all_batch_host]
        observed_shift = jnp.zeros((num_targets, num_genes), dtype=jnp.float64)
        n_all = len(all_rows)
        for start in range(0, n_all, rows_per_chunk):
            stop = min(start + rows_per_chunk, n_all)
            observed_shift = observed_shift + jax.ops.segment_sum(
                all_weight_chunk(start, stop) * e[jnp.asarray(all_keys_host[start:stop])],
                all_codes_d[start:stop],
                num_segments=num_targets,
            )
        # Controls grouped by batch, padded to the widest group; the sentinel
        # (num_cells) indexes the zero row of weight_ext, so padding adds nothing.
        control_batch_host = codes_b_host[control_rows]
        order = np.argsort(control_batch_host, kind="stable")
        counts = np.bincount(control_batch_host, minlength=num_batches)
        width = int(counts.max(initial=0))
        pad_index = np.full((num_batches, max(width, 1)), num_cells, dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        control_rows_arr = np.asarray(control_rows, dtype=np.int64)
        for b in range(num_batches):
            if counts[b]:
                pad_index[b, : counts[b]] = control_rows_arr[order[starts[b] : starts[b] + counts[b]]]
        # Positions within the control block (not cell ids) for the selection gather.
        position = np.full(num_cells + 1, len(control_rows_arr), dtype=np.int64)  # sentinel -> extra zero column
        position[control_rows_arr] = np.arange(len(control_rows_arr))
        control_pad_index = {
            "cells": jnp.asarray(pad_index),
            "positions": jnp.asarray(position[pad_index]),
        }
        batch_ext = jnp.concatenate([jnp.asarray(codes_b_host.astype(np.int32)), jnp.zeros((1,), dtype=jnp.int32)])
        return cls(
            dense=False, e=e, corrected_own=corrected_own, observed_shift=observed_shift,
            control_weight=control_weight, control_design=None,
            control_batch=jnp.asarray(control_batch_host.astype(np.int32)), weight_ext=weight_ext, design_ext=None,
            batch_ext=batch_ext, key_index=key_index, control_pad_index=control_pad_index,
        )

    def correct_control_cumulants(self, targets, selection, bernoulli, third_weight, control_contribution,
                                  control_square, mean, variance, third):
        """Cumulants of ``sum_i B_i (c_i - w_i s_i)`` over the controls, ``s_i`` the row's shift.

        Mean and variance exactly; the third cumulant to second order in the
        shift (the cubic term is dropped), which only affects the screen.
        """
        w = self.control_weight
        cw = control_contribution * w
        ww = w * w
        c2w = control_square * w
        cww = control_contribution * ww
        if self.dense:
            e = self.e[targets]                                                     # (batch, genes, q)
            q = e.shape[-1]
            Z = self.control_design
            for k in range(q):
                zk = Z[:, k, None]
                mean = mean - e[:, :, k] * (selection @ (w * zk))
                variance = variance - 2.0 * e[:, :, k] * (bernoulli @ (cw * zk))
                third = third - 3.0 * e[:, :, k] * (third_weight @ (c2w * zk))
                for l in range(q):
                    zkl = zk * Z[:, l, None]
                    variance = variance + e[:, :, k] * e[:, :, l] * (bernoulli @ (ww * zkl))
                    third = third + 3.0 * e[:, :, k] * e[:, :, l] * (third_weight @ (cww * zkl))
            return mean, variance, third
        # Categorical: e_tb (targets, batches, genes) for this target batch from the
        # sparse rows, and per-batch control moments as one batched einsum over
        # controls grouped by batch.
        e_tb = self.e[self.key_index[targets]]                                     # (batch, batches, genes)
        cells = self.control_pad_index["cells"]                                     # (batches, width) cell ids
        positions = self.control_pad_index["positions"]                             # (batches, width) block positions
        zero_col = jnp.zeros((selection.shape[0], 1), dtype=selection.dtype)
        sel_pad = jnp.concatenate([selection, zero_col], axis=1)[:, positions]       # (batch, batches, width)
        bern_pad = jnp.concatenate([bernoulli, zero_col], axis=1)[:, positions]
        third_pad = jnp.concatenate([third_weight, zero_col], axis=1)[:, positions]
        w_ext = self.weight_ext                                                     # (cells + 1, genes), zero sentinel
        # Contributions are indexed by block position, so extend the block with
        # one zero row for the padding sentinel.
        contrib_ext = jnp.concatenate(
            [control_contribution, jnp.zeros((1, control_contribution.shape[1]), dtype=control_contribution.dtype)],
            axis=0,
        )
        w_pad = w_ext[cells].astype(jnp.float64)                                    # (batches, width, genes)
        c_pad = contrib_ext[positions]
        cw_pad = c_pad * w_pad
        ww_pad = w_pad * w_pad
        m_w = jnp.einsum("tbl,blg->tbg", sel_pad, w_pad)
        m_cw = jnp.einsum("tbl,blg->tbg", bern_pad, cw_pad)
        m_ww = jnp.einsum("tbl,blg->tbg", bern_pad, ww_pad)
        m_c2w = jnp.einsum("tbl,blg->tbg", third_pad, c_pad * cw_pad)
        m_cww = jnp.einsum("tbl,blg->tbg", third_pad, cw_pad * w_pad)
        m_www = jnp.einsum("tbl,blg->tbg", third_pad, ww_pad * w_pad)
        mean = mean - jnp.sum(e_tb * m_w, axis=1)
        variance = variance - 2.0 * jnp.sum(e_tb * m_cw, axis=1) + jnp.sum(jnp.square(e_tb) * m_ww, axis=1)
        third = (
            third
            - 3.0 * jnp.sum(e_tb * m_c2w, axis=1)
            + 3.0 * jnp.sum(jnp.square(e_tb) * m_cww, axis=1)
            - jnp.sum(jnp.power(e_tb, 3) * m_www, axis=1)
        )
        return mean, variance, third

    def correct_blocks(self, t_dev, g_dev, rows, block_controls, block_own):
        """Corrected control and own rows for one packed block of (target, gene) pairs."""
        control_weight_block = jnp.take(self.control_weight, g_dev, axis=1)         # (controls, pairs)
        own_weight_block = self.weight_ext[rows, g_dev[:, None]].T.astype(jnp.float64)  # (width, pairs)
        if self.dense:
            e_pairs = self.e[t_dev, g_dev, :]                                       # (pairs, q)
            control_shift = self.control_design @ e_pairs.T                         # (controls, pairs)
            own_shift = jnp.einsum("pwq,pq->pw", self.design_ext[rows], e_pairs).T  # (width, pairs)
        else:
            keys_t = self.key_index[t_dev]                                          # (pairs, batches)
            control_keys = keys_t[:, self.control_batch]                            # (pairs, controls)
            control_shift = self.e[control_keys, g_dev[:, None]].T                  # (controls, pairs)
            own_keys = jnp.take_along_axis(keys_t, self.batch_ext[rows], axis=1)    # (pairs, width)
            own_shift = self.e[own_keys, g_dev[:, None]].T                          # (width, pairs)
        return block_controls - control_weight_block * control_shift, block_own - own_weight_block * own_shift
