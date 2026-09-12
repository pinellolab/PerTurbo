"""Pairwise low-MOI negative-binomial score permutations.

For each target and gene, the nuisance-only NB model is fit once on that
target's cells plus pooled NTC cells. Observed and permuted perturbation
assignments then reuse the null residuals, weights, and nuisance projection.
No optimization occurs inside the resampling loop.

Two backends compute the same statistic:

``per_pair``
    One scipy L-BFGS fit per (target, gene) pair. This is the reference
    implementation: readable, and the thing the batched path is validated
    against.

``batched``
    The null model is a fixed-dispersion NB GLM, so Fisher scoring converges in
    a handful of steps, and the fit batches across genes because only ``y[:, g]``
    and ``theta_g`` change. Score statistics likewise batch across genes and
    resamples as a small number of matrix products. This removes both the
    per-pair optimizer call and the per-pair JAX recompilation that the
    reference path incurs from building a fresh closure each time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from perturbo._internal.analytic_null import (
    DEFAULT_JMAX as ANALYTIC_DEFAULT_JMAX,
    categorical_batch_null_raw_moments,
    intercept_only_null_raw_moments,
)
from perturbo._internal.jax_kernels import (
    append_zero_weight_row,
    batched_efficient_score_from_indices,
    batched_efficient_score_from_indices_categorical_batch,
    categorical_batch_nb_null_residual_and_weight,
    fit_categorical_batch_nb_null_laplace,
    fisher_nb_null,
    nb_null_residual_and_weight,
    prepare_control_only_target_scores,
    prepare_control_only_categorical_batch_target_scores,
    resample_score_reductions,
)
from perturbo._internal.joint_laplace import (
    JointNBDesign,
    _gene_negative_log_posterior,
    _observed_nb_weights,
    _optimize_map,
)
from perturbo._internal.parametric_null import (
    fit_skew_normal_from_moments,
    fit_student_t_from_moments,
)
from perturbo._internal.saddlepoint import (
    fit_low_moi_propensity_saddlepoint,
    fit_stratified_saddlepoint_from_components,
)

_BACKENDS = ("batched", "per_pair", "jax")
_NULL_MODELS = ("control_only", "pooled")

# Matches the clipping in joint_laplace._observed_nb_weights, so the batched and
# reference paths see the same linear predictor.
_ETA_CLIP = 30.0
# Per-iteration cap on the Fisher-scoring step, in nats of the linear predictor.
# Genes whose null MLE does not exist (for instance an all-zero count column)
# would otherwise send eta to -inf and produce NaN for the whole batch; with the
# cap they exhaust the iteration budget and are reported as unconverged.
_MAX_SCORING_STEP = 5.0
# A target-by-resample score gather has shape ``(targets, resamples, cells,
# genes)`` before reduction. Keep this product at the working-set size proven by
# the original one-target, 256-resample JAX implementation.
_JAX_MAX_TARGET_RESAMPLE_BATCH = 256


@dataclass(frozen=True)
class ScorePermutationResult:
    """Observed statistics and calibrated pairwise permutation results."""

    observed_score: jnp.ndarray
    p_value: jnp.ndarray
    q_value: jnp.ndarray
    null_converged: jnp.ndarray
    null_optimizer_iterations: jnp.ndarray
    target_names: tuple[str, ...]
    gene_names: tuple[str, ...]
    num_resamples: int
    method: str
    resampled_scores: jnp.ndarray | None = None
    backend: str = "per_pair"
    null_model: str = "pooled"
    # Host float64, deliberately not JAX arrays; see _parametric_result_fields.
    parametric_p_value: np.ndarray | None = None
    parametric_log_p_value: np.ndarray | None = None
    parametric_q_value: np.ndarray | None = None
    null_mean: np.ndarray | None = None
    null_variance: np.ndarray | None = None
    null_skewness: np.ndarray | None = None
    null_excess_kurtosis: np.ndarray | None = None
    student_t_degrees_of_freedom: np.ndarray | None = None
    parametric_fit_valid: np.ndarray | None = None
    parametric_used_fallback: np.ndarray | None = None
    tail_approximation: str | None = None
    null_moments: ResampledNullMoments | None = None
    saddlepoint_observed_sum: np.ndarray | None = None
    saddlepoint_max_sampling_fraction: np.ndarray | None = None
    tail_fits: dict[str, dict[str, np.ndarray]] | None = None
    """Every requested tail family, keyed by name.

    Each entry holds ``p_value``, ``log_p_value`` and ``valid`` on the same
    ``(targets, genes)`` grid. Families computed in one run share a fit and a
    set of draws, so a difference between them is the tail approximation and
    nothing else - which two separate runs cannot promise, because the
    dispersion fit is not bit-reproducible across them."""


@dataclass(frozen=True)
class ResampledNullMoments:
    """Raw power sums of the resampled null, per (target, gene).

    Every parametric tail family in :mod:`parametric_null` is fitted from these
    and nothing else, so exposing them lets a caller fit *several* families for
    the price of one run. The resampling and its reduction is the expensive
    part; each fit is a cheap host-side pass over already-reduced moments.

    ``count`` is the number of finite resampled statistics actually contributing,
    which is not always ``num_resamples``: a degenerate pair contributes none.
    """

    count: np.ndarray
    sum_score: np.ndarray
    sum_square: np.ndarray
    sum_cube: np.ndarray
    sum_fourth: np.ndarray


@dataclass(frozen=True)
class NullScoreComponents:
    """Quantities shared by observed and resampled score statistics."""

    score_residual: np.ndarray
    observation_weight: np.ndarray
    nuisance_information_inverse: np.ndarray
    nuisance_mean: np.ndarray
    converged: bool
    optimizer_iterations: int


def target_permutation_rng(seed: int, target_name: str) -> np.random.Generator:
    """An independent resampling stream per ``(seed, target)``.

    Keyed on the target's *name* rather than its position, which is what makes a
    target's resamples independent of how the run was decomposed. A shared
    generator consumed in target order gives target ``k`` whatever draws the
    ``k`` targets before it happened to leave behind, so the same target gets
    different resamples depending on how many targets share its chunk - and
    under perturbation chunking that is a function of the chunk-size flag. Two
    runs of the same data at different chunk sizes would then disagree on their
    p-values for no statistical reason.

    Naming makes the stream invariant to chunk size, chunk composition, target
    ordering, and to targets being added or dropped elsewhere in the screen.
    BLAKE2b rather than ``hash()`` because the latter is salted per process and
    would not even reproduce across two runs of the same script.
    """

    digest = hashlib.blake2b(str(target_name).encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(
        np.random.SeedSequence(entropy=(int(seed), int.from_bytes(digest, "big")))
    )


def make_stratified_permutations(
    observed_assignment: np.ndarray | jnp.ndarray,
    *,
    num_resamples: int,
    strata: np.ndarray | jnp.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Permute a binary assignment while preserving target counts per stratum."""

    assignment = np.asarray(observed_assignment, dtype=np.int8).reshape(-1)
    if assignment.size == 0 or np.any((assignment != 0) & (assignment != 1)):
        raise ValueError("observed_assignment must be a non-empty binary vector.")
    if num_resamples < 1:
        raise ValueError("num_resamples must be positive.")
    if strata is None:
        stratum_values = np.zeros(assignment.size, dtype=np.int64)
    else:
        stratum_values = np.asarray(strata).reshape(-1)
        if stratum_values.shape != assignment.shape:
            raise ValueError("strata must contain one value per assignment.")
    if rng is None:
        rng = np.random.default_rng(0)

    permutations = np.empty((num_resamples, assignment.size), dtype=np.int8)
    unique_strata, stratum_codes = np.unique(stratum_values, return_inverse=True)
    for resample_index in range(num_resamples):
        permuted = np.empty_like(assignment)
        for stratum_index in range(unique_strata.size):
            mask = stratum_codes == stratum_index
            permuted[mask] = rng.permutation(assignment[mask])
        permutations[resample_index] = permuted
    return permutations


# A stratum block picks ``selected`` of ``population`` rows. The random-key
# partition below costs O(population) regardless of how few rows it keeps, so it
# is only worth using when the selection is a large enough fraction of the
# block; past this ratio the collision sampler is cheaper. At the ratio itself a
# row expects ``selected / (2 * _DENSE_SELECTION_RATIO)`` collisions, so the
# repair loop stays short on the sparse side of the switch.
_DENSE_SELECTION_RATIO = 8


def _draw_selected_positions(
    rng: np.random.Generator, *, population: int, selected: int, num_rows: int
) -> np.ndarray:
    """``(num_rows, selected)`` positions in ``range(population)``, distinct within a row.

    Every row is a uniformly random ``selected``-subset, drawn one of two ways.

    When the subset is a large fraction of the block, random keys are
    partitioned - simple, branch-free, and O(population) per row.

    When it is a small fraction, that O(population) is almost all waste: the
    Replogle genome-wide screen picks ~180 of ~92,000 pair rows per target, so
    the partition path drew 92 million doubles per target to keep 180 of them
    and spent about 200 minutes on the precompute alone. The sparse path
    instead draws ``selected`` positions with replacement and redraws only the
    duplicates until a row has none, which costs O(num_rows * selected).

    Repairing duplicates in place rather than rejecting the whole row still
    samples uniformly: the repair depends on the drawn *multiset* alone, never
    on labels, so the procedure commutes with any relabeling of
    ``range(population)``, and the only relabeling-invariant distribution over
    subsets of a fixed size is the uniform one.
    """

    if selected * _DENSE_SELECTION_RATIO > population:
        keys = rng.random((num_rows, population))
        return np.argpartition(keys, selected - 1, axis=1)[:, :selected]

    picked = np.sort(rng.integers(0, population, size=(num_rows, selected)), axis=1)
    while True:
        duplicate = np.zeros(picked.shape, dtype=bool)
        duplicate[:, 1:] = picked[:, 1:] == picked[:, :-1]
        num_duplicate = int(duplicate.sum())
        if num_duplicate == 0:
            return picked
        picked[duplicate] = rng.integers(0, population, size=num_duplicate)
        picked.sort(axis=1)


def make_stratified_permutation_indices(
    observed_assignment: np.ndarray | jnp.ndarray,
    *,
    num_resamples: int,
    strata: np.ndarray | jnp.ndarray | None = None,
    rng: np.random.Generator | None = None,
    generation_chunk_size: int = 256,
) -> np.ndarray:
    """Draw stratified permutations directly as selected cell indices.

    Returns ``(num_resamples, num_selected)``. Equivalent in distribution to
    :func:`make_stratified_permutations` followed by ``nonzero``, but it never
    materializes the binary matrix: within each stratum it picks the required
    number of cells via :func:`_draw_selected_positions`, vectorized over
    resamples.

    The binary form costs a full-length Python-loop shuffle per resample, which
    at realistic sizes was about half the per-target resampling time once the
    null fit had been shared across targets.

    The draws are a valid resample, not a reproduction of any earlier run's:
    :func:`_draw_selected_positions` switched sampler for sparse selections, so
    a given seed now yields different (equally valid) permutations than it did
    before that change, and CRT p-values move by Monte Carlo error. Both the
    precomputed and the drawn-on-the-fly paths route through here, so they stay
    consistent with each other.
    """

    assignment = np.asarray(observed_assignment, dtype=np.int8).reshape(-1)
    if assignment.size == 0 or np.any((assignment != 0) & (assignment != 1)):
        raise ValueError("observed_assignment must be a non-empty binary vector.")
    if num_resamples < 1:
        raise ValueError("num_resamples must be positive.")
    if generation_chunk_size < 1:
        raise ValueError("generation_chunk_size must be positive.")
    if strata is None:
        stratum_values = np.zeros(assignment.size, dtype=np.int64)
    else:
        stratum_values = np.asarray(strata).reshape(-1)
        if stratum_values.shape != assignment.shape:
            raise ValueError("strata must contain one value per assignment.")
    if rng is None:
        rng = np.random.default_rng(0)

    unique_strata, stratum_codes = np.unique(stratum_values, return_inverse=True)
    blocks: list[tuple[np.ndarray, int]] = []
    for stratum_index in range(unique_strata.size):
        members = np.flatnonzero(stratum_codes == stratum_index)
        selected = int(assignment[members].sum())
        if selected:
            blocks.append((members, selected))
    if not blocks:
        return np.empty((num_resamples, 0), dtype=np.int64)

    total_selected = sum(count for _, count in blocks)
    indices = np.empty((num_resamples, total_selected), dtype=np.int64)
    for start in range(0, num_resamples, generation_chunk_size):
        stop = min(start + generation_chunk_size, num_resamples)
        column = 0
        for members, selected in blocks:
            if selected == members.size:
                indices[start:stop, column : column + selected] = members[None, :]
            else:
                picked = _draw_selected_positions(
                    rng, population=members.size, selected=selected, num_rows=stop - start
                )
                indices[start:stop, column : column + selected] = members[picked]
            column += selected
    return indices


# Pool lengths are rounded up to a multiple of this before propensity draws, so
# that the sampler's compiled extraction shape is shared across targets.
_PROPENSITY_POOL_PAD = 1024


@dataclass(frozen=True)
class TargetPermutations:
    """Resample indices for every target, in that target's pair-row space.

    Permutations depend only on which cells a target contributes - never on
    genes - so they can be drawn once and reused across gene chunks. Regenerating
    them per chunk is pure duplicated work: at 148 targets and 999 resamples it
    costs about 11 s each time, so a 43-chunk transcriptome-wide run spends
    roughly 8 minutes redrawing identical numbers.
    """

    indices: tuple[np.ndarray, ...]
    num_resamples: int
    seed: int
    resampling_mechanism: str = "permutation"
    pair_rows: tuple[np.ndarray | None, ...] | None = None
    """Row indices of each target's pool when Bernoulli draws were requested.

    Saddlepoint-only runs keep ``None`` entries because the compact coefficient
    model reconstructs its pools without retaining one control-row vector per
    target."""
    pool_logits: tuple[np.ndarray | None, ...] | None = None
    """Fitted selection log-odds over those pool rows, under the propensity
    mechanism. Logits rather than probabilities because the saddlepoint's CGF
    is written in the logit, and a float32 probability cannot carry one near
    the clip - sigmoid(30) rounds to exactly 1.0."""
    shared_logits: np.ndarray | None = None
    """Legacy shared-slope representation retained for old cached callers."""
    pool_intercepts: np.ndarray | None = None
    """Per-target intercept of the selection model, NaN where the target has
    no pool."""
    propensity_coefficients: np.ndarray | None = None
    """Target-specific pool fits in ``propensity_basis`` coordinates.

    Shape ``(targets, basis)``. Unlike the legacy shared-slope decomposition,
    each row is fitted using only controls and that target's own cells, so an
    unrelated target cannot change this target's assignment law."""
    propensity_basis: np.ndarray | None = None
    """Rank-revealing ``(cells, basis)`` design shared by the compact
    target-specific propensity coefficients."""
    _validation_target_names: tuple[str, ...] | None = None
    _validation_num_cells: int | None = None
    _validation_target_codes: np.ndarray | None = None
    _validation_target_design: np.ndarray | None = None
    _validation_control_mask: np.ndarray | None = None
    _validation_source_cell_indices: np.ndarray | None = None
    _validation_strata: np.ndarray | None = None
    _validation_nuisance_design: np.ndarray | None = None
    _validation_design_token: object | None = None
    _validation_draw_resamples: bool = True
    _validation_shared_propensity_coefficients: np.ndarray | None = None

    def __post_init__(self) -> None:
        if any(entry is not None and entry.ndim != 2 for entry in self.indices):
            raise ValueError("Each target's permutation indices must be two-dimensional.")
        for value in (
            self._validation_target_codes,
            self._validation_target_design,
            self._validation_control_mask,
            self._validation_source_cell_indices,
            self._validation_strata,
            self._validation_nuisance_design,
            self._validation_shared_propensity_coefficients,
        ):
            if value is not None:
                value.flags.writeable = False



def _solve_propensity_intercept(
    offsets: np.ndarray, selected: int, *, iterations: int = 60
) -> float:
    """The intercept that makes the fitted probabilities sum to the observed count.

    An unpenalized logistic's intercept score equation already guarantees this,
    and the guarantee is why the fit carries no penalty: the null's mean count
    is then the observed count exactly, which matters for a statistic that is a
    sum over selected cells. Solving for the intercept alone keeps that
    guarantee per target while the covariate coefficients are shared.

    ``sum(sigmoid(alpha + eta))`` is strictly increasing in ``alpha``, so
    bisection is reliable and needs no bracketing search.
    """

    low, high = -60.0, 60.0
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        total = float(np.sum(1.0 / (1.0 + np.exp(-(offsets + middle)))))
        if total < selected:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


_BASIS_SOLVE_TOLERANCE = 1e-4
"""Relative slack allowed when re-expressing a vector in the propensity basis.

The basis is built and stored in float32, so an exactly representable vector
still comes back with a residual of order the float32 epsilon times the
conditioning of the design. A genuine failure - no intercept column, a dropped
direction - leaves a residual of order the vector itself, which is four orders
of magnitude above this.
"""


def _basis_coordinates(basis: np.ndarray, target: np.ndarray, *, what: str) -> np.ndarray:
    """Exact coordinates of ``target`` in the column space of ``basis``.

    ``basis`` is a rank-revealing orthonormal basis for the column space of the
    nuisance design, so anything that *is* a linear combination of nuisance
    columns has exact coordinates in it and the least-squares solve below
    returns them rather than an approximation. Both vectors this is asked for
    qualify: the shared linear predictor is ``nuisance @ beta`` by
    construction, and the all-ones vector is the nuisance design's own
    intercept column.

    A non-zero residual therefore does not mean "close enough" - it means the
    premise failed, and the coefficients built from these coordinates would
    describe a different selection model than the one that was fitted. That
    would corrupt every p-value silently, so it is an error.
    """

    basis = np.asarray(basis, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    # Solved through the (rank x rank) Gram matrix rather than by an SVD of the
    # (cells x rank) basis: the answer is the same least-squares solution, and
    # the cell axis runs to hundreds of thousands.
    coordinates = np.linalg.lstsq(basis.T @ basis, basis.T @ target, rcond=None)[0]
    residual = float(np.max(np.abs(basis @ coordinates - target))) if target.size else 0.0
    scale = max(1.0, float(np.max(np.abs(target))) if target.size else 1.0)
    if residual > _BASIS_SOLVE_TOLERANCE * scale:
        raise ValueError(
            f"The shared propensity {what} does not lie in the span of the nuisance design's "
            f"propensity basis (residual {residual:.3e} against a scale of {scale:.3e}). "
            "The likely causes are a nuisance design whose first column is not an intercept, "
            "or a basis that dropped a direction the shared coefficients use as rank-deficient."
        )
    return coordinates


def _target_propensity_key(seed: int, target_name: str) -> jax.Array:
    """A JAX key stable to target ordering and chunk composition."""

    digest = hashlib.blake2b(str(target_name).encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    key = jax.random.key(int(seed))
    key = jax.random.fold_in(key, np.uint32(value & 0xFFFFFFFF))
    return jax.random.fold_in(key, np.uint32(value >> 32))


def precompute_low_moi_permutations(
    design: JointNBDesign | object,
    *,
    num_resamples: int = 999,
    strata: np.ndarray | jnp.ndarray | None = None,
    seed: int = 0,
    resampling_mechanism: str = "permutation",
    draw_resamples: bool = True,
    propensity_target_batch_size: int = 64,
    shared_propensity_coefficients: np.ndarray | None = None,
    _cache_validation: bool = False,
) -> TargetPermutations:
    """Draw every target's resamples once, for reuse across gene chunks.

    Reproduces exactly the draws :func:`run_low_moi_score_permutations` would
    make on its own: both key their generator on the target's name via
    :func:`target_permutation_rng`, so passing the result back in leaves the
    p-values unchanged.

    ``shared_propensity_coefficients`` are screen-wide logistic selection
    coefficients in the *original* nuisance parametrisation - one entry per
    column of ``design.nuisance_design``, in that order. When they are given
    and the mechanism is ``"propensity"``, the covariate part of every target's
    selection model is theirs and only the intercept is fitted per target. That
    is the same shared-slope estimator as fitting "carries a targeting guide"
    against "carries a control" inside this design, except that the caller
    fitted it over the whole screen: perturbation chunking hands this function
    one chunk at a time, so an in-design fit would make a target's p-value a
    function of which neighbours the chunk-size flag gave it.

    Leaving them ``None`` fits each target's model on its own pool instead,
    which is chunk-invariant too but spends far less data on the covariates.
    """

    target_codes = getattr(design, "target_codes", None)
    control_mask = np.asarray(design.control_mask, dtype=bool)
    if target_codes is None:
        target_design = np.asarray(design.target_design, dtype=np.int8)
    else:
        target_codes = np.asarray(target_codes, dtype=np.int32).reshape(-1)
        if target_codes.shape != (design.num_cells,):
            raise ValueError("target_codes must contain one code per design cell.")
        target_design = None
    if target_design is not None and np.any(target_design.sum(axis=1) > 1):
        raise ValueError("precompute_low_moi_permutations requires a low-MOI target design.")
    if strata is None:
        full_strata = np.zeros(design.num_cells, dtype=np.int64)
    else:
        full_strata = np.asarray(strata).reshape(-1)
        if full_strata.shape != (design.num_cells,):
            raise ValueError("strata must contain one value per design cell.")

    target_names = tuple(design.target_names)
    if len(target_names) != design.num_targets:
        raise ValueError("design.target_names must contain one name per target.")
    if resampling_mechanism not in ("permutation", "propensity"):
        raise ValueError("resampling_mechanism must be 'permutation' or 'propensity'.")
    # Imported here rather than at module scope: high_moi's package __init__
    # pulls in its api, which imports this module back.
    from perturbo._internal.high_moi.resampling import (
        fit_masked_propensity_coefficients,
        propensity_basis,
        propensity_logits_from_coefficients,
        sample_propensity_indices,
    )
    nuisance = np.asarray(design.nuisance_design, dtype=np.float32)
    propensity_coef = None
    propensity_Q = None
    if resampling_mechanism == "propensity":
        if propensity_target_batch_size < 1:
            raise ValueError("propensity_target_batch_size must be positive.")
        # Coefficients share a compact rank-revealing basis, so the saddlepoint
        # can screen targets in batches without retaining an all-target x
        # all-cell logit matrix: it reads a target's logits as
        # ``propensity_coef[t] @ propensity_Q[rows].T``.
        propensity_Q = np.asarray(propensity_basis(nuisance), dtype=np.float32)
        propensity_coef = np.zeros(
            (design.num_targets, propensity_Q.shape[1]), dtype=np.float32
        )
    if propensity_coef is not None and shared_propensity_coefficients is not None:
        # Shared covariate slopes, per-target intercept. Depth, guide load and
        # batch act on the *cell*, not on which guide it happened to receive,
        # so only abundance is target-specific and abundance is the intercept.
        # The slopes were fitted once over the whole screen by the caller,
        # which is what makes them independent of this chunk's composition.
        beta = np.asarray(shared_propensity_coefficients, dtype=np.float64).reshape(-1)
        if beta.shape[0] != nuisance.shape[1]:
            raise ValueError(
                "shared_propensity_coefficients must hold one coefficient per nuisance column "
                f"(got {beta.shape[0]} for {nuisance.shape[1]} columns)."
            )
        eta_shared = nuisance.astype(np.float64) @ beta
        # ``eta_shared`` is a linear combination of nuisance columns and the
        # all-ones vector is the nuisance design's intercept column, so both
        # have exact coordinates in the basis; ``_basis_coordinates`` refuses
        # to proceed on anything less than exact.
        b_shared = _basis_coordinates(propensity_Q, eta_shared, what="linear predictor")
        c_one = _basis_coordinates(
            propensity_Q, np.ones(nuisance.shape[0]), what="intercept direction"
        )
        # Filled for every target here, before the drawing loop: the
        # saddlepoint-only path skips that loop entirely and still reads these.
        for target_index in range(design.num_targets):
            if target_codes is not None:
                own = target_codes == target_index
            else:
                own = target_design[:, target_index] > 0
            pool = control_mask | own
            selected = int(np.count_nonzero(own & pool))
            pool_size = int(np.count_nonzero(pool))
            if selected == 0 or selected == pool_size:
                # Not testable; the drawing loop drops it and the saddlepoint
                # has nothing to evaluate. Leave the row at zero rather than
                # letting the bisection run to its clip.
                continue
            # The intercept that makes the fitted probabilities over this
            # target's pool sum to its observed cell count - the intercept
            # score equation of the unpenalized logistic, kept per target.
            delta = _solve_propensity_intercept(eta_shared[pool], selected)
            propensity_coef[target_index] = (b_shared + delta * c_one).astype(np.float32)
    elif propensity_coef is not None:
        # One model per target, each fitted on exactly the population used by
        # its CRT: controls plus that target's own cells. This makes the fitted
        # assignment law independent of unrelated targets in the same CLI
        # chunk, at the cost of estimating every covariate slope from one
        # target's worth of events.
        for start in range(0, design.num_targets, int(propensity_target_batch_size)):
            stop = min(start + int(propensity_target_batch_size), design.num_targets)
            targets = np.arange(start, stop, dtype=np.int32)
            if target_codes is not None:
                indicators = target_codes[None, :] == targets[:, None]
            else:
                indicators = target_design[:, start:stop].T > 0
            inclusion = control_mask[None, :] | indicators
            propensity_coef[start:stop] = np.asarray(
                fit_masked_propensity_coefficients(
                    indicators,
                    inclusion,
                    propensity_Q,
                )
            )
    pair_rows: list[np.ndarray | None] = []
    pool_logits: list[np.ndarray | None] = []
    drawn: list[np.ndarray | None] = []
    for target_index in range(design.num_targets):
        if target_design is None:
            pair_mask = control_mask | (target_codes == target_index)
            observed_assignment = target_codes[pair_mask] == target_index
        else:
            observed_full = target_design[:, target_index]
            pair_mask = control_mask | (observed_full > 0)
            observed_assignment = observed_full[pair_mask]
        total = int(observed_assignment.sum())
        if total == 0 or total == observed_assignment.size:
            drawn.append(None)
            pair_rows.append(None)
            pool_logits.append(None)
            continue
        if resampling_mechanism == "permutation":
            pair_rows.append(None)
            pool_logits.append(None)
            drawn.append(
                make_stratified_permutation_indices(
                    observed_assignment,
                    num_resamples=num_resamples,
                    strata=full_strata[pair_mask],
                    rng=target_permutation_rng(seed, target_names[target_index]),
                ).astype(np.int32, copy=False)
            )
            continue
        # Model-X on the pool this target is actually tested against: its own
        # cells and the controls. Cells carrying *other* perturbations stay out,
        # because a perturbed cell is not a valid "could have been this target"
        # counterfactual - its expression already moved.
        #
        # Coefficients are target-specific because the actual pool is
        # target-specific. They still share one compact basis, which lets the
        # saddlepoint batch targets and keeps storage proportional to targets x
        # covariates rather than targets x controls.
        if not draw_resamples:
            pair_rows.append(None)
            pool_logits.append(None)
            drawn.append(None)
            continue
        rows = np.flatnonzero(pair_mask)
        logits = np.asarray(
            propensity_logits_from_coefficients(
                propensity_coef[target_index : target_index + 1],
                propensity_Q[rows],
            )
        ).reshape(-1).astype(np.float32, copy=False)
        pair_rows.append(rows.astype(np.int64, copy=False))
        pool_logits.append(logits)
        # Pad the pool to a common length before drawing. The sampler's index
        # extraction is compiled per (pool, width) shape, and every target's
        # pool - the controls plus its own cells - has its own length, so an
        # exact shape recompiled once per target: about a second each, half an
        # hour at two thousand targets. Padding cells carry probability zero,
        # so they are never selected and the draw over the real cells is the
        # same law; the pad value still points one past the real pool.
        padded_pool = int(np.ceil(rows.size / _PROPENSITY_POOL_PAD) * _PROPENSITY_POOL_PAD)
        selection = jnp.zeros(padded_pool, dtype=jnp.float32).at[: rows.size].set(
            jax.nn.sigmoid(jnp.asarray(logits))
        )
        drawn.append(
            sample_propensity_indices(
                selection,
                num_resamples=num_resamples,
                key=_target_propensity_key(seed, target_names[target_index]),
                pad_value=int(rows.size),
            ).astype(np.int32, copy=False)
        )
    return TargetPermutations(
        indices=tuple(drawn),
        num_resamples=num_resamples,
        seed=seed,
        resampling_mechanism=resampling_mechanism,
        pair_rows=tuple(pair_rows) if resampling_mechanism == "propensity" else None,
        pool_logits=tuple(pool_logits) if resampling_mechanism == "propensity" else None,
        shared_logits=None,
        pool_intercepts=None,
        propensity_coefficients=propensity_coef if resampling_mechanism == "propensity" else None,
        propensity_basis=propensity_Q if resampling_mechanism == "propensity" else None,
        _validation_target_names=target_names if _cache_validation else None,
        _validation_num_cells=int(design.num_cells) if _cache_validation else None,
        _validation_target_codes=(
            None
            if not _cache_validation or target_codes is None
            else np.asarray(target_codes, dtype=np.int32).copy()
        ),
        _validation_target_design=(
            None
            if not _cache_validation or target_design is None
            else np.asarray(target_design, dtype=np.int8).copy()
        ),
        _validation_control_mask=(
            np.asarray(control_mask, dtype=bool).copy() if _cache_validation else None
        ),
        _validation_source_cell_indices=(
            np.asarray(
                getattr(design, "source_cell_indices", np.arange(design.num_cells)), dtype=np.int64
            ).copy()
            if _cache_validation
            else None
        ),
        _validation_strata=np.asarray(full_strata).copy() if _cache_validation else None,
        _validation_nuisance_design=(
            None
            if not _cache_validation or getattr(design, "_gene_independent_token", None) is not None
            else np.asarray(nuisance, dtype=np.float32).copy()
        ),
        _validation_design_token=(
            getattr(design, "_gene_independent_token", None) if _cache_validation else None
        ),
        _validation_draw_resamples=bool(draw_resamples),
        _validation_shared_propensity_coefficients=(
            None
            if not _cache_validation or shared_propensity_coefficients is None
            else np.asarray(shared_propensity_coefficients, dtype=np.float64).copy()
        ),
    )


def indices_to_binary_assignments(
    selected_indices: np.ndarray,
    *,
    num_cells: int,
) -> np.ndarray:
    """Scatter selected indices back into a binary assignment matrix."""

    indices = np.asarray(selected_indices, dtype=np.int64)
    assignments = np.zeros((indices.shape[0], num_cells), dtype=np.int8)
    np.put_along_axis(assignments, indices, 1, axis=1)
    return assignments


def efficient_nb_score_statistics(
    assignments: np.ndarray | jnp.ndarray,
    *,
    score_residual: np.ndarray,
    observation_weight: np.ndarray,
    nuisance_design: np.ndarray,
    nuisance_information_inverse: np.ndarray,
) -> np.ndarray:
    """Compute efficient score z-statistics for one or many binary assignments."""

    assignment_matrix = np.asarray(assignments, dtype=np.float64)
    return_vector = assignment_matrix.ndim == 1
    if return_vector:
        assignment_matrix = assignment_matrix[None, :]
    if assignment_matrix.ndim != 2:
        raise ValueError("assignments must be a cells vector or assignments-by-cells matrix.")

    residual = np.asarray(score_residual, dtype=np.float64).reshape(-1)
    weight = np.asarray(observation_weight, dtype=np.float64).reshape(-1)
    nuisance = np.asarray(nuisance_design, dtype=np.float64)
    information_inverse = np.asarray(nuisance_information_inverse, dtype=np.float64)
    num_cells = residual.size
    if assignment_matrix.shape[1] != num_cells:
        raise ValueError("assignments and score_residual must have the same cell dimension.")
    if weight.shape != (num_cells,) or nuisance.ndim != 2 or nuisance.shape[0] != num_cells:
        raise ValueError("Weights and nuisance design must align with the score residual.")
    if information_inverse.shape != (nuisance.shape[1], nuisance.shape[1]):
        raise ValueError("nuisance_information_inverse has the wrong shape.")

    score = assignment_matrix @ residual
    weighted_assignment = assignment_matrix * weight[None, :]
    nuisance_cross = weighted_assignment @ nuisance
    raw_information = np.sum(weighted_assignment * assignment_matrix, axis=1)
    projected_information = np.einsum(
        "bq,qr,br->b",
        nuisance_cross,
        information_inverse,
        nuisance_cross,
        optimize=True,
    )
    efficient_information = raw_information - projected_information
    statistic = np.full(score.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(efficient_information) & (efficient_information > 1e-12)
    statistic[valid] = score[valid] / np.sqrt(efficient_information[valid])
    return statistic[0] if return_vector else statistic


def _fit_null_score_components(
    *,
    counts: np.ndarray,
    nuisance_design: np.ndarray,
    offset: np.ndarray,
    theta: float,
    nuisance_prior_scale: float | None,
    cell_chunk_size: int | None,
    maxiter: int,
    gradient_tolerance: float,
    curvature_jitter: float,
) -> NullScoreComponents:
    y = jnp.asarray(counts, dtype=jnp.float32)
    z = jnp.asarray(nuisance_design, dtype=jnp.float32)
    offset_jax = jnp.asarray(offset, dtype=jnp.float32)
    theta_jax = jnp.asarray(theta, dtype=jnp.float32)
    empty_target_design = jnp.zeros((y.shape[0], 0), dtype=jnp.float32)
    num_nuisance = nuisance_design.shape[1]
    initial = np.zeros(num_nuisance, dtype=np.float64)
    initial[0] = float(np.log(np.mean(np.asarray(y)) + 0.1) - np.mean(offset))

    def objective_jax(parameters: jnp.ndarray) -> jnp.ndarray:
        return _gene_negative_log_posterior(
            parameters,
            counts=y,
            target_design=empty_target_design,
            nuisance_design=z,
            offset=offset_jax,
            theta=theta_jax,
            effect_prior_scale=None,
            nuisance_prior_scale=nuisance_prior_scale,
            cell_chunk_size=cell_chunk_size,
        )

    result = _optimize_map(
        objective_jax,
        initial,
        maxiter=maxiter,
        gradient_tolerance=gradient_tolerance,
    )
    nuisance_map = np.asarray(result.x, dtype=np.float64)
    eta = np.asarray(offset, dtype=np.float64) + np.asarray(nuisance_design, dtype=np.float64) @ nuisance_map
    mean = np.exp(np.clip(eta, -30.0, 30.0))
    theta_value = float(theta)
    score_residual = theta_value * (np.asarray(counts, dtype=np.float64) - mean) / (theta_value + mean)
    weight = _observed_nb_weights(counts, eta, theta_value)
    information = np.asarray(nuisance_design, dtype=np.float64).T @ (
        weight[:, None] * np.asarray(nuisance_design, dtype=np.float64)
    )
    if nuisance_prior_scale is not None:
        information = information + np.eye(num_nuisance) / float(nuisance_prior_scale) ** 2
    information = information + float(curvature_jitter) * np.eye(num_nuisance)
    information_inverse = np.linalg.inv(information)
    return NullScoreComponents(
        score_residual=score_residual,
        observation_weight=weight,
        nuisance_information_inverse=information_inverse,
        nuisance_mean=nuisance_map,
        converged=bool(result.success and np.isfinite(result.fun)),
        optimizer_iterations=int(result.nit),
    )


@dataclass(frozen=True)
class BatchedNullComponents:
    """Null quantities for every gene of one target-vs-control pair."""

    score_residual: np.ndarray
    observation_weight: np.ndarray
    weighted_nuisance: np.ndarray
    nuisance_information_inverse: np.ndarray
    nuisance_mean: np.ndarray
    gradient_norm: np.ndarray
    converged: np.ndarray
    iterations: int


def fit_batched_nb_null(
    counts: np.ndarray,
    *,
    nuisance_design: np.ndarray,
    offsets: np.ndarray,
    theta: np.ndarray,
    nuisance_prior_scale: float | None = None,
    curvature_jitter: float = 1e-8,
    max_iterations: int = 50,
    gradient_tolerance: float = 1e-6,
) -> BatchedNullComponents:
    """Fit the nuisance-only NB null for every gene at once by Fisher scoring.

    With the dispersion held fixed this is an ordinary GLM fit, so the Newton
    step under the expected information is closed form and needs no linesearch.
    That is what lets all genes share one branch-free iteration: the fit differs
    across genes only through ``counts[:, g]`` and ``theta[g]``.

    Deliberately NumPy rather than JAX. Every step is a GEMM or a batched solve,
    so there is little for a fused kernel to win, and the package does not enable
    ``jax_enable_x64`` - under JAX the gradient would be computed in float32,
    where an absolute gradient summed over thousands of cells cannot reach
    ``gradient_tolerance`` at all.

    ``counts`` is ``(cells, genes)``; ``offsets`` is ``(cells, 1)`` or
    ``(cells, genes)``. Returned coefficients are ``(nuisance, genes)``.
    """

    y = np.asarray(counts, dtype=np.float64)
    z = np.asarray(nuisance_design, dtype=np.float64)
    theta_row = np.asarray(theta, dtype=np.float64).reshape(1, -1)
    offset_matrix = np.asarray(offsets, dtype=np.float64)
    if offset_matrix.ndim == 1:
        offset_matrix = offset_matrix[:, None]
    if y.ndim != 2 or z.ndim != 2 or z.shape[0] != y.shape[0]:
        raise ValueError("counts must be (cells, genes) and nuisance_design (cells, nuisance).")
    if theta_row.shape[1] != y.shape[1]:
        raise ValueError("theta must contain one value per gene.")
    if offset_matrix.shape[0] != y.shape[0] or offset_matrix.shape[1] not in {1, y.shape[1]}:
        raise ValueError("offsets must be (cells,), (cells, 1), or (cells, genes).")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")

    num_nuisance = z.shape[1]
    num_genes = y.shape[1]
    prior_precision = 0.0 if nuisance_prior_scale is None else 1.0 / float(nuisance_prior_scale) ** 2
    ridge = (prior_precision + float(curvature_jitter)) * np.eye(num_nuisance)

    beta = np.zeros((num_nuisance, num_genes), dtype=np.float64)
    beta[0] = np.log(y.mean(axis=0) + 0.1) - offset_matrix.mean(axis=0)

    def linear_predictor(coefficients: np.ndarray) -> np.ndarray:
        return offset_matrix + z @ coefficients

    gradient_norm = np.full(num_genes, np.inf)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        mean = np.exp(np.clip(linear_predictor(beta), -_ETA_CLIP, _ETA_CLIP))
        denominator = theta_row + mean
        residual = theta_row * (y - mean) / denominator
        # Expected NB information weight: always positive, so the batched solve
        # below never sees an indefinite matrix.
        weight = theta_row * mean / denominator
        gradient = z.T @ residual - prior_precision * beta
        gradient_norm = np.max(np.abs(gradient), axis=0)
        curvature = np.einsum("nq,ng,nr->gqr", z, weight, z, optimize=True) + ridge
        step = np.linalg.solve(curvature, gradient.T[:, :, None])[:, :, 0].T
        beta = beta + np.clip(step, -_MAX_SCORING_STEP, _MAX_SCORING_STEP)
        if np.max(gradient_norm) <= gradient_tolerance:
            break

    # The score test uses observed rather than expected weights, matching the
    # reference path, so recompute them at the fitted coefficients.
    eta = linear_predictor(beta)
    mean = np.exp(np.clip(eta, -_ETA_CLIP, _ETA_CLIP))
    denominator = theta_row + mean
    score_residual = theta_row * (y - mean) / denominator
    observation_weight = theta_row * (y + theta_row) * mean / np.square(denominator)
    gradient_norm = np.max(np.abs(z.T @ score_residual - prior_precision * beta), axis=0)
    # A gene whose null MLE does not exist walks the fitted mean toward zero. The
    # gradient underflows long before eta hits the clip, so a small gradient
    # alone would read as convergence. An all-zero count column is the exact
    # criterion for the intercept-only case; the saturation check is a backstop
    # for separation induced by the nuisance covariates.
    degenerate = (y.sum(axis=0) <= 0) | np.any(np.abs(eta) >= _ETA_CLIP, axis=0)
    information = np.einsum("nq,ng,nr->gqr", z, observation_weight, z, optimize=True) + ridge
    return BatchedNullComponents(
        score_residual=score_residual,
        observation_weight=observation_weight,
        # Hoisted out of the resampling loop: independent of the assignment.
        weighted_nuisance=np.ascontiguousarray(
            (observation_weight[:, :, None] * z[:, None, :]).reshape(y.shape[0], num_genes * num_nuisance)
        ),
        nuisance_information_inverse=np.linalg.inv(information),
        nuisance_mean=beta,
        gradient_norm=gradient_norm,
        converged=(gradient_norm <= gradient_tolerance) & ~degenerate,
        iterations=int(iterations),
    )


def evaluate_nb_null_components(
    counts: np.ndarray,
    *,
    nuisance_design: np.ndarray,
    offsets: np.ndarray,
    theta: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score residuals and observed weights at given nuisance coefficients.

    Split out from the fit so a control-only null can be evaluated on cells that
    took no part in estimating it.
    """

    y = np.asarray(counts, dtype=np.float64)
    z = np.asarray(nuisance_design, dtype=np.float64)
    theta_row = np.asarray(theta, dtype=np.float64).reshape(1, -1)
    offset_matrix = np.asarray(offsets, dtype=np.float64)
    if offset_matrix.ndim == 1:
        offset_matrix = offset_matrix[:, None]
    eta = offset_matrix + z @ np.asarray(coefficients, dtype=np.float64)
    mean = np.exp(np.clip(eta, -_ETA_CLIP, _ETA_CLIP))
    denominator = theta_row + mean
    residual = theta_row * (y - mean) / denominator
    weight = theta_row * (y + theta_row) * mean / np.square(denominator)
    return residual, weight


def fit_control_only_nb_null(
    counts: np.ndarray,
    *,
    control_mask: np.ndarray,
    nuisance_design: np.ndarray,
    offsets: np.ndarray,
    theta: np.ndarray,
    nuisance_prior_scale: float | None = None,
    curvature_jitter: float = 1e-8,
    max_iterations: int = 50,
    gradient_tolerance: float = 1e-6,
) -> tuple[BatchedNullComponents, np.ndarray, np.ndarray]:
    """Fit one score-test null on control cells and evaluate it everywhere.

    The pooled alternative refits the null per target on controls plus that
    target's cells. Those cell sets overlap almost completely - typically a few
    hundred target cells against thousands of shared controls - so the pooled fit
    repeats nearly the same work once per target.

    Fitting on controls alone is also the cleaner cut. The nuisance estimate then
    cannot depend on the cells whose assignment is being tested, and the
    permutation test stays exact either way: given fixed residuals, weights, and
    nuisance projection, the statistic is a deterministic function of the
    assignment, so the permutation distribution is valid whatever produced those
    fixed quantities. The choice affects efficiency, not validity.

    Returns the control fit plus residuals and observed weights for *all* cells.
    """

    y = np.asarray(counts, dtype=np.float64)
    mask = np.asarray(control_mask, dtype=bool).reshape(-1)
    if mask.shape != (y.shape[0],) or not np.any(mask):
        raise ValueError("control_mask must flag at least one of the supplied cells.")
    z = np.asarray(nuisance_design, dtype=np.float64)
    offset_matrix = np.asarray(offsets, dtype=np.float64)
    if offset_matrix.ndim == 1:
        offset_matrix = offset_matrix[:, None]

    control_null = fit_batched_nb_null(
        y[mask],
        nuisance_design=z[mask],
        offsets=offset_matrix[mask],
        theta=theta,
        nuisance_prior_scale=nuisance_prior_scale,
        curvature_jitter=curvature_jitter,
        max_iterations=max_iterations,
        gradient_tolerance=gradient_tolerance,
    )
    residual, weight = evaluate_nb_null_components(
        y,
        nuisance_design=z,
        offsets=offset_matrix,
        theta=theta,
        coefficients=control_null.nuisance_mean,
    )
    return control_null, residual, weight


_NEWTON_STEP_GENE_BLOCK = 256
"""Genes per block in :func:`newton_step_magnitude`.

Bounds the live curvature tensor at ``block x nuisance^2`` doubles - 146 MiB at
267 nuisance columns - instead of materializing every gene's at once.
"""


def newton_step_magnitude(
    counts: np.ndarray,
    nuisance_design: np.ndarray,
    offsets: np.ndarray,
    theta: np.ndarray,
    beta: np.ndarray,
    *,
    prior_precision: float,
    ridge: np.ndarray,
) -> np.ndarray:
    """Per-gene ``max |Newton step|`` at ``beta``, in nats of the linear predictor.

    One float64 Fisher-scoring step, computed purely as a diagnostic: it is the
    distance from ``beta`` to the nuisance mode measured in the natural metric,
    ``(Z'WZ)^-1 Z'r``. Returning the magnitude rather than a pass/fail lets a
    caller report *how far* off a supplied baseline is, which is what matters
    when the baseline came from somewhere other than this module's own fit.

    Uses the expected NB weight, so the curvature is positive definite by
    construction and the solve never sees an indefinite matrix.
    """

    design = np.asarray(nuisance_design, dtype=np.float64)
    coefficients = np.asarray(beta, dtype=np.float64)
    count_panel = np.asarray(counts)
    offset_panel = np.asarray(offsets, dtype=np.float64)
    if offset_panel.ndim == 1:
        offset_panel = offset_panel[:, None]
    dispersion = np.asarray(theta, dtype=np.float64).reshape(-1)

    # Gene-block every cells-by-genes intermediate as well as the curvature.
    # This matters before the solve even starts: eta, mean, denominator,
    # residual and weight are five full float64 copies of the count panel if
    # they are formed above this loop.
    #
    # A matmul rather than an einsum matters once the nuisance design is wide:
    # the work is cells x genes x nuisance^2, which at
    # 10k control cells, 8k genes and 267 gem-group columns is 5.9e12
    # multiply-adds. np.einsum is not BLAS-backed and runs that single-threaded
    # - measured 1.8 hours against 2.9 minutes for the same arithmetic
    # expressed as Z' (Z * w) - and it materializes the whole
    # (genes, nuisance, nuisance) tensor, 4.4 GiB here, when only one block is
    # ever live.
    num_genes = int(coefficients.shape[1])
    num_nuisance = int(design.shape[1])
    block = max(1, min(num_genes, _NEWTON_STEP_GENE_BLOCK))
    magnitude = np.empty(num_genes, dtype=np.float64)
    curvature = np.empty((block, num_nuisance, num_nuisance), dtype=np.float64)
    for start in range(0, num_genes, block):
        stop = min(start + block, num_genes)
        width = stop - start
        block_counts = np.asarray(count_panel[:, start:stop], dtype=np.float64)
        block_beta = coefficients[:, start:stop]
        block_theta = dispersion[None, start:stop]
        block_offsets = (
            offset_panel if offset_panel.shape[1] == 1 else offset_panel[:, start:stop]
        )
        eta = block_offsets + design @ block_beta
        mean = np.exp(np.clip(eta, -_ETA_CLIP, _ETA_CLIP))
        denominator = block_theta + mean
        residual = block_theta * (block_counts - mean) / denominator
        weight = block_theta * mean / denominator
        gradient = design.T @ residual - prior_precision * block_beta
        for offset in range(width):
            column = weight[:, offset, None]
            curvature[offset] = design.T @ (design * column)
        block_curvature = curvature[:width] + ridge
        step = np.linalg.solve(block_curvature, gradient.T[:, :, None])[:, :, 0].T
        magnitude[start:stop] = np.max(np.abs(step), axis=0)
    return magnitude


def _newton_step_converged(
    counts: np.ndarray,
    nuisance_design: np.ndarray,
    offsets: np.ndarray,
    theta: np.ndarray,
    beta: np.ndarray,
    *,
    prior_precision: float,
    ridge: np.ndarray,
    step_tolerance: float = 1e-4,
) -> np.ndarray:
    """One float64 Newton step at ``beta``, as a trustworthy convergence check.

    An absolute float32 gradient cannot be compared to a fixed tolerance once
    summed over thousands of cells, so the JAX kernels converge on the Newton
    step instead. This recomputes that same step in float64 purely for
    diagnostics, since the fit itself never leaves float32.
    """

    return (
        newton_step_magnitude(
            counts,
            nuisance_design,
            offsets,
            theta,
            beta,
            prior_precision=prior_precision,
            ridge=ridge,
        )
        <= step_tolerance
    )


def _nuisance_information(
    nuisance_design: np.ndarray,
    observation_weight: np.ndarray,
    *,
    ridge: np.ndarray,
) -> np.ndarray:
    """``(genes, nuisance, nuisance)`` weighted nuisance information."""

    # A matmul per gene rather than one einsum: see newton_step_magnitude for
    # why. np.einsum is not BLAS-backed, and the gap grows with the square of
    # the nuisance width.
    num_genes = int(observation_weight.shape[1])
    information = np.empty(
        (num_genes, nuisance_design.shape[1], nuisance_design.shape[1]), dtype=np.float64
    )
    for gene in range(num_genes):
        column = observation_weight[:, gene, None]
        information[gene] = nuisance_design.T @ (nuisance_design * column)
    return information + ridge


def sparse_nb_score_statistics(
    selected_indices: np.ndarray,
    *,
    score_residual: np.ndarray,
    observation_weight: np.ndarray,
    weighted_nuisance: np.ndarray,
    nuisance_information_inverse: np.ndarray,
    nuisance_score: np.ndarray | None = None,
) -> np.ndarray:
    """Score z-statistics for binary assignments given as selected row indices.

    Every term the statistic needs - the score, the raw information, and the
    nuisance cross-product - is a sum over the selected rows, so a dense
    ``(assignments, cells)`` indicator wastes almost all of its work: a low-MOI
    target typically activates a couple of percent of the cells in its pair.
    Expressing the indicator as a CSR matrix with one nonzero per selection makes
    the cost proportional to the selections rather than to the cell count.

    ``selected_indices`` is ``(assignments, selections_per_assignment)``.

    ``nuisance_score`` is ``Z' r`` over the cells being tested, shape
    ``(nuisance, genes)``. The efficient score is
    ``x'r - x'WZ (Z'WZ)^-1 Z'r``, and that second term drops out only when the
    nuisance model was fit on exactly these cells, which makes ``Z' r`` zero by
    the score equation. A control-only null is not fit on the target cells, so
    the term is nonzero and must be subtracted explicitly. Omitting it inflates
    the statistic - by a factor of two or more in practice.
    """

    indices = np.asarray(selected_indices, dtype=np.int64)
    if indices.ndim == 1:
        indices = indices[None, :]
    if indices.ndim != 2 or indices.shape[1] == 0:
        raise ValueError("selected_indices must be (assignments, selections) and non-empty.")
    num_cells = score_residual.shape[0]
    if np.any(indices < 0) or np.any(indices >= num_cells):
        raise ValueError("selected_indices are out of range for the supplied residuals.")

    num_assignments, per_assignment = indices.shape
    num_genes = score_residual.shape[1]
    num_nuisance = nuisance_information_inverse.shape[-1]
    indicator = sparse.csr_matrix(
        (
            np.ones(indices.size, dtype=np.float64),
            indices.reshape(-1),
            np.arange(0, indices.size + 1, per_assignment, dtype=np.int64),
        ),
        shape=(num_assignments, num_cells),
    )
    score = indicator @ score_residual
    # Binary indicator, so the squared indicator used by the raw information term
    # is the indicator itself.
    raw_information = indicator @ observation_weight
    cross = (indicator @ weighted_nuisance).reshape(-1, num_genes, num_nuisance)
    if nuisance_score is not None:
        score = score - np.einsum(
            "bgq,gqr,rg->bg",
            cross,
            nuisance_information_inverse,
            np.asarray(nuisance_score, dtype=np.float64),
            optimize=True,
        )
    projected = np.einsum("bgq,gqr,bgr->bg", cross, nuisance_information_inverse, cross, optimize=True)
    efficient_information = raw_information - projected
    valid = np.isfinite(efficient_information) & (efficient_information > 1e-12)
    safe_information = np.where(valid, efficient_information, 1.0)
    return np.where(valid, score / np.sqrt(safe_information), np.nan)


def batched_nb_score_statistics(
    assignments: np.ndarray,
    *,
    null: BatchedNullComponents,
) -> np.ndarray:
    """Efficient score z-statistics for many assignments and many genes.

    Returns ``(assignments, genes)``. Every term is a matrix product, so a whole
    block of resamples costs three GEMMs regardless of how many genes are in
    flight.
    """

    a = np.asarray(assignments, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != null.score_residual.shape[0]:
        raise ValueError("assignments must be a cells vector or assignments-by-cells matrix.")

    num_genes = null.score_residual.shape[1]
    num_nuisance = null.nuisance_information_inverse.shape[-1]
    score = a @ null.score_residual
    raw_information = (a * a) @ null.observation_weight
    cross = (a @ null.weighted_nuisance).reshape(-1, num_genes, num_nuisance)
    projected = np.einsum("bgq,gqr,bgr->bg", cross, null.nuisance_information_inverse, cross, optimize=True)
    efficient_information = raw_information - projected
    valid = np.isfinite(efficient_information) & (efficient_information > 1e-12)
    safe_information = np.where(valid, efficient_information, 1.0)
    return np.where(valid, score / np.sqrt(safe_information), np.nan)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    flat = values.reshape(-1)
    adjusted = np.full(flat.shape, np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(flat))
    if finite_indices.size == 0:
        return adjusted.reshape(values.shape)
    finite_values = flat[finite_indices]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    scale = finite_values.size / np.arange(1, finite_values.size + 1)
    ranked_adjusted = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    restored = np.empty_like(ranked_adjusted)
    restored[order] = np.clip(ranked_adjusted, 0.0, 1.0)
    adjusted[finite_indices] = restored
    return adjusted.reshape(values.shape)


def _jax_selection_pad_widths(target_sizes: np.ndarray, *, num_buckets: int) -> np.ndarray:
    """Choose a small set of selection widths and assign one to each target.

    A target's selected-cell count is shared by every gene and every CRT
    resample.  JAX therefore only needs one compiled gather shape per target
    *size bucket*, not per target.  The bucket ceilings split the sorted
    positive target sizes into approximately equally populated groups, which
    bounds both recompilations and the average amount of padding.
    """

    sizes = np.asarray(target_sizes, dtype=np.int64).reshape(-1)
    if num_buckets < 1:
        raise ValueError("num_buckets must be positive.")
    if np.any(sizes < 0):
        raise ValueError("target_sizes must be non-negative.")
    positive = np.sort(sizes[sizes > 0])
    if positive.size == 0:
        return np.zeros_like(sizes)

    bucket_count = min(num_buckets, positive.size)
    ceiling_indices = (
        np.ceil(np.arange(1, bucket_count + 1, dtype=np.float64) * positive.size / bucket_count).astype(np.int64) - 1
    )
    ceilings = np.unique(positive[ceiling_indices])
    widths = np.zeros_like(sizes)
    positive_mask = sizes > 0
    widths[positive_mask] = ceilings[np.searchsorted(ceilings, sizes[positive_mask], side="left")]
    return widths


def _pad_jax_indices(
    indices: np.ndarray,
    *,
    rows: int,
    columns: int,
    dummy_index: int,
) -> np.ndarray:
    """Pad index arrays on the host before they enter a JAX computation.

    Doing this with ``jnp.at[...,].set`` merely moves retracing to a sequence of
    variable-width scatter kernels. NumPy padding is negligible beside the score
    calculation and leaves every JAX input shape fixed within a size bucket.
    """

    array = np.asarray(indices, dtype=np.int32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[0] > rows or array.shape[1] > columns:
        raise ValueError("indices do not fit the requested padded JAX shape.")
    padded = np.full((rows, columns), dummy_index, dtype=np.int32)
    padded[: array.shape[0], : array.shape[1]] = array
    return padded


def _run_jax_control_only_score_permutations(
    *,
    design: JointNBDesign,
    target_codes: np.ndarray,
    control_mask: np.ndarray,
    full_strata: np.ndarray,
    num_resamples: int,
    seed: int,
    permutations: TargetPermutations | None,
    resample_chunk_size: int,
    num_cell_buckets: int,
    targets_per_batch: int,
    max_target_resample_batch: int,
    max_gather_bytes: int | None,
    return_resampled_scores: bool,
    accumulate_score_moments: bool,
    score_residual: jnp.ndarray,
    observation_weight: jnp.ndarray,
    weighted_nuisance: jnp.ndarray,
    nuisance_design: jnp.ndarray,
    control_information: jnp.ndarray,
    control_nuisance_score: jnp.ndarray,
    batch_codes: jnp.ndarray | None = None,
    num_batches: int | None = None,
    null_converged: np.ndarray,
    dummy_index: int,
    analytic_null_moments: bool = False,
    observed_only: bool = False,
    analytic_null_jmax: int = ANALYTIC_DEFAULT_JMAX,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
]:
    """Evaluate the control-only CRT in static-shape target-size batches.

    The NumPy backend already batches all genes. This is its JAX analogue: target
    cells and their resamples are packed by cell-count bucket on the host, then
    each bucket runs a small fixed number of compiled target-by-resample kernels.
    ``observed_only`` omits permutation draws and score evaluation while still
    computing the observed target score needed by a pure saddlepoint run.
    """

    num_targets = design.num_targets
    num_genes = design.num_genes
    observed_scores = np.full((num_targets, num_genes), np.nan, dtype=np.float32)
    p_values = np.full_like(observed_scores, np.nan)
    converged = np.zeros((num_targets, num_genes), dtype=bool)
    iterations = np.zeros((num_targets, num_genes), dtype=np.int32)
    stored_resampled = (
        np.full((num_targets, num_genes, num_resamples), np.nan, dtype=np.float32) if return_resampled_scores else None
    )
    moment_count = np.zeros((num_targets, num_genes), dtype=np.int64)
    moment_sum = np.zeros((num_targets, num_genes), dtype=np.float64)
    moment_square = np.zeros((num_targets, num_genes), dtype=np.float64)
    moment_cube = np.zeros((num_targets, num_genes), dtype=np.float64)
    moment_fourth = np.zeros((num_targets, num_genes), dtype=np.float64)
    target_sizes = np.bincount(target_codes[target_codes >= 0], minlength=num_targets)
    # Permutation draws select exactly the observed number of cells, so the
    # observed size is the gather width. Propensity draws are Bernoulli and can
    # select more cells than were observed; their padded rows are also rounded
    # up to a bucket by the sampler. Size each target's width by the widest
    # real draw instead, and trim the trailing all-pad columns off the draws so
    # the padding does not itself set the width.
    gather_sizes = target_sizes.copy()
    trimmed_draw_columns: dict[int, int] = {}
    if (
        permutations is not None
        and permutations.resampling_mechanism == "propensity"
        and permutations.pair_rows is not None
    ):
        for target_index, drawn in enumerate(permutations.indices):
            if drawn is None or permutations.pair_rows[target_index] is None:
                continue
            pad_value = int(permutations.pair_rows[target_index].size)
            real_columns = int((drawn != pad_value).sum(axis=1).max()) if drawn.size else 0
            trimmed_draw_columns[target_index] = real_columns
            gather_sizes[target_index] = max(int(target_sizes[target_index]), real_columns)
    widths = _jax_selection_pad_widths(gather_sizes, num_buckets=num_cell_buckets)
    # Group every target's cells in one sorting pass rather than rescanning the
    # full code vector per target. The old form cost a `flatnonzero` over all
    # cells plus a whole-length `control_mask.copy()` for each of thousands of
    # targets - ~0.9s per gene chunk at transcriptome scale, and pure fixed
    # cost, since it does not fall with the gene count. Stable sorting leaves
    # each target's cells in ascending index order, as `flatnonzero` did.
    order = np.argsort(target_codes, kind="stable")
    sorted_codes = target_codes[order]
    all_targets = np.arange(num_targets)
    target_starts = np.searchsorted(sorted_codes, all_targets, side="left")
    target_stops = np.searchsorted(sorted_codes, all_targets, side="right")
    # A pair pool is degenerate when it holds no target cell, or no control cell
    # outside the target. Both are counts, so neither needs a per-target mask.
    num_controls = int(control_mask.sum())
    controls_outside = num_controls - np.bincount(
        target_codes[control_mask & (target_codes >= 0)], minlength=num_targets
    )

    target_names = tuple(design.target_names)
    if len(target_names) != num_targets:
        raise ValueError("design.target_names must contain one name per target.")
    grouped: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    analytic_target_cells: dict[int, np.ndarray] = {}
    for target_index in range(num_targets):
        target_rows = order[target_starts[target_index] : target_stops[target_index]].astype(np.int32, copy=False)
        if target_rows.size == 0 or controls_outside[target_index] == 0:
            continue
        if analytic_null_moments or observed_only:
            # Neither analytic moments nor a pure SPA needs resampled
            # assignments. Both use only the observed target rows here.
            if analytic_null_moments:
                analytic_target_cells[target_index] = target_rows.astype(np.int64)
            grouped.setdefault(int(widths[target_index]), []).append(
                (target_index, target_rows, np.zeros((0, 0), dtype=np.int32))
            )
            continue
        pair_mask = control_mask.copy()
        pair_mask[target_rows] = True
        observed_assignment = target_codes[pair_mask] == target_index
        if permutations is None:
            resampled_indices = make_stratified_permutation_indices(
                observed_assignment,
                num_resamples=num_resamples,
                strata=full_strata[pair_mask],
                rng=target_permutation_rng(seed, target_names[target_index]),
            )
        else:
            resampled_indices = permutations.indices[target_index]
            if resampled_indices is None:
                continue
            if target_index in trimmed_draw_columns:
                resampled_indices = resampled_indices[:, : trimmed_draw_columns[target_index]]
        pair_rows = np.flatnonzero(pair_mask)
        # Propensity draws are ragged: each resample selects a different number
        # of pool cells, so the rows are padded with ``pair_rows.size`` - one
        # past the pool. Point that padding at the kernel's zero-contribution
        # sentinel row rather than indexing off the end of the pool.
        pool_rows = np.append(pair_rows, dummy_index)
        grouped.setdefault(int(widths[target_index]), []).append(
            (
                target_index,
                target_rows,
                pool_rows[resampled_indices].astype(np.int32, copy=False),
            )
        )

    num_nuisance = nuisance_design.shape[1]
    for width, entries in grouped.items():
        # ``target x resample`` alone does not bound memory: each gathered
        # block is also ``width x genes``. A rare, very large perturbation can
        # otherwise turn an apparently modest microbatch into a multi-GiB
        # allocation. The score kernel gathers ``2 + num_nuisance`` such
        # tensors concurrently before reducing them - the residual, the weight,
        # and the nuisance cross-term, whose last axis is ``genes * nuisance``
        # rather than ``genes`` - so sizing against one tensor understates the
        # peak by 4x at two nuisance columns and blows the caller's budget.
        #
        # The budget is therefore spent per width bucket, not once globally: a
        # single batch size has to be safe for the widest bucket, and since
        # ``_jax_selection_pad_widths`` makes the buckets roughly equally
        # populated, that strands most targets in narrow buckets at a batch
        # sized for the widest one.
        per_target_bytes = (2 + num_nuisance) * width * num_genes * np.dtype(np.float32).itemsize
        if analytic_null_moments:
            # The closed-form null draws nothing, so the resample axis is absent
            # from every gathered tensor and the whole budget goes to targets.
            # At transcriptome scale this is the difference between a few kernel
            # launches per gene chunk and one per four targets, and the launch
            # count does not fall with the gene count - it is pure fixed cost.
            if max_gather_bytes is None:
                # No explicit cap: match the working set the resampled path
                # would have allocated for this width, which is bounded by
                # construction rather than by the caller remembering to pass one.
                batch_size = max_target_resample_batch
            else:
                batch_size = int(max_gather_bytes // max(per_target_bytes, 1))
            batch_size = int(np.clip(batch_size, 1, len(entries)))
            effective_resample_chunk_size = resample_chunk_size
        else:
            # Pure SPA has no resample axis, but it keeps the ordinary small
            # target batches. A giant analytic-style batch makes the observed
            # score compilation unnecessarily expensive on CPU and does not
            # improve the subsequent target-specific CGF work.
            batch_size = targets_per_batch
            resamples_by_target_batch = max(1, max_target_resample_batch // batch_size)
            if max_gather_bytes is None:
                resamples_by_memory = resamples_by_target_batch
            else:
                resamples_by_memory = max(1, max_gather_bytes // max(batch_size * per_target_bytes, 1))
            effective_resample_chunk_size = min(resample_chunk_size, resamples_by_target_batch, resamples_by_memory)
        for batch_start in range(0, len(entries), batch_size):
            batch_entries = entries[batch_start : batch_start + batch_size]
            target_indices = np.full((batch_size, width), dummy_index, dtype=np.int32)
            for local_index, (_, target_rows, _) in enumerate(batch_entries):
                target_indices[local_index] = _pad_jax_indices(
                    target_rows,
                    rows=1,
                    columns=width,
                    dummy_index=dummy_index,
                )[0]
            if batch_codes is None:
                information_inverse, nuisance_score, observed = prepare_control_only_target_scores(
                    jnp.asarray(target_indices),
                    score_residual,
                    observation_weight,
                    None if observed_only else weighted_nuisance,
                    nuisance_design,
                    control_information,
                    control_nuisance_score,
                )
            else:
                if num_batches is None:
                    raise ValueError("num_batches is required for categorical batch scores.")
                information_inverse, nuisance_score, observed = prepare_control_only_categorical_batch_target_scores(
                    jnp.asarray(target_indices),
                    score_residual,
                    observation_weight,
                    batch_codes,
                    control_information,
                    control_nuisance_score,
                    num_batches=num_batches,
                )
            observed_device = observed
            observed = np.asarray(observed, dtype=np.float64)
            if analytic_null_moments or observed_only:
                # The observed statistic is still the kernel's; only its null
                # reference set is replaced, or it is supplied directly to SPA.
                # No empirical p-value exists in either case.
                finite_observed = np.isfinite(observed[: len(batch_entries)])
                for local_index, (target_index, _, _) in enumerate(batch_entries):
                    observed_scores[target_index, finite_observed[local_index]] = observed[
                        local_index, finite_observed[local_index]
                    ]
                    converged[target_index] = null_converged
                continue
            kept = len(batch_entries)
            exceedances = np.zeros((kept, num_genes), dtype=np.int64)
            finite_resamples = np.zeros_like(exceedances)
            # Power sums are accumulated across blocks in float64 although the
            # kernel reduces each block in float32; see resample_score_reductions.
            power_sums = np.zeros((4, kept, num_genes), dtype=np.float64) if accumulate_score_moments else None
            for start in range(0, num_resamples, effective_resample_chunk_size):
                stop = min(start + effective_resample_chunk_size, num_resamples)
                block_rows = stop - start
                padded_block = np.full(
                    (batch_size, effective_resample_chunk_size, width),
                    dummy_index,
                    dtype=np.int32,
                )
                for local_index, (_, _, resampled) in enumerate(batch_entries):
                    padded_block[local_index, :block_rows, : resampled.shape[1]] = resampled[start:stop]
                kernel = (
                    batched_efficient_score_from_indices
                    if batch_codes is None
                    else batched_efficient_score_from_indices_categorical_batch
                )
                if batch_codes is None:
                    score_block = kernel(
                        jnp.asarray(padded_block),
                        score_residual,
                        observation_weight,
                        weighted_nuisance,
                        information_inverse,
                        nuisance_score,
                    )
                else:
                    score_block = kernel(
                        jnp.asarray(padded_block),
                        score_residual,
                        observation_weight,
                        batch_codes,
                        information_inverse,
                        nuisance_score,
                        num_batches=num_batches,
                    )
                reductions = resample_score_reductions(score_block, observed_device, block_rows)
                finite_resamples += np.asarray(reductions[0][:kept], dtype=np.int64)
                exceedances += np.asarray(reductions[1][:kept], dtype=np.int64)
                if power_sums is not None:
                    for index in range(4):
                        power_sums[index] += np.asarray(reductions[2 + index][:kept], dtype=np.float64)
                if stored_resampled is not None:
                    # The only path that still needs the block itself on the host.
                    scores = np.asarray(score_block, dtype=np.float32)[:kept, :block_rows]
                    for local_index, (target_index, _, _) in enumerate(batch_entries):
                        stored_resampled[target_index, :, start:stop] = scores[local_index].T

            if power_sums is not None:
                for local_index, (target_index, _, _) in enumerate(batch_entries):
                    moment_count[target_index] += finite_resamples[local_index]
                    moment_sum[target_index] += power_sums[0, local_index]
                    moment_square[target_index] += power_sums[1, local_index]
                    moment_cube[target_index] += power_sums[2, local_index]
                    moment_fourth[target_index] += power_sums[3, local_index]

            usable = np.isfinite(observed[:kept]) & (finite_resamples > 0)
            for local_index, (target_index, _, _) in enumerate(batch_entries):
                observed_scores[target_index, usable[local_index]] = observed[local_index, usable[local_index]]
                p_values[target_index, usable[local_index]] = (1.0 + exceedances[local_index, usable[local_index]]) / (
                    1.0 + finite_resamples[local_index, usable[local_index]]
                )
                converged[target_index] = null_converged

    if analytic_null_moments and accumulate_score_moments:
        if batch_codes is None or num_batches is None:
            # Intercept-only dense design: the nuisance correction is the global
            # scalar u / I and the draw is one unstratified SRSWOR, so the null
            # is the single-stratum specialization. The dense kernel's 1e-8
            # curvature jitter is omitted; its relative effect is O(1e-8 / I).
            if nuisance_design.shape[1] != 1:
                raise ValueError(
                    "analytic_null_moments without a categorical batch requires an "
                    f"intercept-only nuisance design; saw {nuisance_design.shape[1]} columns."
                )
            raw = intercept_only_null_raw_moments(
                score_residual=np.asarray(score_residual[:dummy_index], dtype=np.float64),
                observation_weight=np.asarray(observation_weight[:dummy_index], dtype=np.float64),
                control_mask=control_mask,
                target_cells=analytic_target_cells,
                num_targets=num_targets,
                jmax=analytic_null_jmax,
            )
        else:
            raw = categorical_batch_null_raw_moments(
                score_residual=np.asarray(score_residual[:dummy_index], dtype=np.float64),
                observation_weight=np.asarray(observation_weight[:dummy_index], dtype=np.float64),
                batch_codes=np.asarray(batch_codes[:dummy_index], dtype=np.int64),
                control_mask=control_mask,
                target_cells=analytic_target_cells,
                num_targets=num_targets,
                num_batches=int(num_batches),
                jmax=analytic_null_jmax,
            )
        # The accumulators are consumed as sums over a reference set, so scale
        # the closed-form raw moments by a nominal count. Nothing was drawn;
        # the count only sets the divisor and the downstream validity floor.
        finite = np.isfinite(raw).all(axis=2) & np.isfinite(observed_scores)
        moment_count[finite] = num_resamples
        for index, accumulator in enumerate((moment_sum, moment_square, moment_cube, moment_fourth)):
            accumulator[finite] = raw[:, :, index][finite] * num_resamples

    moments = (
        (moment_count, moment_sum, moment_square, moment_cube, moment_fourth) if accumulate_score_moments else None
    )
    return observed_scores, p_values, converged, iterations, stored_resampled, moments


def _fit_parametric_tail(
    tail_approximation: str | None,
    *,
    observed_scores: np.ndarray,
    moment_count: np.ndarray,
    moment_sum: np.ndarray,
    moment_square: np.ndarray,
    moment_cube: np.ndarray,
    moment_fourth: np.ndarray,
):
    if tail_approximation is None:
        return None
    common = dict(
        observed_score=observed_scores,
        count=moment_count,
        sum_score=moment_sum,
        sum_square=moment_square,
        sum_cube=moment_cube,
    )
    if tail_approximation == "skew_normal_moments":
        return fit_skew_normal_from_moments(**common)
    if tail_approximation == "student_t_moments":
        return fit_student_t_from_moments(**common, sum_fourth=moment_fourth)
    raise ValueError(f"Unknown tail approximation {tail_approximation!r}.")



def _parametric_result_fields(parametric) -> dict[str, np.ndarray | None]:
    """Parametric tail summaries as host float64, for the result container.

    These are computed in float64 on the host by :mod:`parametric_null` and are
    kept that way. Routing them through ``jnp.asarray`` used to narrow them to
    float32 under JAX's default precision, which flushed every p-value below
    ~1e-45 to zero - and the Cornish-Fisher tail exists precisely to resolve the
    far tail, where the strongest on-target hits sit at ``z`` well past 13. The
    downstream ``-log10(clip(p, 1e-300))`` then tied all of them at 300.

    The high-MOI path in :mod:`perturbo._internal.high_moi.api` already
    returns these as NumPy; this keeps the low-MOI path consistent with it, and
    keeps the fix independent of whether JAX x64 is enabled.

    ``parametric_log_p_value`` carries the same tail on the log scale. float64
    only moves the floor described above from ~1e-45 to ~1e-308; it does not
    remove it, and the far tail is exactly where a parametric null earns its
    keep. The log field is the one with no floor at all, so consumers that want
    a ranking rather than a threshold should read it.
    """
    if parametric is None:
        return dict.fromkeys(
            (
                "parametric_p_value",
                "parametric_log_p_value",
                "parametric_q_value",
                "null_mean",
                "null_variance",
                "null_skewness",
                "null_excess_kurtosis",
                "student_t_degrees_of_freedom",
                "parametric_fit_valid",
                "parametric_used_fallback",
            )
        )
    optional = lambda name: (  # noqa: E731
        np.asarray(getattr(parametric, name), dtype=np.float64) if hasattr(parametric, name) else None
    )
    p_value = np.asarray(parametric.p_value, dtype=np.float64)
    return {
        "parametric_p_value": p_value,
        "parametric_log_p_value": np.asarray(parametric.log_p_value, dtype=np.float64),
        # BH stays on the linear scale: it compares p against i/m * alpha, and
        # the thresholds it is used with live nowhere near the floor.
        "parametric_q_value": np.asarray(_benjamini_hochberg(p_value), dtype=np.float64),
        "null_mean": np.asarray(parametric.null_mean, dtype=np.float64),
        "null_variance": np.asarray(parametric.null_variance, dtype=np.float64),
        "null_skewness": np.asarray(parametric.null_skewness, dtype=np.float64),
        "null_excess_kurtosis": optional("null_excess_kurtosis"),
        "student_t_degrees_of_freedom": optional("degrees_of_freedom"),
        "parametric_fit_valid": np.asarray(parametric.valid, dtype=bool),
        "parametric_used_fallback": (
            np.asarray(parametric.used_fallback, dtype=bool) if hasattr(parametric, "used_fallback") else None
        ),
    }


def validate_analytic_null_strata(categorical_batch_codes, full_strata) -> None:
    """Reject designs the closed-form null does not actually describe.

    The closed form treats every batch as an independent draw of a fixed size.
    That is the real resampling law only when the CRT strata and the batch
    levels are the same partition. With batch in the nuisance design but
    stratification switched off, the per-batch counts are random rather than
    fixed, and the closed form would silently answer a different question.
    """

    if categorical_batch_codes is None:
        # Intercept-only path: the nuisance correction is global, so the closed
        # form describes one unstratified SRSWOR draw. Stratified resampling
        # would make the denominator couple strata through the global weight
        # sum, which the closed form does not model - refuse it.
        n_strata = int(np.unique(np.asarray(full_strata).reshape(-1)).size)
        if n_strata > 1:
            raise ValueError(
                "analytic_null_moments without a categorical batch covariate requires an "
                f"unstratified CRT (saw {n_strata} strata)."
            )
        return
    # Encode both sides first: CRT strata arrive as raw batch labels, which may
    # be strings, so they cannot be combined numerically as they stand.
    batch_index = np.unique(np.asarray(categorical_batch_codes).reshape(-1), return_inverse=True)[1]
    strata_index = np.unique(np.asarray(full_strata).reshape(-1), return_inverse=True)[1]
    if batch_index.shape != strata_index.shape:
        raise ValueError("categorical batch codes and CRT strata must have one entry per cell.")
    n_batch = int(batch_index.max()) + 1
    n_strata = int(strata_index.max()) + 1
    n_joint = int(np.unique(strata_index.astype(np.int64) * n_batch + batch_index).size)
    if not (n_batch == n_strata == n_joint):
        raise ValueError(
            "analytic_null_moments requires the CRT strata to be exactly the batch levels "
            f"(saw {n_batch} batches, {n_strata} strata, {n_joint} joint levels); "
            "stratify the CRT by batch, or use the resampled null."
        )


def run_low_moi_score_permutations(
    design: JointNBDesign,
    *,
    num_resamples: int = 999,
    strata: np.ndarray | jnp.ndarray | None = None,
    seed: int = 0,
    nuisance_prior_scale: float | None = None,
    cell_chunk_size: int | None = None,
    maxiter: int = 500,
    gradient_tolerance: float = 1e-6,
    curvature_jitter: float = 1e-8,
    return_resampled_scores: bool = False,
    backend: str = "jax",
    resample_chunk_size: int = 256,
    jax_num_cell_buckets: int = 5,
    jax_targets_per_batch: int = 4,
    jax_max_target_resample_batch: int = _JAX_MAX_TARGET_RESAMPLE_BATCH,
    jax_max_gather_gib: float | None = None,
    null_model: str = "control_only",
    permutations: TargetPermutations | None = None,
    tail_approximation: str | Sequence[str] | None = None,
    saddlepoint_screen_p_value: float = 0.01,
    saddlepoint_two_sided: str | None = None,
    saddlepoint_gene_block_size: int = 64,
    saddlepoint_only: bool = False,
    analytic_null_moments: bool = False,
    analytic_null_jmax: int = ANALYTIC_DEFAULT_JMAX,
    nuisance_coefficients: np.ndarray | jnp.ndarray | None = None,
    return_null_moments: bool = False,
) -> ScorePermutationResult:
    """Run target-vs-NTC NB score permutations for every gene.

    ``backend='batched'`` fits all genes at once by Fisher scoring and evaluates
    resamples as sparse matrix products. ``'per_pair'`` is the slower reference
    path that fits each pair with L-BFGS; the two are checked against each other
    in the tests.

    ``backend='jax'`` runs the same control-only null fit and score statistics
    through the float32 kernels in :mod:`perturbo._internal.jax_kernels`
    instead of the NumPy/float64 versions ``'batched'`` uses. It is the
    device-portable path: everything it does is an elementwise pass, a gather, or
    a segment reduction, so it is the backend to use on GPU or at cell counts
    where NumPy's single-threaded per-target loop dominates. It currently only
    supports ``null_model='control_only'`` - the pooled null is not yet ported.
    Its selected-cell gathers use at most ``jax_num_cell_buckets`` padded widths
    (five by default), rather than compiling once for each target size. Set it
    to one to use one global maximum-width kernel. ``jax_targets_per_batch``
    controls how many same-width targets share a kernel call; it defaults to a
    conservative four and automatically reduces the resample block to hold the
    target-by-resample working set below ``jax_max_target_resample_batch``. It
    is **ignored when ``analytic_null_moments`` is set**, where the target batch
    is instead sized per width bucket from ``jax_max_gather_gib``: no resamples
    are drawn there, so the resample axis is absent from the working set and a
    batch of four would only multiply kernel launches.
    ``jax_max_gather_gib`` additionally caps the gather working set using the
    padded target-cell width, the gene count, and the ``2 + nuisance`` float32
    tensors the score kernel materializes concurrently, which prevents rare
    very large perturbations from exhausting device memory. It bounds that
    step's total peak, not one tensor.

    ``permutations`` accepts the output of
    :func:`precompute_low_moi_permutations`, so a caller sweeping gene chunks
    draws each target's resamples once instead of once per chunk. Permutations
    never depend on genes, so this changes nothing but the runtime.

    ``null_model='control_only'`` fits the nuisance model once on control cells
    and reuses it for every target, which is both far cheaper and a cleaner cut -
    see :func:`fit_control_only_nb_null`. ``'pooled'`` refits per target on
    controls plus that target's cells; it is only available on the batched
    backend, and ``'per_pair'`` always implies pooled.

    ``saddlepoint_only`` requires ``tail_approximation='saddlepoint'``. It
    computes the observed efficient score and SPA tail but neither draws nor
    scores permutations; the empirical p- and q-value arrays are therefore
    NaN, and the result reports zero resamples.

    ``nuisance_coefficients`` supplies the ``(nuisance, genes)`` null
    coefficients directly and skips the control-only fit entirely. It exists so
    a caller that has *already* estimated this exact quantity - notably a
    two-stage fit whose stage one is a control-cell baseline - can avoid fitting
    the same model twice. This does not weaken the test: the resampling null is
    conditional on the residuals and weights, so it stays exact whatever
    produced them, and the statistic's ``x'WZ (Z'WZ)^-1 Z'r`` correction absorbs
    a non-MLE estimate to first order. What it does assume is that the supplied
    coefficients are *near* the control-cell null mode, since that correction is
    only a first-order one; the reported ``null_converged`` measures exactly
    that and should be checked rather than ignored.
    """

    if num_resamples < 1:
        raise ValueError("num_resamples must be positive.")
    if nuisance_prior_scale is not None and nuisance_prior_scale <= 0:
        raise ValueError("nuisance_prior_scale must be positive or None.")
    if cell_chunk_size is not None and cell_chunk_size < 1:
        raise ValueError("cell_chunk_size must be positive when provided.")
    if maxiter < 1:
        raise ValueError("maxiter must be positive.")
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}; got {backend!r}.")
    if resample_chunk_size < 1:
        raise ValueError("resample_chunk_size must be positive.")
    if jax_num_cell_buckets < 1:
        raise ValueError("jax_num_cell_buckets must be positive.")
    if jax_targets_per_batch < 1:
        raise ValueError("jax_targets_per_batch must be positive.")
    if jax_max_target_resample_batch < 1:
        raise ValueError("jax_max_target_resample_batch must be positive.")
    if jax_max_gather_gib is not None and (not np.isfinite(jax_max_gather_gib) or jax_max_gather_gib <= 0):
        raise ValueError("jax_max_gather_gib must be finite and positive or None.")
    if permutations is not None:
        if permutations.num_resamples != num_resamples:
            raise ValueError(
                f"permutations hold {permutations.num_resamples} resamples but {num_resamples} were requested."
            )
        if len(permutations.indices) != design.num_targets:
            raise ValueError("permutations must contain one entry per design target.")
    if null_model not in _NULL_MODELS:
        raise ValueError(f"null_model must be one of {sorted(_NULL_MODELS)}; got {null_model!r}.")
    known_tails = ("skew_normal_moments", "student_t_moments", "saddlepoint")
    if tail_approximation is None:
        requested_tails: tuple[str, ...] = ()
    elif isinstance(tail_approximation, str):
        requested_tails = (tail_approximation,)
    else:
        # Several families on one fit and one set of draws, so a difference
        # between their columns is the approximation rather than the run.
        requested_tails = tuple(dict.fromkeys(tail_approximation))
    for _name in requested_tails:
        if _name not in known_tails:
            raise ValueError(
                f"tail_approximation must be None or drawn from {known_tails}; got {_name!r}."
            )
    primary_tail = requested_tails[0] if requested_tails else None
    low_moi_mechanism = (
        "permutation" if permutations is None else permutations.resampling_mechanism
    )
    if backend == "jax" and null_model == "pooled":
        raise ValueError("backend='jax' only supports null_model='control_only'; use backend='batched' for pooled.")
    if "saddlepoint" in requested_tails and (backend != "jax" or null_model != "control_only"):
        raise ValueError("saddlepoint currently requires backend='jax' with null_model='control_only'.")
    if saddlepoint_only:
        if requested_tails != ("saddlepoint",):
            raise ValueError("saddlepoint_only requires tail_approximation='saddlepoint' alone.")
        if permutations is not None and any(
            entry is not None for entry in permutations.indices
        ):
            raise ValueError(
                "saddlepoint_only does not accept drawn resamples. Under the propensity "
                "mechanism pass permutations built with draw_resamples=False: the "
                "saddlepoint needs the fitted selection model but never reads a draw, "
                "which is the property that makes it sampling-free."
            )
        if return_resampled_scores or return_null_moments:
            raise ValueError("saddlepoint_only cannot return resampled scores or null moments.")
    if not 0.0 < saddlepoint_screen_p_value <= 1.0:
        raise ValueError("saddlepoint_screen_p_value must lie in (0, 1].")
    if saddlepoint_gene_block_size < 1:
        raise ValueError("saddlepoint_gene_block_size must be positive.")
    if nuisance_coefficients is not None:
        if backend != "jax" or null_model != "control_only":
            raise ValueError(
                "nuisance_coefficients requires backend='jax' with null_model='control_only'; "
                "the pooled null refits per target by definition, so there is nothing to supply."
            )
        nuisance_coefficients = np.asarray(nuisance_coefficients, dtype=np.float64)
        expected = (np.asarray(design.nuisance_design).shape[1], design.num_genes)
        if nuisance_coefficients.shape != expected:
            raise ValueError(
                f"nuisance_coefficients must have shape {expected} (nuisance, genes); "
                f"got {nuisance_coefficients.shape}."
            )
        if not np.all(np.isfinite(nuisance_coefficients)):
            raise ValueError("nuisance_coefficients must be finite.")
    if analytic_null_moments:
        if backend != "jax":
            raise ValueError("analytic_null_moments requires backend='jax'.")
        if tail_approximation is None:
            raise ValueError(
                "analytic_null_moments replaces the resampled null with closed-form moments, "
                "so it produces no empirical p-value and requires a tail_approximation."
            )
        if tail_approximation == "saddlepoint":
            raise ValueError("saddlepoint and analytic_null_moments are separate analytic null paths.")
    resolved_null_model = "pooled" if backend == "per_pair" else null_model

    counts = np.asarray(design.counts)
    target_codes = getattr(design, "target_codes", None)
    categorical_batch_codes = getattr(design, "batch_codes", None)
    categorical_design = target_codes is not None
    if categorical_design:
        target_codes = np.asarray(target_codes, dtype=np.int32).reshape(-1)
        if target_codes.shape != (design.num_cells,):
            raise ValueError("target_codes must contain one code per design cell.")
        if np.any((target_codes < -1) | (target_codes >= design.num_targets)):
            raise ValueError("target_codes contain an unknown target.")
        if backend != "jax":
            raise ValueError("categorical low-MOI designs support only backend='jax'.")
        target_design = None
    else:
        target_design = np.asarray(design.target_design, dtype=np.int8)
        target_codes = np.full(design.num_cells, -1, dtype=np.int32)
        target_rows = target_design.sum(axis=1) > 0
        target_codes[target_rows] = np.argmax(target_design[target_rows], axis=1)
    nuisance_design = np.asarray(design.nuisance_design, dtype=np.float32)
    offsets = np.asarray(design.offsets, dtype=np.float32)
    dispersion = np.asarray(design.dispersion, dtype=np.float32)
    control_mask = np.asarray(design.control_mask, dtype=bool)
    if categorical_batch_codes is not None:
        categorical_batch_codes = np.asarray(categorical_batch_codes, dtype=np.int32).reshape(-1)
        if categorical_batch_codes.shape != (design.num_cells,) or design.batch_names is None:
            raise ValueError("categorical batch codes must align with the design and have names.")
        if nuisance_prior_scale is not None:
            raise ValueError("categorical batch score fitting does not support nuisance_prior_scale.")
        if nuisance_coefficients is not None:
            raise ValueError(
                "nuisance_coefficients cannot be combined with a categorical batch design: that "
                "path parametrizes the nuisance as one intercept per batch, not as coefficients "
                "on a dense nuisance design. Dummy-code the batch into the covariate matrix to "
                "supply coefficients for it."
            )
    if target_design is not None and np.any(target_design.sum(axis=1) > 1):
        raise ValueError("run_low_moi_score_permutations requires a low-MOI target design.")
    if strata is None:
        full_strata = np.zeros(design.num_cells, dtype=np.int64)
    else:
        full_strata = np.asarray(strata).reshape(-1)
        if full_strata.shape != (design.num_cells,):
            raise ValueError("strata must contain one value per design cell.")

    if analytic_null_moments:
        validate_analytic_null_strata(categorical_batch_codes, full_strata)
        if categorical_batch_codes is None and nuisance_prior_scale is not None:
            raise ValueError(
                "analytic_null_moments does not support a nuisance prior: the closed form "
                "takes the correction as u / I, without the prior's ridge."
            )
    if tail_approximation == "saddlepoint":
        validate_analytic_null_strata(categorical_batch_codes, full_strata)
        if categorical_batch_codes is None and nuisance_prior_scale is not None:
            raise ValueError(
                "saddlepoint does not support a nuisance prior in the intercept-only path: "
                "the efficient contribution uses the un-ridged correction u / I."
            )

    observed_scores = np.full((design.num_targets, design.num_genes), np.nan, dtype=np.float32)
    p_values = np.full_like(observed_scores, np.nan)
    null_converged = np.zeros((design.num_targets, design.num_genes), dtype=bool)
    null_iterations = np.zeros((design.num_targets, design.num_genes), dtype=np.int32)
    stored_resampled = (
        np.full(
            (design.num_targets, design.num_genes, num_resamples),
            np.nan,
            dtype=np.float32,
        )
        if return_resampled_scores
        else None
    )
    # The moment families read these accumulators; the saddlepoint does not.
    accumulate_score_moments = (
        any(name != "saddlepoint" for name in requested_tails) or return_null_moments
    )
    moment_count = np.zeros((design.num_targets, design.num_genes), dtype=np.int64)
    moment_sum = np.zeros((design.num_targets, design.num_genes), dtype=np.float64)
    moment_square = np.zeros((design.num_targets, design.num_genes), dtype=np.float64)
    moment_cube = np.zeros((design.num_targets, design.num_genes), dtype=np.float64)
    moment_fourth = np.zeros((design.num_targets, design.num_genes), dtype=np.float64)

    num_nuisance = nuisance_design.shape[1]
    prior_precision = 0.0 if nuisance_prior_scale is None else 1.0 / float(nuisance_prior_scale) ** 2
    ridge = (prior_precision + float(curvature_jitter)) * np.eye(num_nuisance)
    control_residual = control_weight = control_weighted_nuisance = None
    control_information = None
    control_fit = None
    if resolved_null_model == "control_only" and backend != "jax":
        control_fit, control_residual, control_weight = fit_control_only_nb_null(
            counts,
            control_mask=control_mask,
            nuisance_design=nuisance_design,
            offsets=offsets,
            theta=dispersion,
            nuisance_prior_scale=nuisance_prior_scale,
            curvature_jitter=curvature_jitter,
            max_iterations=maxiter,
            gradient_tolerance=gradient_tolerance,
        )
        nuisance_float = nuisance_design.astype(np.float64)
        control_weighted_nuisance = np.ascontiguousarray(
            (control_weight[:, :, None] * nuisance_float[:, None, :]).reshape(
                counts.shape[0], design.num_genes * num_nuisance
            )
        )
        # The control block of the nuisance information is shared by every
        # target; only that target's own cells need adding per iteration.
        control_information = _nuisance_information(
            nuisance_float[control_mask], control_weight[control_mask], ridge=ridge
        )
        control_nuisance_score = nuisance_float[control_mask].T @ control_residual[control_mask]
    elif resolved_null_model == "control_only" and backend == "jax":
        counts_jax = jnp.asarray(counts, dtype=jnp.float32)
        offsets_jax = jnp.asarray(offsets, dtype=jnp.float32)
        theta_jax = jnp.asarray(dispersion, dtype=jnp.float32)
        control_indices_np = np.flatnonzero(control_mask)
        control_indices = jnp.asarray(control_indices_np)
        if categorical_batch_codes is None:
            nuisance_jax = jnp.asarray(nuisance_design, dtype=jnp.float32)
            if nuisance_coefficients is None:
                control_beta, _, _ = fisher_nb_null(
                    counts_jax[control_indices],
                    nuisance_jax[control_indices],
                    offsets_jax[control_indices],
                    theta_jax,
                    jnp.asarray(prior_precision, dtype=jnp.float32),
                    jnp.asarray(curvature_jitter, dtype=jnp.float32),
                    jnp.asarray(1e-6, dtype=jnp.float32),
                    max_iterations=maxiter,
                )
            else:
                # Supplied from outside, so nothing is fit here. Everything
                # downstream is derived from the coefficients rather than passed
                # in alongside them, which keeps the residual, weight,
                # information, and nuisance score mutually consistent by
                # construction instead of by the caller's discipline.
                control_beta = jnp.asarray(nuisance_coefficients, dtype=jnp.float32)
            control_residual_jax, control_weight_jax = nb_null_residual_and_weight(
                counts_jax, nuisance_jax, offsets_jax, theta_jax, control_beta
            )
            # Pure saddlepoint testing needs only observed target rows. Keep
            # the full weighted design only when assignments will be resampled.
            control_weighted_nuisance_jax = (
                jnp.empty((counts.shape[0], 0), dtype=jnp.float32)
                if saddlepoint_only
                else (control_weight_jax[:, :, None] * nuisance_jax[:, None, :]).reshape(
                    counts.shape[0], design.num_genes * num_nuisance
                )
            )
            control_information_jax = jnp.einsum(
                "nq,ng,nr->gqr",
                nuisance_jax[control_indices],
                control_weight_jax[control_indices],
                nuisance_jax[control_indices],
            ) + jnp.asarray(ridge, dtype=jnp.float32)
            control_nuisance_score_jax = nuisance_jax[control_indices].T @ control_residual_jax[control_indices]
            control_fit_converged = _newton_step_converged(
                counts[control_mask].astype(np.float64),
                nuisance_design[control_mask].astype(np.float64),
                offsets[control_mask].astype(np.float64),
                np.asarray(dispersion, dtype=np.float64),
                np.asarray(control_beta, dtype=np.float64),
                prior_precision=prior_precision,
                ridge=ridge,
            )
            batch_codes_jax = None
            num_batches = None
        else:
            num_batches = len(design.batch_names)
            batch_codes_jax = jnp.asarray(categorical_batch_codes, dtype=jnp.int32)
            control_beta, _, control_step, _ = fit_categorical_batch_nb_null_laplace(
                counts_jax[control_indices],
                batch_codes_jax[control_indices],
                offsets_jax[control_indices],
                theta_jax,
                jnp.asarray(curvature_jitter, dtype=jnp.float32),
                jnp.asarray(1e-6, dtype=jnp.float32),
                num_batches=num_batches,
                max_iterations=maxiter,
            )
            control_residual_jax, control_weight_jax = categorical_batch_nb_null_residual_and_weight(
                counts_jax, batch_codes_jax, offsets_jax, theta_jax, control_beta
            )
            control_information_jax = (
                jnp.zeros((num_batches, design.num_genes), dtype=jnp.float32)
                .at[batch_codes_jax[control_indices]]
                .add(control_weight_jax[control_indices])
            )
            control_nuisance_score_jax = (
                jnp.zeros((num_batches, design.num_genes), dtype=jnp.float32)
                .at[batch_codes_jax[control_indices]]
                .add(control_residual_jax[control_indices])
            )
            control_fit_converged = np.asarray(control_step <= 1e-6)
            # Unused by the categorical score branch, retained to keep the
            # common executor signature compact.
            control_weighted_nuisance_jax = jnp.empty((counts.shape[0], 0), dtype=jnp.float32)
            nuisance_jax = jnp.empty((counts.shape[0] + 1, 0), dtype=jnp.float32)

        # Every gathered cell array needs the same sentinel row. The target and
        # resample index blocks themselves are padded with NumPy in the batched
        # executor below, before they enter JAX.
        jax_dummy_index = counts.shape[0]
        control_residual_jax, control_weight_jax, control_weighted_nuisance_jax = append_zero_weight_row(
            control_residual_jax, control_weight_jax, control_weighted_nuisance_jax
        )
        if batch_codes_jax is not None:
            batch_codes_jax = jnp.concatenate([batch_codes_jax, jnp.zeros((1,), dtype=jnp.int32)])
        else:
            nuisance_jax = jnp.concatenate([nuisance_jax, jnp.zeros_like(nuisance_jax[:1])], axis=0)
        (
            observed_scores,
            p_values,
            null_converged,
            null_iterations,
            stored_resampled,
            moments,
        ) = _run_jax_control_only_score_permutations(
            design=design,
            target_codes=target_codes,
            control_mask=control_mask,
            full_strata=full_strata,
            num_resamples=num_resamples,
            seed=seed,
            permutations=permutations,
            resample_chunk_size=resample_chunk_size,
            num_cell_buckets=jax_num_cell_buckets,
            targets_per_batch=jax_targets_per_batch,
            max_target_resample_batch=jax_max_target_resample_batch,
            max_gather_bytes=(None if jax_max_gather_gib is None else int(float(jax_max_gather_gib) * 1024**3)),
            return_resampled_scores=return_resampled_scores,
            accumulate_score_moments=accumulate_score_moments,
            score_residual=control_residual_jax,
            observation_weight=control_weight_jax,
            weighted_nuisance=control_weighted_nuisance_jax,
            nuisance_design=nuisance_jax,
            control_information=control_information_jax,
            control_nuisance_score=control_nuisance_score_jax,
            batch_codes=batch_codes_jax,
            num_batches=num_batches,
            null_converged=control_fit_converged,
            dummy_index=jax_dummy_index,
            analytic_null_moments=analytic_null_moments,
            observed_only=saddlepoint_only,
            analytic_null_jmax=analytic_null_jmax,
        )
        if moments is not None:
            moment_count, moment_sum, moment_square, moment_cube, moment_fourth = moments
        q_values = _benjamini_hochberg(p_values)
        saddlepoint = None

        def _low_moi_contribution():
            """Efficient contributions under the control-only null fit.

            Two nuisance representations reach here. A general design carries a
            dense (genes, q, q) information; a categorical batch carries a
            (batches, genes) diagonal, where the projection collapses to
            subtracting each batch's weighted mean residual. Both give the same
            (cells, genes) array.
            """

            residual = control_residual_jax[:jax_dummy_index]
            weight = control_weight_jax[:jax_dummy_index]
            if batch_codes_jax is None:
                inverse = jnp.linalg.inv(control_information_jax)
                direction = jnp.einsum("gqr,rg->gq", inverse, control_nuisance_score_jax)
                design_rows = nuisance_jax[:jax_dummy_index]
                return residual - weight * (design_rows @ direction.T)
            codes = batch_codes_jax[:jax_dummy_index]
            safe = jnp.where(control_information_jax > 0.0, control_information_jax, jnp.nan)
            per_batch = control_nuisance_score_jax / safe
            return residual - weight * per_batch[codes]

        def _fit_family(name):
            nonlocal saddlepoint
            if name == "saddlepoint" and low_moi_mechanism == "propensity":
                if permutations is None or permutations.pair_rows is None:
                    raise ValueError(
                        "The propensity saddlepoint needs the fitted selection model. Pass "
                        "permutations from precompute_low_moi_permutations with "
                        "resampling_mechanism='propensity', which keeps each target's pool "
                        "rows and logits."
                    )
                compact_propensity = (
                    permutations.propensity_coefficients is not None
                    and permutations.propensity_basis is not None
                )
                legacy_propensity = (
                    permutations.shared_logits is not None
                    and permutations.pool_intercepts is not None
                )
                if not compact_propensity and not legacy_propensity:
                    raise ValueError(
                        "permutations do not carry a compact selection model; recompute them "
                        "with precompute_low_moi_permutations."
                    )
                # The pool projection: the contributions are efficient under the
                # control-only fit, and a target's own cells are out of sample
                # for it. Passing the weights and the nuisance representation lets
                # the kernel re-project each target's pool as if the nuisance had
                # been fit on the pool, which is what keeps the test calibrated
                # when targets are not small next to the control pool.
                saddlepoint = fit_low_moi_propensity_saddlepoint(
                    contribution=_low_moi_contribution(),
                    target_codes=target_codes,
                    control_mask=control_mask,
                    shared_logits=(permutations.shared_logits if legacy_propensity else None),
                    intercepts=(permutations.pool_intercepts if legacy_propensity else None),
                    propensity_coefficients=(
                        permutations.propensity_coefficients if compact_propensity else None
                    ),
                    propensity_basis=(permutations.propensity_basis if compact_propensity else None),
                    num_targets=design.num_targets,
                    screen_p_value=saddlepoint_screen_p_value,
                    two_sided=saddlepoint_two_sided or "equal-tail",
                    gene_block_size=saddlepoint_gene_block_size,
                    weight=control_weight_jax[:jax_dummy_index],
                    nuisance_design=None if batch_codes_jax is not None else nuisance_jax[:jax_dummy_index],
                    batch_codes=None if batch_codes_jax is None else batch_codes_jax[:jax_dummy_index],
                    control_information=control_information_jax,
                )
                return saddlepoint
            if name != "saddlepoint":
                return _fit_parametric_tail(
                    name,
                    observed_scores=observed_scores,
                    moment_count=moment_count,
                    moment_sum=moment_sum,
                    moment_square=moment_square,
                    moment_cube=moment_cube,
                    moment_fourth=moment_fourth,
                )
            if saddlepoint_two_sided not in (None, "symmetric"):
                raise ValueError(
                    "The stratified permutation saddlepoint implements only the symmetric "
                    "two-sided event P(|S| >= |observed|); it is a diagnostic. Leave "
                    f"saddlepoint_two_sided unset for it, or use the propensity mechanism for "
                    f"the {saddlepoint_two_sided!r} convention."
                )
            target_cells = {
                target: np.flatnonzero(target_codes == target)
                for target in range(design.num_targets)
                if np.any(target_codes == target)
            }
            saddlepoint = fit_stratified_saddlepoint_from_components(
                score_residual=np.asarray(control_residual_jax[:jax_dummy_index], dtype=np.float64),
                observation_weight=np.asarray(control_weight_jax[:jax_dummy_index], dtype=np.float64),
                strata=(
                    np.zeros(jax_dummy_index, dtype=np.int64)
                    if categorical_batch_codes is None
                    else categorical_batch_codes
                ),
                control_mask=control_mask,
                target_cells=target_cells,
                num_targets=design.num_targets,
                observed_score=observed_scores,
                screen_p_value=saddlepoint_screen_p_value,
                gene_block_size=saddlepoint_gene_block_size,
            )
            return saddlepoint

        tail_fits: dict[str, dict[str, np.ndarray]] = {}
        parametric = None
        for name in requested_tails:
            fit = _fit_family(name)
            if fit is None:
                continue
            linear = np.asarray(fit.p_value)
            tail_fits[name] = {
                "p_value": linear,
                "log_p_value": np.asarray(fit.log_p_value, dtype=np.float64),
                "valid": np.asarray(fit.valid, dtype=bool),
            }
            if name == primary_tail:
                parametric = fit
        return ScorePermutationResult(
            observed_score=jnp.asarray(observed_scores),
            p_value=jnp.asarray(p_values),
            q_value=jnp.asarray(q_values),
            null_converged=jnp.asarray(null_converged),
            null_optimizer_iterations=jnp.asarray(null_iterations),
            target_names=design.target_names,
            gene_names=design.gene_names,
            num_resamples=(0 if saddlepoint_only else num_resamples),
            method="pairwise_stratified_nb_score_permutation",
            resampled_scores=None if stored_resampled is None else jnp.asarray(stored_resampled),
            backend=backend,
            null_model=resolved_null_model,
            tail_approximation=primary_tail,
            tail_fits=tail_fits or None,
            saddlepoint_observed_sum=(
                None if saddlepoint is None else np.asarray(saddlepoint.observed_sum, dtype=np.float64)
            ),
            saddlepoint_max_sampling_fraction=(
                None
                if saddlepoint is None
                else np.asarray(saddlepoint.max_sampling_fraction, dtype=np.float64)
            ),
            null_moments=(
                ResampledNullMoments(
                    count=moment_count,
                    sum_score=moment_sum,
                    sum_square=moment_square,
                    sum_cube=moment_cube,
                    sum_fourth=moment_fourth,
                )
                if return_null_moments
                else None
            ),
            **_parametric_result_fields(parametric),
        )

    for target_index in range(design.num_targets):
        observed_full = target_design[:, target_index]
        pair_mask = control_mask | (observed_full > 0)
        observed_assignment = observed_full[pair_mask]
        if observed_assignment.sum() == 0 or observed_assignment.sum() == observed_assignment.size:
            continue
        # Drawn as indices so both backends see identical resamples; the per_pair
        # reference scatters them back to binary vectors.
        if permutations is None:
            resampled_indices = make_stratified_permutation_indices(
                observed_assignment,
                num_resamples=num_resamples,
                strata=full_strata[pair_mask],
                rng=target_permutation_rng(seed, design.target_names[target_index]),
            )
        else:
            resampled_indices = permutations.indices[target_index]
            if resampled_indices is None:
                continue
        pair_nuisance = nuisance_design[pair_mask]
        pair_offsets = offsets[pair_mask]

        if backend == "per_pair":
            assignments = np.vstack(
                [
                    observed_assignment[None, :],
                    indices_to_binary_assignments(resampled_indices, num_cells=observed_assignment.size),
                ]
            )
            for gene_index in range(design.num_genes):
                offset_column = offsets[:, 0] if offsets.shape[1] == 1 else offsets[:, gene_index]
                null = _fit_null_score_components(
                    counts=counts[pair_mask, gene_index],
                    nuisance_design=pair_nuisance,
                    offset=offset_column[pair_mask],
                    theta=float(dispersion[gene_index]),
                    nuisance_prior_scale=nuisance_prior_scale,
                    cell_chunk_size=cell_chunk_size,
                    maxiter=maxiter,
                    gradient_tolerance=gradient_tolerance,
                    curvature_jitter=curvature_jitter,
                )
                statistics = efficient_nb_score_statistics(
                    assignments,
                    score_residual=null.score_residual,
                    observation_weight=null.observation_weight,
                    nuisance_design=pair_nuisance,
                    nuisance_information_inverse=null.nuisance_information_inverse,
                )
                observed_score = statistics[0]
                resampled_scores = statistics[1:]
                finite_resampled = resampled_scores[np.isfinite(resampled_scores)]
                if np.isfinite(observed_score) and finite_resampled.size:
                    exceedances = np.count_nonzero(np.abs(finite_resampled) >= abs(observed_score))
                    p_values[target_index, gene_index] = (1.0 + exceedances) / (1.0 + finite_resampled.size)
                    observed_scores[target_index, gene_index] = observed_score
                    if accumulate_score_moments:
                        moment_count[target_index, gene_index] = finite_resampled.size
                        moment_sum[target_index, gene_index] = finite_resampled.sum()
                        moment_square[target_index, gene_index] = np.square(finite_resampled).sum()
                        moment_cube[target_index, gene_index] = np.power(finite_resampled, 3).sum()
                        moment_fourth[target_index, gene_index] = np.power(finite_resampled, 4).sum()
                null_converged[target_index, gene_index] = null.converged
                null_iterations[target_index, gene_index] = null.optimizer_iterations
                if stored_resampled is not None:
                    stored_resampled[target_index, gene_index] = resampled_scores.astype(np.float32)
            continue

        if resolved_null_model == "control_only":
            pair_rows = np.flatnonzero(pair_mask)
            target_rows = np.flatnonzero(observed_full > 0)
            # control block + this target's own cells, instead of a fresh einsum
            # over every pair cell.
            information = control_information + np.einsum(
                "nq,ng,nr->gqr",
                nuisance_design[target_rows].astype(np.float64),
                control_weight[target_rows],
                nuisance_design[target_rows].astype(np.float64),
                optimize=True,
            )
            # Z'r over the pair cells. The control block is shared and is very
            # nearly zero by the control fit's own score equation, but include it
            # so the prior and jitter terms are accounted for exactly.
            nuisance_score = (
                control_nuisance_score
                + nuisance_design[target_rows].astype(np.float64).T @ control_residual[target_rows]
            )
            statistics_kwargs = {
                "score_residual": control_residual,
                "observation_weight": control_weight,
                "weighted_nuisance": control_weighted_nuisance,
                "nuisance_information_inverse": np.linalg.inv(information),
                "nuisance_score": nuisance_score,
            }
            # Indices into the full design, so no per-target row slicing is needed.
            observed_statistic = sparse_nb_score_statistics(target_rows[None, :], **statistics_kwargs)[0]
            pair_converged = control_fit.converged
            pair_iterations = control_fit.iterations
            score_statistics_fn = sparse_nb_score_statistics
        else:
            batched_null = fit_batched_nb_null(
                counts[pair_mask],
                nuisance_design=pair_nuisance,
                offsets=pair_offsets,
                theta=dispersion,
                nuisance_prior_scale=nuisance_prior_scale,
                curvature_jitter=curvature_jitter,
                max_iterations=maxiter,
                gradient_tolerance=gradient_tolerance,
            )
            statistics_kwargs = {
                "score_residual": batched_null.score_residual,
                "observation_weight": batched_null.observation_weight,
                "weighted_nuisance": batched_null.weighted_nuisance,
                "nuisance_information_inverse": batched_null.nuisance_information_inverse,
            }
            pair_rows = np.arange(int(pair_mask.sum()), dtype=np.int64)
            observed_statistic = sparse_nb_score_statistics(
                np.flatnonzero(observed_assignment > 0)[None, :], **statistics_kwargs
            )[0]
            pair_converged = batched_null.converged
            pair_iterations = batched_null.iterations
            score_statistics_fn = sparse_nb_score_statistics
        exceedances = np.zeros(design.num_genes, dtype=np.int64)
        finite_resamples = np.zeros(design.num_genes, dtype=np.int64)
        absolute_observed = np.abs(observed_statistic)
        for start in range(0, num_resamples, resample_chunk_size):
            stop = min(start + resample_chunk_size, num_resamples)
            block = score_statistics_fn(
                pair_rows[resampled_indices[start:stop]],
                **statistics_kwargs,
            )
            finite_block = np.isfinite(block)
            finite_resamples += finite_block.sum(axis=0)
            exceedances += (finite_block & (np.abs(block) >= absolute_observed[None, :])).sum(axis=0)
            if accumulate_score_moments:
                values = np.where(finite_block, block, 0.0)
                moment_count[target_index] += finite_block.sum(axis=0)
                moment_sum[target_index] += values.sum(axis=0)
                moment_square[target_index] += np.square(values).sum(axis=0)
                moment_cube[target_index] += np.power(values, 3).sum(axis=0)
                moment_fourth[target_index] += np.power(values, 4).sum(axis=0)
            if stored_resampled is not None:
                stored_resampled[target_index, :, start:stop] = block.T.astype(np.float32)
        usable = np.isfinite(observed_statistic) & (finite_resamples > 0)
        observed_scores[target_index, usable] = observed_statistic[usable]
        p_values[target_index, usable] = (1.0 + exceedances[usable]) / (1.0 + finite_resamples[usable])
        null_converged[target_index] = pair_converged
        # One shared Fisher-scoring loop serves every gene, so the iteration
        # count is per target rather than per pair.
        null_iterations[target_index] = pair_iterations

    q_values = _benjamini_hochberg(p_values)
    parametric = _fit_parametric_tail(
        tail_approximation,
        observed_scores=observed_scores,
        moment_count=moment_count,
        moment_sum=moment_sum,
        moment_square=moment_square,
        moment_cube=moment_cube,
        moment_fourth=moment_fourth,
    )
    return ScorePermutationResult(
        observed_score=jnp.asarray(observed_scores),
        p_value=jnp.asarray(p_values),
        q_value=jnp.asarray(q_values),
        null_converged=jnp.asarray(null_converged),
        null_optimizer_iterations=jnp.asarray(null_iterations),
        target_names=design.target_names,
        gene_names=design.gene_names,
        num_resamples=num_resamples,
        method="pairwise_stratified_nb_score_permutation",
        resampled_scores=None if stored_resampled is None else jnp.asarray(stored_resampled),
        backend=backend,
        null_model=resolved_null_model,
        tail_approximation=tail_approximation,
        **_parametric_result_fields(parametric),
    )


__all__ = [
    "BatchedNullComponents",
    "NullScoreComponents",
    "ScorePermutationResult",
    "TargetPermutations",
    "batched_nb_score_statistics",
    "efficient_nb_score_statistics",
    "evaluate_nb_null_components",
    "fit_batched_nb_null",
    "fit_control_only_nb_null",
    "sparse_nb_score_statistics",
    "indices_to_binary_assignments",
    "make_stratified_permutation_indices",
    "make_stratified_permutations",
    "precompute_low_moi_permutations",
    "run_low_moi_score_permutations",
]
