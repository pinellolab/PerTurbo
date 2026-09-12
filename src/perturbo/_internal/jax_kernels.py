"""float32 JAX kernels for the low-MOI NB score test and effect fit.

These mirror the NumPy implementations in :mod:`score_resampling` and
:mod:`control_anchored_laplace` and are validated against them. They exist to
make the pipeline device-portable: every kernel here is a fused elementwise pass,
a gather, or a segment reduction, which is the shape that scales to GPU and to
millions of cells.

**float32 is deliberate, not a compromise.** Measured end to end against float64
on a 400-gene, 5.5k-cell problem, the nuisance coefficients agreed to 1.4e-7, the
score statistics to 6.4e-4 relative, and the permutation p-values were *bit
identical* - no gene differed, and no call flipped at either 0.01 or 0.05.

That works because these solvers never evaluate the objective. A float32
log-likelihood summed over thousands of cells resolves to ~1e-4 absolute, which
is what strands L-BFGS in a flat region and forced float64 elsewhere in this
package. Newton and Fisher scoring use only the gradient and curvature, and their
fixed point is where the gradient vanishes, so they converge to the same place in
either precision. The one thing that does not survive is an *absolute* gradient
tolerance, so convergence here is judged on the Newton step in coefficient space,
which is scale free.

**Sparsity.** Two different structures, handled differently on purpose.

The permutation indicator is structurally regular: every resample selects exactly
``K`` cells, so it is a gather, not a sparse matrix. ``jnp.take`` followed by a
sum does precisely the required work with no coordinate bookkeeping, and it is
agnostic to MOI, since each target's assignment vector is binary whatever the
other guides do.

The perturbation design is genuinely sparse and irregular, so it is carried as
``(cell, target)`` index pairs - the coordinate half of a BCOO - and reduced with
``segment_sum``. That is what buys MOI > 1 support: at one guide per cell each
cell index appears once, at a few guides per cell it appears a few times, and the
reduction is identical. Note this yields the *diagonal* of the effect
information, which is exact for low MOI and is the same approximation
``projected_diagonal_marginal_variances`` already makes for high MOI.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

# Matches the NumPy kernels so both see the same linear predictor.
_ETA_CLIP = 30.0
_MAX_STEP = 5.0


def _nb_null_terms(
    counts: jnp.ndarray,
    eta: jnp.ndarray,
    theta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Score residual, expected weight, and observed weight at ``eta``."""

    mean = jnp.exp(jnp.clip(eta, -_ETA_CLIP, _ETA_CLIP))
    denominator = theta + mean
    residual = theta * (counts - mean) / denominator
    expected_weight = theta * mean / denominator
    observed_weight = theta * (counts + theta) * mean / jnp.square(denominator)
    return residual, expected_weight, observed_weight


@jax.jit
def nb_null_residual_and_weight(
    counts: jnp.ndarray,
    nuisance_design: jnp.ndarray,
    offsets: jnp.ndarray,
    theta: jnp.ndarray,
    coefficients: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Score residual and observed weight at fixed coefficients.

    Split out from :func:`fisher_nb_null` so a control-only null - fit on control
    cells alone - can be evaluated on cells that took no part in estimating it.
    """

    eta = offsets + nuisance_design @ coefficients
    residual, _, observed_weight = _nb_null_terms(counts, eta, theta[None, :])
    return residual, observed_weight


def _fisher_nb_null_impl(
    counts: jnp.ndarray,
    nuisance_design: jnp.ndarray,
    offsets: jnp.ndarray,
    theta: jnp.ndarray,
    prior_precision: jnp.ndarray,
    jitter: jnp.ndarray,
    step_tolerance: jnp.ndarray,
    max_iterations: int = 50,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Shared implementation for nuisance-only Fisher-scoring kernels.

    ``counts`` is ``(cells, genes)`` and ``offsets`` broadcasts against it.
    Returns coefficients ``(nuisance, genes)`` plus the score residual and
    observed weight at the fit, both ``(cells, genes)``.

    Convergence is on the Newton step, not on an absolute gradient: the latter
    scales with the cell count and is not representable in float32.
    """

    num_nuisance = nuisance_design.shape[1]
    theta_row = theta[None, :]
    eye = jnp.eye(num_nuisance, dtype=counts.dtype)
    ridge = (prior_precision + jitter) * eye[None, :, :]

    def step(beta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        eta = offsets + nuisance_design @ beta
        residual, expected_weight, _ = _nb_null_terms(counts, eta, theta_row)
        gradient = nuisance_design.T @ residual - prior_precision * beta
        curvature = (
            jnp.einsum("nq,ng,nr->gqr", nuisance_design, expected_weight, nuisance_design) + ridge
        )
        delta = jnp.linalg.solve(curvature, gradient.T[:, :, None])[:, :, 0].T
        return jnp.clip(delta, -_MAX_STEP, _MAX_STEP), gradient

    def body(state):
        beta, _, iteration = state
        delta, _ = step(beta)
        return beta + delta, jnp.max(jnp.abs(delta)), iteration + 1

    def condition(state):
        _, largest_step, iteration = state
        return (iteration < max_iterations) & (largest_step > step_tolerance)

    initial = jnp.zeros((num_nuisance, counts.shape[1]), dtype=counts.dtype)
    initial = initial.at[0].set(
        jnp.log(counts.mean(axis=0) + 0.1) - offsets.mean(axis=0)
    )
    beta, largest_step, iterations = jax.lax.while_loop(
        condition,
        body,
        (initial, jnp.asarray(jnp.inf, dtype=counts.dtype), jnp.asarray(0, dtype=jnp.int32)),
    )
    eta = offsets + nuisance_design @ beta
    residual, _, observed_weight = _nb_null_terms(counts, eta, theta_row)
    final_delta, _ = step(beta)
    per_gene_step = jnp.max(jnp.abs(final_delta), axis=0)
    return beta, residual, observed_weight, per_gene_step, iterations


@partial(jax.jit, static_argnames=("max_iterations",))
def fisher_nb_null(
    counts: jnp.ndarray,
    nuisance_design: jnp.ndarray,
    offsets: jnp.ndarray,
    theta: jnp.ndarray,
    prior_precision: jnp.ndarray,
    jitter: jnp.ndarray,
    step_tolerance: jnp.ndarray,
    max_iterations: int = 50,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fit the nuisance-only NB null for every gene by Fisher scoring.

    ``counts`` is ``(cells, genes)`` and ``offsets`` broadcasts against it.
    Returns coefficients ``(nuisance, genes)`` plus the score residual and
    observed weight at the fit, both ``(cells, genes)``.

    Convergence is on the Newton step, not on an absolute gradient: the latter
    scales with the cell count and is not representable in float32.
    """

    beta, residual, observed_weight, _, _ = _fisher_nb_null_impl(
        counts,
        nuisance_design,
        offsets,
        theta,
        prior_precision,
        jitter,
        step_tolerance,
        max_iterations,
    )
    return beta, residual, observed_weight


@partial(jax.jit, static_argnames=("max_iterations",))
def fit_nb_null_laplace(
    counts: jnp.ndarray,
    nuisance_design: jnp.ndarray,
    offsets: jnp.ndarray,
    theta: jnp.ndarray,
    prior_precision: jnp.ndarray,
    jitter: jnp.ndarray,
    step_tolerance: jnp.ndarray,
    max_iterations: int = 50,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Batched nuisance MAPs and observed-information covariance by gene."""

    beta, _, observed_weight, per_gene_step, iterations = _fisher_nb_null_impl(
        counts,
        nuisance_design,
        offsets,
        theta,
        prior_precision,
        jitter,
        step_tolerance,
        max_iterations,
    )
    num_nuisance = nuisance_design.shape[1]
    ridge = (prior_precision + jitter) * jnp.eye(num_nuisance, dtype=counts.dtype)
    information = (
        jnp.einsum(
            "nq,ng,nr->gqr", nuisance_design, observed_weight, nuisance_design
        )
        + ridge[None, :, :]
    )
    return beta, jnp.linalg.inv(information), per_gene_step, iterations


@partial(jax.jit, static_argnames=("num_batches", "max_iterations"))
def fit_categorical_batch_nb_null_laplace(
    counts: jnp.ndarray,
    batch_codes: jnp.ndarray,
    offsets: jnp.ndarray,
    theta: jnp.ndarray,
    jitter: jnp.ndarray,
    step_tolerance: jnp.ndarray,
    *,
    num_batches: int,
    max_iterations: int = 50,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fit one NB intercept per categorical batch and gene.

    This is algebraically equivalent to an intercept plus reference-coded batch
    dummies, but the Fisher information is diagonal in the full batch-intercept
    parameterization. It intentionally has no nuisance prior: a reference-code
    Gaussian prior is not invariant under this reparameterization.
    """

    theta_row = theta[None, :]
    batch_codes = batch_codes.astype(jnp.int32)

    def predictor(beta: jnp.ndarray) -> jnp.ndarray:
        return offsets + beta[batch_codes]

    def grouped(values: jnp.ndarray) -> jnp.ndarray:
        return jax.ops.segment_sum(values, batch_codes, num_segments=num_batches)

    def step(beta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        residual, expected_weight, _ = _nb_null_terms(counts, predictor(beta), theta_row)
        gradient = grouped(residual)
        curvature = grouped(expected_weight) + jitter
        delta = jnp.clip(gradient / jnp.where(curvature > 0, curvature, 1.0), -_MAX_STEP, _MAX_STEP)
        return delta, gradient

    initial = jnp.zeros((num_batches, counts.shape[1]), dtype=counts.dtype)
    overall = jnp.log(counts.mean(axis=0) + 0.1) - offsets.mean(axis=0)
    initial = initial.at[:].set(overall[None, :])

    def body(state):
        beta, _, iteration = state
        delta, _ = step(beta)
        return beta + delta, jnp.max(jnp.abs(delta)), iteration + 1

    def condition(state):
        _, largest_step, iteration = state
        return (iteration < max_iterations) & (largest_step > step_tolerance)

    beta, _, iterations = jax.lax.while_loop(
        condition,
        body,
        (initial, jnp.asarray(jnp.inf, dtype=counts.dtype), jnp.asarray(0, dtype=jnp.int32)),
    )
    eta = predictor(beta)
    residual, _, observed_weight = _nb_null_terms(counts, eta, theta_row)
    final_delta, _ = step(beta)
    per_gene_step = jnp.max(jnp.abs(final_delta), axis=0)
    variance = 1.0 / jnp.where(grouped(observed_weight) + jitter > 0, grouped(observed_weight) + jitter, jnp.inf)
    return beta, variance, per_gene_step, iterations


@jax.jit
def categorical_batch_nb_null_residual_and_weight(
    counts: jnp.ndarray,
    batch_codes: jnp.ndarray,
    offsets: jnp.ndarray,
    theta: jnp.ndarray,
    batch_coefficients: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate a fitted categorical-batch NB null on arbitrary cells."""

    residual, _, observed_weight = _nb_null_terms(
        counts, offsets + batch_coefficients[batch_codes.astype(jnp.int32)], theta[None, :]
    )
    return residual, observed_weight


@partial(jax.jit, static_argnames=("num_targets", "max_iterations"))
def _solve_effects_segment_all_outputs(
    counts: jnp.ndarray,
    baseline_eta: jnp.ndarray,
    theta: jnp.ndarray,
    cell_index: jnp.ndarray,
    target_index: jnp.ndarray,
    prior_precision: jnp.ndarray,
    step_tolerance: jnp.ndarray,
    num_targets: int,
    max_iterations: int = 50,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Jitted entry point exposing every output of the shared segment solver."""

    return _solve_effects_segment_impl(
        counts, baseline_eta, theta, cell_index, target_index, prior_precision, step_tolerance, num_targets, max_iterations
    )


@partial(jax.jit, static_argnames=("num_targets", "num_batches"))
def _categorical_batch_baseline_variance(
    observed_weight: jnp.ndarray,
    curvature: jnp.ndarray,
    cell_index: jnp.ndarray,
    target_index: jnp.ndarray,
    batch_codes: jnp.ndarray,
    batch_variance: jnp.ndarray,
    *,
    num_targets: int,
    num_batches: int,
) -> jnp.ndarray:
    """Propagate diagonal batch-baseline variance into each target's effect.

    The cross term ``sum_{i in target t, batch b} w_ig`` is a dense
    ``(targets, batches, genes)`` intermediate that exists only to be reduced
    over batches, so it sets the peak memory of the whole categorical fit while
    contributing a ``(targets, genes)`` result. Squaring the raw cross term and
    dividing by ``curvature**2`` after the reduction, rather than forming the
    per-batch sensitivity first, is the same expression with one fewer array of
    that shape live at once.
    """

    flat = target_index.astype(jnp.int32) * num_batches + batch_codes[cell_index].astype(jnp.int32)
    cross = jax.ops.segment_sum(
        observed_weight[cell_index], flat, num_segments=num_targets * num_batches
    ).reshape(num_targets, num_batches, observed_weight.shape[1])
    weighted = jnp.sum(jnp.square(cross) * batch_variance[None, :, :], axis=1)
    return jnp.maximum(weighted / jnp.square(curvature), 0.0)


def categorical_batch_gene_block(
    *,
    num_targets: int,
    num_batches: int,
    num_genes: int,
    max_cross_gib: float | None,
) -> int:
    """Genes per baseline-variance sub-block under a float32 cross-term budget.

    The cross term costs ``num_targets * num_batches * 4`` bytes per gene and
    nothing else in the categorical fit grows with the batch count, so a budget
    on that one tensor is what bounds the fit. On the Replogle genome-wide screen
    - 9,899 targets over 267 gem groups - that is 10.6 MB per gene, so a 400-gene
    driver chunk asks for 3.9 GiB in one allocation.
    """

    if max_cross_gib is None or max_cross_gib <= 0:
        return int(num_genes)
    per_gene = max(int(num_targets) * int(num_batches) * 4, 1)
    budget = int(float(max_cross_gib) * float(1024**3))
    return int(max(1, min(int(num_genes), budget // per_gene)))


def solve_effects_segment_categorical_batch_laplace(
    counts: jnp.ndarray,
    baseline_eta: jnp.ndarray,
    theta: jnp.ndarray,
    cell_index: jnp.ndarray,
    target_index: jnp.ndarray,
    batch_codes: jnp.ndarray,
    batch_variance: jnp.ndarray,
    prior_precision: jnp.ndarray,
    step_tolerance: jnp.ndarray,
    *,
    num_targets: int,
    num_batches: int,
    max_iterations: int = 50,
    max_cross_gib: float | None = 2.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Categorical low-MOI effects with diagonal batch-baseline propagation.

    Unlike its non-batch sibling this is not one jitted region: the effect solve
    is jitted whole, then the baseline-variance propagation runs in gene
    sub-blocks sized by ``max_cross_gib``. Sub-blocking is exact rather than an
    approximation - no term in that propagation couples two genes - and it is the
    only place the batch count enters the memory model at all.
    """

    effects, curvature, observed_weight, per_gene_step, iterations = _solve_effects_segment_all_outputs(
        counts, baseline_eta, theta, cell_index, target_index, prior_precision, step_tolerance, num_targets, max_iterations
    )
    num_genes = int(counts.shape[1])
    block = categorical_batch_gene_block(
        num_targets=num_targets, num_batches=num_batches, num_genes=num_genes, max_cross_gib=max_cross_gib
    )
    variance_kwargs = dict(num_targets=num_targets, num_batches=num_batches)
    if block >= num_genes:
        baseline_variance = _categorical_batch_baseline_variance(
            observed_weight, curvature, cell_index, target_index, batch_codes, batch_variance, **variance_kwargs
        )
    else:
        baseline_variance = jnp.concatenate(
            [
                _categorical_batch_baseline_variance(
                    observed_weight[:, window],
                    curvature[:, window],
                    cell_index,
                    target_index,
                    batch_codes,
                    batch_variance[:, window],
                    **variance_kwargs,
                )
                for window in (slice(start, min(start + block, num_genes)) for start in range(0, num_genes, block))
            ],
            axis=1,
        )
    conditional_variance = 1.0 / curvature
    posterior_scale = jnp.sqrt(conditional_variance + baseline_variance)
    return effects, posterior_scale, conditional_variance, per_gene_step, iterations


def pad_selected_indices(
    indices: jnp.ndarray | list,
    *,
    pad_columns_to: int,
    pad_rows_to: int | None,
    dummy_index: int,
) -> jnp.ndarray:
    """Pad a ``(rows, K)`` index block to a fixed shape with a sentinel index.

    ``efficient_score_from_indices`` is jitted, so a fresh ``K`` - which happens
    naturally here because real perturbations carry different cell counts -
    retraces it: about 70 ms per distinct shape, which dominates the runtime once
    a design has on the order of a hundred differently sized targets. Padding
    every call to the same ``(pad_rows_to, pad_columns_to)`` shape with an
    out-of-range ``dummy_index`` makes the kernel compile once for the whole run.

    The sentinel row it reads from is expected to carry zero weight (see
    :func:`append_zero_weight_row`), so a padded selection contributes zero raw
    information and comes back ``nan`` - it is excluded from the p-value exactly
    like any other non-finite entry, with no separate masking required.
    """

    array = jnp.asarray(indices, dtype=jnp.int32)
    if array.ndim == 1:
        array = array[None, :]
    rows, columns = array.shape
    if columns > pad_columns_to:
        raise ValueError(f"indices has {columns} columns, more than pad_columns_to={pad_columns_to}.")
    padded = jnp.full((rows, pad_columns_to), dummy_index, dtype=jnp.int32)
    padded = padded.at[:, :columns].set(array)
    if pad_rows_to is not None:
        if rows > pad_rows_to:
            raise ValueError(f"indices has {rows} rows, more than pad_rows_to={pad_rows_to}.")
        full = jnp.full((pad_rows_to, pad_columns_to), dummy_index, dtype=jnp.int32)
        padded = full.at[:rows].set(padded)
    return padded


def append_zero_weight_row(*arrays: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    """Append one all-zero sentinel row to each ``(cells, genes)`` array.

    Used together with :func:`pad_selected_indices`: an index equal to the
    original cell count now points at this row, contributing zero score, zero
    weight, and zero nuisance cross-term.
    """

    return tuple(jnp.concatenate([array, jnp.zeros_like(array[:1])], axis=0) for array in arrays)


@jax.jit
def efficient_score_from_indices(
    selected_indices: jnp.ndarray,
    score_residual: jnp.ndarray,
    observation_weight: jnp.ndarray,
    weighted_nuisance: jnp.ndarray,
    nuisance_information_inverse: jnp.ndarray,
    nuisance_score: jnp.ndarray,
) -> jnp.ndarray:
    """Efficient score z-statistics for assignments given as selected rows.

    ``selected_indices`` is ``(assignments, selections)``. Every term is a sum
    over the selected rows, so this is a gather and a reduction rather than a
    sparse matrix product - the indicator has a fixed number of nonzeros per row,
    which makes coordinate formats pure overhead here.
    """

    num_genes = score_residual.shape[1]
    num_nuisance = nuisance_information_inverse.shape[-1]
    score = jnp.take(score_residual, selected_indices, axis=0).sum(axis=1)
    raw_information = jnp.take(observation_weight, selected_indices, axis=0).sum(axis=1)
    cross = jnp.take(weighted_nuisance, selected_indices, axis=0).sum(axis=1)
    cross = cross.reshape(-1, num_genes, num_nuisance)
    score = score - jnp.einsum(
        "bgq,gqr,rg->bg", cross, nuisance_information_inverse, nuisance_score
    )
    projected = jnp.einsum("bgq,gqr,bgr->bg", cross, nuisance_information_inverse, cross)
    efficient_information = raw_information - projected
    valid = efficient_information > 1e-12
    return jnp.where(
        valid, score / jnp.sqrt(jnp.where(valid, efficient_information, 1.0)), jnp.nan
    )


@jax.jit
def batched_efficient_score_from_indices(
    selected_indices: jnp.ndarray,
    score_residual: jnp.ndarray,
    observation_weight: jnp.ndarray,
    weighted_nuisance: jnp.ndarray,
    nuisance_information_inverse: jnp.ndarray,
    nuisance_score: jnp.ndarray,
) -> jnp.ndarray:
    """Score resamples for several targets with one static gather shape.

    ``selected_indices`` has shape ``(targets, assignments, selections)``.
    The nuisance inverse and score correction are target-specific, while the
    residuals and weights are shared from the control-only null fit. Keeping the
    target dimension inside the kernel is what avoids one Python/JAX dispatch
    per perturbation.
    """

    num_genes = score_residual.shape[1]
    num_nuisance = nuisance_information_inverse.shape[-1]
    score = jnp.take(score_residual, selected_indices, axis=0).sum(axis=2)
    raw_information = jnp.take(observation_weight, selected_indices, axis=0).sum(axis=2)
    cross = jnp.take(weighted_nuisance, selected_indices, axis=0).sum(axis=2)
    cross = cross.reshape(selected_indices.shape[0], selected_indices.shape[1], num_genes, num_nuisance)
    score = score - jnp.einsum(
        "tbgq,tgqr,trg->tbg", cross, nuisance_information_inverse, nuisance_score
    )
    projected = jnp.einsum(
        "tbgq,tgqr,tbgr->tbg", cross, nuisance_information_inverse, cross
    )
    efficient_information = raw_information - projected
    valid = efficient_information > 1e-12
    return jnp.where(
        valid, score / jnp.sqrt(jnp.where(valid, efficient_information, 1.0)), jnp.nan
    )


@jax.jit
def prepare_control_only_target_scores(
    target_indices: jnp.ndarray,
    score_residual: jnp.ndarray,
    observation_weight: jnp.ndarray,
    weighted_nuisance: jnp.ndarray | None,
    nuisance_design: jnp.ndarray,
    control_information: jnp.ndarray,
    control_nuisance_score: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Prepare all target-specific efficient-score terms for one size bucket.

    ``target_indices`` is ``(targets, padded_cells)`` with a zero-contribution
    sentinel row already included in every gathered input. The returned inverse,
    score correction, and observed statistic are respectively ``(targets,
    genes, nuisance, nuisance)``, ``(targets, nuisance, genes)``, and
    ``(targets, genes)``. With no resampled assignments, ``weighted_nuisance``
    may be None: its observed cross term uses the target rows already gathered.
    """

    target_nuisance = jnp.take(nuisance_design, target_indices, axis=0)
    target_weight = jnp.take(observation_weight, target_indices, axis=0)
    target_residual = jnp.take(score_residual, target_indices, axis=0)
    information = control_information[None, ...] + jnp.einsum(
        "tkq,tkg,tkr->tgqr", target_nuisance, target_weight, target_nuisance
    )
    nuisance_score = control_nuisance_score[None, ...] + jnp.einsum(
        "tkq,tkg->tqg", target_nuisance, target_residual
    )
    information_inverse = jnp.linalg.inv(information)
    if weighted_nuisance is None:
        # Keep the same product-then-sum and score formulas as the gathered
        # path, without materializing weighted covariates for every cell.
        cross = (target_weight[:, :, :, None] * target_nuisance[:, :, None, :]).sum(axis=1)[:, None, :, :]
        score = target_residual.sum(axis=1)[:, None, :]
        raw_information = target_weight.sum(axis=1)[:, None, :]
        score = score - jnp.einsum("tbgq,tgqr,trg->tbg", cross, information_inverse, nuisance_score)
        projected = jnp.einsum("tbgq,tgqr,tbgr->tbg", cross, information_inverse, cross)
        efficient_information = raw_information - projected
        valid = efficient_information > 1e-12
        observed = jnp.where(
            valid, score / jnp.sqrt(jnp.where(valid, efficient_information, 1.0)), jnp.nan
        )[:, 0, :]
    else:
        observed = batched_efficient_score_from_indices(
            target_indices[:, None, :],
            score_residual,
            observation_weight,
            weighted_nuisance,
            information_inverse,
            nuisance_score,
        )[:, 0, :]
    return information_inverse, nuisance_score, observed


@partial(jax.jit, static_argnames=("num_batches",))
def prepare_control_only_categorical_batch_target_scores(
    target_indices: jnp.ndarray,
    score_residual: jnp.ndarray,
    observation_weight: jnp.ndarray,
    batch_codes: jnp.ndarray,
    control_information: jnp.ndarray,
    control_batch_score: jnp.ndarray,
    *,
    num_batches: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Target-specific diagonal batch information for a control-only CRT."""

    targets, width = target_indices.shape
    num_genes = score_residual.shape[1]
    target_batch = jnp.take(batch_codes, target_indices, axis=0).astype(jnp.int32)
    target_weight = jnp.take(observation_weight, target_indices, axis=0)
    target_residual = jnp.take(score_residual, target_indices, axis=0)
    flat = (jnp.arange(targets, dtype=jnp.int32)[:, None] * num_batches + target_batch).reshape(-1)
    target_information = jax.ops.segment_sum(
        target_weight.reshape(targets * width, num_genes), flat, num_segments=targets * num_batches
    ).reshape(targets, num_batches, num_genes)
    target_score = jax.ops.segment_sum(
        target_residual.reshape(targets * width, num_genes), flat, num_segments=targets * num_batches
    ).reshape(targets, num_batches, num_genes)
    information = control_information[None, :, :] + target_information
    inverse_information = 1.0 / jnp.where(information > 0, information, jnp.inf)
    nuisance_score = control_batch_score[None, :, :] + target_score
    observed = batched_efficient_score_from_indices_categorical_batch(
        target_indices[:, None, :], score_residual, observation_weight, batch_codes,
        inverse_information, nuisance_score, num_batches=num_batches,
    )[:, 0, :]
    return inverse_information, nuisance_score, observed


@partial(jax.jit, static_argnames=("num_batches",))
def batched_efficient_score_from_indices_categorical_batch(
    selected_indices: jnp.ndarray,
    score_residual: jnp.ndarray,
    observation_weight: jnp.ndarray,
    batch_codes: jnp.ndarray,
    inverse_information: jnp.ndarray,
    nuisance_score: jnp.ndarray,
    *,
    num_batches: int,
) -> jnp.ndarray:
    """Efficient scores using diagonal information for categorical batches."""

    targets, assignments, width = selected_indices.shape
    num_genes = score_residual.shape[1]
    selected_residual = jnp.take(score_residual, selected_indices, axis=0)
    selected_weight = jnp.take(observation_weight, selected_indices, axis=0)
    selected_batch = jnp.take(batch_codes, selected_indices, axis=0).astype(jnp.int32)
    raw_score = selected_residual.sum(axis=2)
    raw_information = selected_weight.sum(axis=2)
    group = (
        jnp.arange(targets * assignments, dtype=jnp.int32)[:, None] * num_batches
        + selected_batch.reshape(targets * assignments, width)
    ).reshape(-1)
    cross = jax.ops.segment_sum(
        selected_weight.reshape(targets * assignments * width, num_genes),
        group,
        num_segments=targets * assignments * num_batches,
    ).reshape(targets, assignments, num_batches, num_genes)
    correction = jnp.sum(cross * inverse_information[:, None, :, :] * nuisance_score[:, None, :, :], axis=2)
    projected = jnp.sum(jnp.square(cross) * inverse_information[:, None, :, :], axis=2)
    efficient_information = raw_information - projected
    valid = efficient_information > 1e-12
    return jnp.where(valid, (raw_score - correction) / jnp.sqrt(jnp.where(valid, efficient_information, 1.0)), jnp.nan)


def _solve_effects_segment_impl(
    counts: jnp.ndarray,
    baseline_eta: jnp.ndarray,
    theta: jnp.ndarray,
    cell_index: jnp.ndarray,
    target_index: jnp.ndarray,
    prior_precision: jnp.ndarray,
    step_tolerance: jnp.ndarray,
    num_targets: int,
    max_iterations: int = 50,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Shared implementation for segment-reduced effect solvers.

    ``cell_index`` and ``target_index`` are the coordinates of the nonzero design
    entries. At one guide per cell each cell appears once and this reproduces the
    contiguous-block solver exactly; at a few guides per cell it appears a few
    times and the same reduction applies, which is how MOI > 1 is supported
    without a dense design.

    Returns the effects and their conditional curvature, both
    ``(targets, genes)``. Under MOI > 1 the curvature is the diagonal of the
    effect information, ignoring guide-guide coupling.
    """

    theta_row = theta[None, :]
    num_cells = counts.shape[0]

    def linear_predictor(effects: jnp.ndarray) -> jnp.ndarray:
        # A cell's predictor accumulates every guide it carries, so this is a
        # reduction over cells, not a gather. At MOI 1 each cell has one term and
        # it degenerates to the gather; at MOI > 1 the guide effects add.
        contribution = jax.ops.segment_sum(
            effects[target_index], cell_index, num_segments=num_cells
        )
        return baseline_eta + contribution

    def body(state):
        effects, _, iteration = state
        residual, expected_weight, _ = _nb_null_terms(
            counts, linear_predictor(effects), theta_row
        )
        gradient = (
            jax.ops.segment_sum(residual[cell_index], target_index, num_segments=num_targets)
            - prior_precision * effects
        )
        curvature = (
            jax.ops.segment_sum(
                expected_weight[cell_index], target_index, num_segments=num_targets
            )
            + prior_precision
        )
        delta = jnp.clip(gradient / jnp.where(curvature > 0, curvature, 1.0), -_MAX_STEP, _MAX_STEP)
        return effects + delta, jnp.max(jnp.abs(delta)), iteration + 1

    def condition(state):
        _, largest_step, iteration = state
        return (iteration < max_iterations) & (largest_step > step_tolerance)

    initial = jnp.zeros((num_targets, counts.shape[1]), dtype=counts.dtype)
    effects, _, iterations = jax.lax.while_loop(
        condition,
        body,
        (initial, jnp.asarray(jnp.inf, dtype=counts.dtype), jnp.asarray(0, dtype=jnp.int32)),
    )
    _, _, observed_weight = _nb_null_terms(counts, linear_predictor(effects), theta_row)
    curvature = (
        jax.ops.segment_sum(observed_weight[cell_index], target_index, num_segments=num_targets)
        + prior_precision
    )
    residual, expected_weight, _ = _nb_null_terms(
        counts, linear_predictor(effects), theta_row
    )
    gradient = (
        jax.ops.segment_sum(residual[cell_index], target_index, num_segments=num_targets)
        - prior_precision * effects
    )
    expected_curvature = (
        jax.ops.segment_sum(
            expected_weight[cell_index], target_index, num_segments=num_targets
        )
        + prior_precision
    )
    final_step = gradient / jnp.where(expected_curvature > 0, expected_curvature, 1.0)
    per_gene_step = jnp.max(jnp.abs(final_step), axis=0)
    return effects, curvature, observed_weight, per_gene_step, iterations


@partial(jax.jit, static_argnames=("num_targets", "max_iterations"))
def solve_effects_segment(
    counts: jnp.ndarray,
    baseline_eta: jnp.ndarray,
    theta: jnp.ndarray,
    cell_index: jnp.ndarray,
    target_index: jnp.ndarray,
    prior_precision: jnp.ndarray,
    step_tolerance: jnp.ndarray,
    num_targets: int,
    max_iterations: int = 50,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Newton-solve every target's effect from sparse design coordinates.

    Returns effects and observed conditional curvature, both
    ``(targets, genes)``. Under MOI > 1 the curvature is the diagonal of the
    effect information, ignoring guide-guide coupling.
    """

    effects, curvature, _, _, _ = _solve_effects_segment_impl(
        counts,
        baseline_eta,
        theta,
        cell_index,
        target_index,
        prior_precision,
        step_tolerance,
        num_targets,
        max_iterations,
    )
    return effects, curvature


@partial(jax.jit, static_argnames=("num_targets", "max_iterations"))
def solve_effects_segment_laplace(
    counts: jnp.ndarray,
    baseline_eta: jnp.ndarray,
    theta: jnp.ndarray,
    cell_index: jnp.ndarray,
    target_index: jnp.ndarray,
    nuisance_design: jnp.ndarray,
    baseline_covariance: jnp.ndarray,
    prior_precision: jnp.ndarray,
    step_tolerance: jnp.ndarray,
    num_targets: int,
    max_iterations: int = 50,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Effect MAPs and control-uncertainty propagation entirely on device."""

    effects, curvature, observed_weight, per_gene_step, iterations = (
        _solve_effects_segment_impl(
            counts,
            baseline_eta,
            theta,
            cell_index,
            target_index,
            prior_precision,
            step_tolerance,
            num_targets,
            max_iterations,
        )
    )
    weighted_nuisance = (
        observed_weight[cell_index, :, None] * nuisance_design[cell_index, None, :]
    )
    nuisance_cross = jax.ops.segment_sum(
        weighted_nuisance, target_index, num_segments=num_targets
    )
    sensitivity = -nuisance_cross / curvature[:, :, None]
    baseline_variance = jnp.maximum(
        jnp.einsum(
            "tgq,gqr,tgr->tg", sensitivity, baseline_covariance, sensitivity
        ),
        0.0,
    )
    conditional_variance = 1.0 / curvature
    posterior_scale = jnp.sqrt(conditional_variance + baseline_variance)
    return effects, posterior_scale, conditional_variance, per_gene_step, iterations


@jax.jit
def resample_score_reductions(scores, observed, num_valid_rows):
    """Reduce one ``target x resample x gene`` score block over its resamples.

    Everything the CRT consumes from a resample block is a reduction over the
    resample axis: the finite count, the exceedance count, and the first four
    power sums. Doing them here means only those six ``target x gene`` arrays
    cross to the host, rather than the block itself once per resample chunk.

    Rows past ``num_valid_rows`` are the trailing block's padding. They are
    masked rather than sliced so the traced shape does not depend on the block
    length, which would otherwise force a second compilation for the last chunk.
    ``num_valid_rows`` is deliberately a traced argument for the same reason.

    The power sums come back in float32 and the caller accumulates them across
    blocks in float64, which keeps float32 confined to within-block sums. The
    tightest downstream consumer is the excess kurtosis the Cornish-Fisher tail
    differences down from ``central_4 / var^2 - 3``; measured against a float64
    host recomputation it lands 4.8e-7 absolute at a 64-resample block and
    6.0e-7 at 299, against values of order 1e-2 to 1. So the block-wise split is
    cheap insurance rather than a necessity - a full-length float32 accumulation
    would very likely also be fine - but it costs nothing and removes the
    question.

    Exceedance counting is bit identical to the host version it replaces: both
    compare float32 magnitudes, the host one having merely widened them to
    float64 first, which changes no comparison.
    """
    rows = jnp.arange(scores.shape[1])[None, :, None] < num_valid_rows
    finite = jnp.isfinite(scores) & rows
    values = jnp.where(finite, scores, 0.0)
    exceeded = finite & (jnp.abs(scores) >= jnp.abs(observed)[:, None, :])
    square = values * values
    return (
        finite.sum(axis=1),
        exceeded.sum(axis=1),
        values.sum(axis=1),
        square.sum(axis=1),
        (square * values).sum(axis=1),
        (square * square).sum(axis=1),
    )


__all__ = [
    "append_zero_weight_row",
    "batched_efficient_score_from_indices",
    "efficient_score_from_indices",
    "resample_score_reductions",
    "fit_nb_null_laplace",
    "fisher_nb_null",
    "nb_null_residual_and_weight",
    "pad_selected_indices",
    "prepare_control_only_target_scores",
    "solve_effects_segment",
    "solve_effects_segment_laplace",
]
