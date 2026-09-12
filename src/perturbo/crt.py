"""Conditional randomization test (CRT) against the production SVI baseline.

The CRT is a negative-binomial score-permutation test: for each (perturbation,
gene) pair it forms an efficient score statistic under the null that the
perturbation label carries no information, then calibrates it by resampling the
label within a pool of control cells. The statistic and its resampling machinery
live in :mod:`perturbo._internal.score_resampling`; this module is the layer
that lets the *production* two-stage fit supply the null instead of the
research pipeline's own control-only GLM fit.

Why that substitution is sound, and where it is not
---------------------------------------------------
Per gene ``g``, with ``eta_i = o_i + z_i' gamma_g`` and ``mu = exp(eta)``, the
test consumes exactly four things: the offsets ``o``, the dispersion ``theta``,
the nuisance design ``Z``, and the nuisance coefficients ``gamma``. It needs
nothing else from whatever produced them.

* ``gamma`` is **first-order neutral**. The statistic subtracts
  ``x'WZ (Z'WZ)^-1 Z'r``, and that correction is precisely the first-order
  Taylor term for ``gamma`` being off the null mode: writing
  ``gamma~ = gamma^ + delta`` gives ``r(gamma~) ~ r(gamma^) - W Z delta``, whose
  contribution the correction cancels, leaving ``O(||delta||^2)``. So a
  variational posterior median, complete with prior shrinkage, substitutes for
  the GLM MLE at second-order cost.
* The first-order neutrality above holds for the projection *over the pool the
  target is tested against*. The contributions arrive efficient under the
  control-only fit, so the saddlepoint assembly re-projects each target's pool
  onto the pool's nuisance fit (``fit_low_moi_propensity_saddlepoint`` with the
  ``weight`` group of arguments); without that step a target's out-of-sample
  residuals inflate the statistic by a factor near ``1 + W_own / W_controls``.
* ``theta`` affects **power, never calibration**. Both the observed and the
  resampled statistics are computed from the same fixed ``r`` and ``W``, so a
  misspecified dispersion rescales them together.
* Neither of the above can break type-I error at all. CRT validity is
  *conditional*: given fixed ``(r, W, Z, I)``, the statistic is a deterministic
  function of the label, so resampling the label is exact whatever estimated
  those quantities.
* ``o`` is the exception, and it is a validity break rather than an efficiency
  one. Under ``--size-factor-mode=infer`` the offset is a per-cell latent fit by
  SVI, and in stage two it is fit *jointly with the effect* on the perturbed
  cells. A perturbed cell's offset then depends on its own label, which is
  exactly what exchangeability forbids. Hence :func:`validate_crt_config`
  requires observed or fixed offsets.

Because ``gamma`` is only first-order neutral, the expansion has to actually be
in its valid regime. :func:`check_baseline_is_null_mode` is that guard: it
measures the distance from the supplied baseline to the control-cell null mode
and lets the caller refuse to proceed when the baseline is not near it.
"""

from __future__ import annotations

import dataclasses

import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from perturbo.core import PerTurboData, ControlFit, _normalize_likelihood_name
from perturbo.sparse_design import IndexedDesignMatrix
from perturbo._internal.chunked_runner import iter_gene_chunks, replace_gene_axis
from perturbo._internal.low_moi.design import prepare_low_moi_design
from perturbo._internal.parametric_null import (
    fit_skew_normal_from_moments,
    fit_student_t_from_moments,
)
from perturbo._internal.arrow_nuisance import batch_codes_from_one_hot, is_intercept_and_one_hot
from perturbo._internal.score_resampling import (
    _benjamini_hochberg,
    _nuisance_information,
    evaluate_nb_null_components,
    newton_step_magnitude,
    precompute_low_moi_permutations,
    run_low_moi_score_permutations,
)
from perturbo.utils import compute_size_factors

__all__ = [
    "BaselineNullCheck",
    "AllCellsPropensityFit",
    "ChunkCRTResult",
    "ControlBlock",
    "ControlNuisance",
    "CRTAccumulator",
    "CRTBaseline",
    "CRT_CONTROL_NAME",
    "CRT_TAIL_FAMILIES",
    "CRT_SADDLEPOINT_FAMILY",
    "CRT_ALL_TAIL_FAMILIES",
    "CRT_MECHANISMS",
    "fit_tail_families",
    "DEFAULT_NEWTON_STEP_TOLERANCE",
    "SUPPORTED_LIKELIHOODS",
    "SUPPORTED_SIZE_FACTOR_MODES",
    "assemble_control_nuisance",
    "build_chunk_design",
    "check_baseline_is_null_mode",
    "control_block_for_genes",
    "exclude_targets",
    "prepare_crt_baseline",
    "prepare_all_cells_propensity",
    "polish_baseline_to_null_mode",
    "run_crt_for_chunk",
    "run_crt_all_cells",
    "CRT_POOLS",
    "validate_crt_config",
    "validate_offset_compatibility",
]

# Reserved name for the pooled control pool inside a chunk design. Chunk
# perturbation names come from the user's own categories, so this must be
# something they cannot collide with by accident.
CRT_CONTROL_NAME = "__crt_control__"

# Every parametric tail family is fitted from the same resampled moments, so
# running all of them costs one extra host-side pass each rather than another
# resampling run. They are reported side by side deliberately: each makes a
# different shape assumption about the null's far tail, none is uniformly best,
# and at genome-wide scale the empirical p-value cannot reach a usable FDR
# threshold at all - it floors at 1/(resamples+1), which over millions of pairs
# is orders of magnitude above the Benjamini-Hochberg cutoff. Disagreement
# between the families is the honest signal that the extrapolation is being
# pushed, and hiding it behind a single default would suppress that.
CRT_TAIL_FAMILIES: tuple[str, ...] = ("skew_normal", "student_t")

# The kernel-side saddlepoint tail. Under the propensity mechanism it is the
# exact Bernoulli-sum CGF with a Pearson III screen and needs no draws at all;
# under permutation it is the stratified with-replacement surrogate and should
# be read as a diagnostic.
CRT_SADDLEPOINT_FAMILY = "saddlepoint"
CRT_ALL_TAIL_FAMILIES: tuple[str, ...] = (*CRT_TAIL_FAMILIES, CRT_SADDLEPOINT_FAMILY)

# How a target's label is resampled inside its pool (controls plus its own
# cells): a stratified permutation holding the count fixed, or model-X
# Bernoulli draws at each cell's fitted selection probability.
CRT_MECHANISMS: tuple[str, ...] = ("permutation", "propensity")

# Version one deliberately supports only the plain NB likelihood. The score
# residual and weight below are the NB ones; a lognormal-NB adds a per-cell
# multiplicative latent, and a mixture NB's score is a posterior-weighted blend
# of two component scores. Neither is this formula, so neither may silently use
# it. Censored NB is the cheap extension (mask r and w on censored cells).
SUPPORTED_LIKELIHOODS = ("negbin",)

# "observed" conditions on supplied/computed size factors; "none" fixes them to
# zero. Both leave the offset a fixed, label-independent quantity, which is what
# the CRT requires. "infer" does not - see the module docstring.
SUPPORTED_SIZE_FACTOR_MODES = ("observed", "none")

# Newton step magnitude, in nats of the linear predictor, below which a supplied
# baseline counts as "at the null mode". A step of 0.05 means the fitted mean is
# within roughly 5% of the control-cell mode.
#
# This is a judgement call, not a theorem. It is far looser than the 1e-4 a
# converged GLM reaches, because an SVI median is not trying to solve the score
# equation; it is tight enough that the O(||delta||^2) residual bias in the
# statistic stays well under the float32 noise the score kernels already carry.
# Inspect the reported distribution on a new dataset before trusting it blindly.
DEFAULT_NEWTON_STEP_TOLERANCE = 0.05


def validate_crt_config(
    *,
    likelihood: str,
    size_factor_mode: str,
    num_factors: int | None = None,
    guide_random_effects: bool = False,
    retain_guide_structure: bool = False,
) -> None:
    """Reject configurations the low-MOI NB CRT does not describe.

    Every problem is collected and reported together. A CRT run is preceded by
    a full stage-one fit, so surfacing one flag at a time would cost the user a
    training run per mistake; this is meant to be called during argument
    parsing, before any fitting starts.

    Raises:
        ValueError: if any part of the configuration is unsupported, with one
            line per problem naming the remedy.
    """

    problems: list[str] = []

    normalized_likelihood = _normalize_likelihood_name(likelihood)
    if normalized_likelihood not in SUPPORTED_LIKELIHOODS:
        problems.append(
            f"likelihood={likelihood!r} is not supported by the CRT (supported: 'nb'/'negbin'). "
            "The score residual and weight are the plain negative-binomial ones; other "
            "likelihoods have a different score function. Re-run with --likelihood nb."
        )

    mode = str(size_factor_mode).lower()
    if mode not in SUPPORTED_SIZE_FACTOR_MODES:
        problems.append(
            f"size_factor_mode={size_factor_mode!r} is not supported by the CRT "
            f"(supported: {list(SUPPORTED_SIZE_FACTOR_MODES)}). Latent size factors are fit "
            "jointly with the effect on the perturbed cells, so a cell's offset depends on its "
            "own perturbation label and the resampling null is no longer exact. "
            "Re-run with --size-factor-mode observed."
        )

    if num_factors:
        problems.append(
            f"num_factors={num_factors} is not supported by the CRT. Factor scores are per-cell "
            "latents fit in stage two and can absorb perturbation signal directly, which breaks "
            "the same exchangeability argument as latent size factors. Re-run with --num-factors 0."
        )

    if guide_random_effects:
        problems.append(
            "guide_random_effects=True is not supported by the CRT. The guide random effect is a "
            "per-guide latent that shifts the null mean per cell and is not part of the nuisance "
            "design the score test projects out. Re-run without --guide-random-effects."
        )

    # A guide-to-element map is fine for the control-anchored test: the chunk's
    # assignment matrix is collapsed to elements, and a cell that then carries more
    # than one element is set aside (counted and reported) rather than reinterpreted,
    # because the null needs one assignment per cell. `retain_guide_structure` is
    # accepted for that reason and no longer refused.
    del retain_guide_structure

    if problems:
        raise ValueError(
            "Unsupported configuration for the CRT:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )


@dataclass(frozen=True)
class ControlNuisance:
    """Control-cell null quantities in the form the score test consumes.

    ``nuisance_design`` is ``(cells, 1 + covariates)`` and ``coefficients`` is
    the matching ``(1 + covariates, genes)``: the intercept row is stage one's
    ``beta_0`` and the remaining rows are its ``covariate_coef``, in the column
    order the covariate transform produced. ``offsets`` is ``(cells, 1)``.
    """

    counts: np.ndarray
    nuisance_design: np.ndarray
    coefficients: np.ndarray
    offsets: np.ndarray
    dispersion: np.ndarray
    nuisance_names: tuple[str, ...]
    gene_names: tuple[str, ...]


def assemble_control_nuisance(
    control_data: PerTurboData,
    control_fit: ControlFit,
) -> ControlNuisance:
    """Lift a production stage-one fit into the score test's nuisance form.

    Stacking is direct: ``covariate_coef`` is already ``(covariates, genes)``
    from the covariate plate, so it concatenates under ``beta_0`` without a
    transpose, and the resulting row order matches ``[intercept, *covariates]``
    in the design.

    Offsets are read from ``control_data`` rather than recomputed, so the CRT
    conditions on the same per-cell offsets stage one did. Under a fixed-zero
    size-factor mode those are zeros and this is still correct.
    """

    # Keep the source dtype here. The baseline owns this panel for the whole
    # CRT run, and eagerly widening a float32 count matrix to float64 doubles
    # its resident CPU footprint. Numerical kernels cast only the active gene
    # block to float64 when they need diagnostic precision.
    counts = np.asarray(control_data.counts)
    if counts.ndim != 2:
        raise ValueError("control_data.counts must be a cells-by-genes matrix.")
    num_cells, num_genes = counts.shape

    beta_0 = np.asarray(control_fit.beta_0, dtype=np.float64).reshape(-1)
    if beta_0.shape != (num_genes,):
        raise ValueError(
            f"control_fit.beta_0 has {beta_0.size} entries but the control counts have "
            f"{num_genes} genes."
        )

    design_blocks = [np.ones((num_cells, 1), dtype=np.float64)]
    coefficient_blocks = [beta_0[None, :]]
    nuisance_names = ["intercept"]

    covariates = control_data.covariates
    if covariates is not None and np.asarray(covariates).size:
        covariate_matrix = np.asarray(covariates, dtype=np.float64)
        if covariate_matrix.ndim != 2 or covariate_matrix.shape[0] != num_cells:
            raise ValueError("control_data.covariates must have shape (n_cells, n_covariates).")
        if control_fit.covariate_coef is None:
            raise ValueError(
                "control_data carries covariates but control_fit.covariate_coef is missing. "
                "Run fit_control with the same covariates first."
            )
        covariate_coef = np.asarray(control_fit.covariate_coef, dtype=np.float64)
        if covariate_coef.shape != (covariate_matrix.shape[1], num_genes):
            raise ValueError(
                "control_fit.covariate_coef must have shape (n_covariates, n_genes); got "
                f"{covariate_coef.shape} for {covariate_matrix.shape[1]} covariates and "
                f"{num_genes} genes."
            )
        design_blocks.append(covariate_matrix)
        coefficient_blocks.append(covariate_coef)
        names = control_data.covariate_names
        if names is None:
            nuisance_names.extend(f"covariate_{index}" for index in range(covariate_matrix.shape[1]))
        elif len(names) == covariate_matrix.shape[1]:
            nuisance_names.extend(str(name) for name in names)
        else:
            raise ValueError("control_data.covariate_names must contain one name per covariate.")

    offsets = control_data.size_factors
    offsets = compute_size_factors(control_data.counts) if offsets is None else offsets
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.ndim == 1:
        offsets = offsets[:, None]
    if offsets.shape not in {(num_cells, 1), (num_cells, num_genes)}:
        raise ValueError("control size factors must be (cells,), (cells, 1), or (cells, genes).")

    dispersion = np.asarray(control_fit.theta, dtype=np.float64).reshape(-1)
    if dispersion.shape != (num_genes,):
        raise ValueError(
            f"control_fit.theta has {dispersion.size} entries but the control counts have "
            f"{num_genes} genes."
        )
    if np.any(~np.isfinite(dispersion)) or np.any(dispersion <= 0):
        raise ValueError("control_fit.theta must be finite and strictly positive for every gene.")

    return ControlNuisance(
        counts=counts,
        nuisance_design=np.concatenate(design_blocks, axis=1),
        coefficients=np.concatenate(coefficient_blocks, axis=0),
        offsets=offsets,
        dispersion=dispersion,
        nuisance_names=tuple(nuisance_names),
        gene_names=tuple(str(name) for name in control_data.gene_names),
    )


@dataclass(frozen=True)
class BaselineNullCheck:
    """How far a supplied baseline sits from the control-cell null mode.

    ``newton_step`` is per-gene ``max |(Z'WZ)^-1 Z'r|`` in nats of the linear
    predictor. ``degenerate`` flags genes with no control counts at all, whose
    null mode does not exist; they are excluded from the verdict rather than
    reported as failures, since no baseline could satisfy them.
    """

    newton_step: np.ndarray
    degenerate: np.ndarray
    tolerance: float
    gene_names: tuple[str, ...] | None = None

    @property
    def failed(self) -> np.ndarray:
        """Per-gene boolean: too far from the mode, and not degenerate."""

        return (~self.degenerate) & ~(self.newton_step <= self.tolerance)

    @property
    def num_failed(self) -> int:
        return int(np.count_nonzero(self.failed))

    @property
    def ok(self) -> bool:
        return self.num_failed == 0

    def describe(self, *, max_genes: int = 5) -> str:
        """One-paragraph summary suitable for logging or an error message."""

        testable = ~self.degenerate
        num_testable = int(np.count_nonzero(testable))
        num_degenerate = int(np.count_nonzero(self.degenerate))
        if num_testable == 0:
            return (
                f"No testable genes: all {self.degenerate.size} have zero total control counts."
            )
        steps = self.newton_step[testable]
        lines = [
            f"Baseline-vs-null-mode Newton step over {num_testable} genes: "
            f"median {np.median(steps):.2e}, 99th pct {np.quantile(steps, 0.99):.2e}, "
            f"max {steps.max():.2e} nats (tolerance {self.tolerance:.2e}); "
            f"{self.num_failed} gene(s) over tolerance"
            + (f", {num_degenerate} degenerate gene(s) excluded." if num_degenerate else ".")
        ]
        if self.num_failed:
            failed_indices = np.flatnonzero(self.failed)
            order = failed_indices[np.argsort(-self.newton_step[failed_indices])][:max_genes]
            if self.gene_names is not None:
                worst = ", ".join(f"{self.gene_names[i]} ({self.newton_step[i]:.2e})" for i in order)
            else:
                worst = ", ".join(f"gene {i} ({self.newton_step[i]:.2e})" for i in order)
            lines.append(f"Worst: {worst}.")
        return " ".join(lines)


def check_baseline_is_null_mode(
    nuisance: ControlNuisance,
    *,
    step_tolerance: float = DEFAULT_NEWTON_STEP_TOLERANCE,
    curvature_jitter: float = 1e-8,
) -> BaselineNullCheck:
    """Measure how far the supplied baseline is from the control-cell null mode.

    Reusing the SVI baseline is justified by a first-order expansion around the
    null mode, so this measures the expansion point: the single Fisher-scoring
    step ``(Z'WZ)^-1 Z'r`` that would move the supplied coefficients onto it.
    A small step means the correction term the statistic already applies is
    doing its job in the regime where it is accurate.

    The step is computed **unpenalized**. Production puts ``Normal(0, 3)`` on
    ``beta_0`` and ``Normal(0, 1)`` on ``covariate_coef``, but the quantity that
    governs the expansion is the distance to the unpenalized score-equation
    root, and at realistic control-cell counts the prior contributes a
    negligible fraction of the information anyway.

    The step reads as nats of displacement only *near* the mode, where the NB
    log-likelihood is approximately quadratic. Further out a single step
    undershoots, so the reported number is a lower bound on the true distance
    rather than an estimate of it. That is the safe direction for a guard - it
    never invents a failure - but a value close to tolerance should not be read
    as "only just off".

    This reports rather than raises, so a caller can log the distribution before
    deciding. Nothing about a large step is automatically fatal to *validity* -
    the resampling null stays exact regardless - but it does mean the statistic
    is no longer the efficient score for the model that produced the baseline.
    """

    if not np.isfinite(step_tolerance) or step_tolerance <= 0:
        raise ValueError("step_tolerance must be finite and positive.")

    num_nuisance = nuisance.nuisance_design.shape[1]
    magnitude = newton_step_magnitude(
        nuisance.counts,
        nuisance.nuisance_design,
        nuisance.offsets,
        nuisance.dispersion,
        nuisance.coefficients,
        prior_precision=0.0,
        ridge=float(curvature_jitter) * np.eye(num_nuisance),
    )
    return BaselineNullCheck(
        newton_step=magnitude,
        degenerate=nuisance.counts.sum(axis=0) <= 0,
        tolerance=float(step_tolerance),
        gene_names=nuisance.gene_names,
    )


def validate_offset_compatibility(control_data: PerTurboData, chunk_data: PerTurboData) -> None:
    """Require chunk offsets to be centered exactly as the control offsets were.

    Size factors are centered log library sizes, and the centering constant is a
    property of the *run*, not of the cells in front of you. If a chunk were
    centered on its own cells, its offsets would sit at a different origin from
    the control cells that produced ``beta_0``, and every score residual in that
    chunk would inherit a constant shift - a silent, chunk-specific bias that no
    downstream diagnostic would attribute to its cause.

    Production already threads the control constant through
    ``load_analysis_cells``; this refuses to run if that ever stops happening.
    """

    control_center = control_data.library_size_center_log_mean
    chunk_center = chunk_data.library_size_center_log_mean
    if control_center is None and chunk_center is None:
        return
    if control_center is None or chunk_center is None:
        raise ValueError(
            "Control and chunk size factors must share a centering constant, but only one "
            f"carries one (control={control_center!r}, chunk={chunk_center!r}). Load analysis "
            "cells with library_size_center_log_mean taken from the control load."
        )
    if not np.isclose(float(control_center), float(chunk_center), rtol=0.0, atol=1e-6):
        raise ValueError(
            "Chunk size factors are centered differently from the control cells "
            f"(control={float(control_center):.8g}, chunk={float(chunk_center):.8g}). "
            "beta_0 is on the control centering, so this would shift every score residual "
            "in this chunk by a constant."
        )


def polish_baseline_to_null_mode(
    nuisance: ControlNuisance,
    *,
    curvature_jitter: float = 1e-8,
    max_iterations: int = 50,
    gene_block: int = 2000,
) -> ControlNuisance:
    """Move the nuisance coefficients onto the control-cell null mode.

    The CRT statistic is the efficient score *at the null mode*; a stage-one
    SVI baseline is a posterior median under its own priors and, with a batch
    covariate, sits a few hundredths of a nat away from that mode. The
    resampling null stays exact wherever the expansion point is, but the
    statistic loses efficiency. With the dispersion fixed the NB log-likelihood
    is concave in the coefficients, so the mode is unique and Fisher scoring
    from any start reaches it: this fits the unpenalized control-only null per
    gene on the device, in gene blocks, and keeps stage one's ``theta``,
    offsets and design. Genes with no control counts keep their supplied
    coefficients, since their mode is not defined.
    """

    from perturbo._internal.jax_kernels import fisher_nb_null

    counts = np.asarray(nuisance.counts)
    num_cells, num_genes = counts.shape
    if gene_block < 1 or max_iterations < 1:
        raise ValueError("gene_block and max_iterations must be positive.")
    degenerate = counts.sum(axis=0) <= 0
    coefficients = np.array(nuisance.coefficients, dtype=np.float64, copy=True)
    design = jnp.asarray(nuisance.nuisance_design, dtype=jnp.float32)
    offsets = np.asarray(nuisance.offsets, dtype=np.float64)
    theta = np.asarray(nuisance.dispersion, dtype=np.float64)
    for start in range(0, num_genes, gene_block):
        stop = min(start + gene_block, num_genes)
        block_offsets = offsets if offsets.shape[1] == 1 else offsets[:, start:stop]
        beta, _, _ = fisher_nb_null(
            jnp.asarray(counts[:, start:stop], dtype=jnp.float32),
            design,
            jnp.asarray(block_offsets, dtype=jnp.float32),
            jnp.asarray(theta[start:stop], dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(float(curvature_jitter), dtype=jnp.float32),
            jnp.asarray(1e-6, dtype=jnp.float32),
            max_iterations=max_iterations,
        )
        fitted = np.asarray(beta, dtype=np.float64)
        block = coefficients[:, start:stop]
        keep = ~degenerate[start:stop] & np.all(np.isfinite(fitted), axis=0)
        block[:, keep] = fitted[:, keep]
        coefficients[:, start:stop] = block
    return dataclasses.replace(nuisance, coefficients=coefficients)


@dataclass(frozen=True)
class CRTBaseline:
    """Everything the CRT needs from stage one, validated once and reused.

    Held across every perturbation chunk. The control cells are the same for all
    of them, so the expensive per-gene quantities derived from them are computed
    once here rather than rebuilt per chunk.

    Deliberately stores the control *counts* rather than precomputed residuals
    and weights. Those are each cells-by-genes, so materializing them for the
    whole panel would triple the resident control footprint, while recomputing
    a gene slice's worth is a single elementwise pass. See
    :func:`control_block_for_genes`.
    """

    nuisance: ControlNuisance
    null_check: BaselineNullCheck
    curvature_jitter: float
    pre_polish_check: BaselineNullCheck | None = None
    """The null-mode check of the supplied baseline before polishing, when
    :func:`prepare_crt_baseline` was asked to polish; ``None`` otherwise."""

    @property
    def gene_names(self) -> tuple[str, ...]:
        return self.nuisance.gene_names

    @property
    def num_genes(self) -> int:
        return len(self.nuisance.gene_names)

    @property
    def num_control_cells(self) -> int:
        return int(self.nuisance.counts.shape[0])


def prepare_crt_baseline(
    control_data: PerTurboData,
    control_fit: ControlFit,
    *,
    step_tolerance: float = DEFAULT_NEWTON_STEP_TOLERANCE,
    curvature_jitter: float = 1e-8,
    strict: bool = True,
    polish: bool = False,
) -> CRTBaseline:
    """Lift a production stage-one fit into a validated CRT null.

    ``strict`` decides what happens when the baseline is not at the control-cell
    null mode. Because this path reuses the SVI baseline rather than refitting,
    that check is the only thing standing between an under-trained stage one and
    a statistic that is quietly no longer the efficient score, so it fails hard
    by default. Set ``strict=False`` to downgrade it to a warning when you want
    to inspect the result anyway.

    ``polish`` first moves the coefficients onto the control-cell null mode
    (:func:`polish_baseline_to_null_mode`), keeping stage one's dispersion,
    offsets and design; the check then measures the polished baseline and the
    supplied one's check is kept as ``pre_polish_check``.
    """

    nuisance = assemble_control_nuisance(control_data, control_fit)
    pre_polish_check = None
    if polish:
        pre_polish_check = check_baseline_is_null_mode(
            nuisance, step_tolerance=step_tolerance, curvature_jitter=curvature_jitter
        )
        nuisance = polish_baseline_to_null_mode(nuisance, curvature_jitter=curvature_jitter)
    null_check = check_baseline_is_null_mode(
        nuisance,
        step_tolerance=step_tolerance,
        curvature_jitter=curvature_jitter,
    )
    if not null_check.ok:
        message = (
            "The stage-one baseline is not at the control-cell null mode, so the CRT statistic "
            "would not be the efficient score for the fitted model. "
            f"{null_check.describe()} Train stage one longer, or pass strict=False to proceed "
            "anyway (the resampling null stays exact either way; what degrades is power and the "
            "interpretation of the statistic)."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return CRTBaseline(
        nuisance=nuisance,
        null_check=null_check,
        curvature_jitter=float(curvature_jitter),
        pre_polish_check=pre_polish_check,
    )


@dataclass(frozen=True)
class ControlBlock:
    """Control-cell null quantities for one gene slice.

    ``information`` is ``(genes, nuisance, nuisance)`` and ``nuisance_score`` is
    ``(nuisance, genes)``. Both are the control cells' contribution only; a
    target adds its own cells to each before its statistic is formed, which is
    what makes the control side shareable across every target and every chunk.

    ``nuisance_score`` is ``Z'r`` over the controls and is emphatically *not*
    assumed to be zero. It would be, had the baseline been fit by solving this
    gene's score equation on exactly these cells - but it came from a
    variational posterior instead, so the term is carried explicitly and
    subtracted. Dropping it inflates the statistic substantially.
    """

    score_residual: np.ndarray
    observation_weight: np.ndarray
    information: np.ndarray
    nuisance_score: np.ndarray
    gene_names: tuple[str, ...]


def control_block_for_genes(baseline: CRTBaseline, gene_slice: slice | None = None) -> ControlBlock:
    """Control-side residuals, weights, and nuisance blocks for a gene slice.

    Every quantity is independent across genes, so slicing the gene axis is a
    decomposition rather than an approximation: a slice's numbers are what the
    full-panel call would have produced for those columns.

    The information is unpenalized. Production puts ``Normal(0, 3)`` on
    ``beta_0``, but the score test projects the nuisance out as a fixed effect,
    and the quantity to project against is the observed information ``Z'WZ``.
    Only a jitter is added, to keep the per-gene solve well posed.
    """

    nuisance = baseline.nuisance
    columns = slice(None) if gene_slice is None else gene_slice
    counts = nuisance.counts[:, columns]
    dispersion = nuisance.dispersion[columns]
    coefficients = nuisance.coefficients[:, columns]
    offsets = nuisance.offsets if nuisance.offsets.shape[1] == 1 else nuisance.offsets[:, columns]

    residual, weight = evaluate_nb_null_components(
        counts,
        nuisance_design=nuisance.nuisance_design,
        offsets=offsets,
        theta=dispersion,
        coefficients=coefficients,
    )
    num_nuisance = nuisance.nuisance_design.shape[1]
    information = _nuisance_information(
        nuisance.nuisance_design,
        weight,
        ridge=baseline.curvature_jitter * np.eye(num_nuisance),
    )
    return ControlBlock(
        score_residual=residual,
        observation_weight=weight,
        information=information,
        nuisance_score=nuisance.nuisance_design.T @ residual,
        gene_names=nuisance.gene_names[columns],
    )


def _cell_axis(values, rows: int, name: str) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.shape[0] != rows:
        raise ValueError(f"{name} must have one row per cell; got {array.shape[0]} for {rows} cells.")
    return array


def _stacked_batch_coding(control_data, chunk_data, control_covariates, chunk_covariates) -> dict:
    """Carry a categorical batch through to the stacked design, if there is one.

    The score kernel has a categorical path that keeps the nuisance information
    diagonal instead of forming ``(genes, nuisance, nuisance)``, and it is
    selected by the design carrying batch codes. Without this the CLI's CRT
    silently took the dense path: at genome-wide scale that is 267 nuisance
    columns and a gather two orders of magnitude larger than it needs to be.

    Explicit codes on the inputs win. Failing that the codes are recovered from
    a reference-coded one-hot covariate block, which is what
    ``--batch-covariate`` produces. Anything else returns nothing and the dense
    path stands.
    """

    explicit_control = getattr(control_data, "categorical_batch_codes", None)
    explicit_chunk = getattr(chunk_data, "categorical_batch_codes", None)
    if explicit_control is not None and explicit_chunk is not None:
        names = getattr(control_data, "categorical_batch_names", None)
        codes = np.concatenate(
            [np.asarray(explicit_control).reshape(-1), np.asarray(explicit_chunk).reshape(-1)]
        )
    else:
        control_codes = batch_codes_from_one_hot(control_covariates)
        chunk_codes = batch_codes_from_one_hot(chunk_covariates)
        if control_codes is None or chunk_codes is None:
            return {}
        codes = np.concatenate([control_codes, chunk_codes])
        num_levels = int(codes.max()) + 1
        names = [f"batch_{index}" for index in range(num_levels)]
    if names is None:
        names = [f"batch_{index}" for index in range(int(np.asarray(codes).max()) + 1)]
    return {
        "categorical_batch_codes": jnp.asarray(np.asarray(codes), dtype=jnp.int32),
        "categorical_batch_names": [str(name) for name in names],
    }


def build_chunk_design(
    baseline: CRTBaseline,
    chunk_data: PerTurboData,
    *,
    control_data: PerTurboData,
    dispersion: np.ndarray | None = None,
):
    """Stack the shared control pool on top of one perturbation chunk's cells.

    Perturbation chunking hands each chunk only its own targets' cells, but the
    CRT resamples a target's label within controls-plus-that-target, so the
    control pool has to be physically present in the arrays the score kernel
    gathers from. Controls go first and carry target code ``-1``; the chunk's
    own cells follow with their local codes.

    The controls are identical for every chunk, so this is the one place their
    cost is paid per chunk. That cost is a concatenation, not a re-derivation:
    the expensive per-gene quantities come from :class:`CRTBaseline`.

    Padded chunk buffers must not reach here. Padding appends all-zero cells
    that the model excludes through ``cell_mask``, but they would enter the
    control pool as legitimate zero-count cells and dilute every null. Pass the
    unpadded chunk, which is what ``cell_mask`` on the result then reflects.
    """

    validate_offset_compatibility(control_data, chunk_data)

    control_counts = np.asarray(control_data.counts)
    chunk_counts = np.asarray(chunk_data.counts)
    if control_counts.shape[1] != chunk_counts.shape[1]:
        raise ValueError(
            "Control and chunk counts must span the same genes; got "
            f"{control_counts.shape[1]} and {chunk_counts.shape[1]}."
        )
    num_control = control_counts.shape[0]
    num_chunk = chunk_counts.shape[0]

    chunk_labels = chunk_data.pert_id
    if isinstance(chunk_labels, IndexedDesignMatrix):
        indices = np.asarray(chunk_labels.indices, dtype=np.int64)
        values = np.asarray(chunk_labels.values)
        active = (indices >= 0) & (values > 0)
        assignments = active.sum(axis=1)
        num_multi_dropped = int(np.count_nonzero(assignments > 1))
        if num_multi_dropped:
            print(
                f"[perturbo] CRT: setting aside {num_multi_dropped} of {num_chunk} chunk cells that carry "
                "more than one perturbation; the control-anchored test needs one per cell "
                "(--crt-pool all-cells keeps them)."
            )
        first = np.argmax(active, axis=1)
        codes = np.where(assignments == 1, indices[np.arange(num_chunk), first], -1)
        unassigned = assignments != 1
    elif chunk_labels.ndim == 2:
        chunk_labels = np.asarray(chunk_labels)
        assignments = np.asarray(chunk_labels > 0).sum(axis=1)
        num_multi_dropped = int(np.count_nonzero(assignments > 1))
        if num_multi_dropped:
            # The control-anchored null resamples one label per cell, so a cell with
            # two perturbations has no place in it. It is set aside, not split or
            # reassigned; the all-cells pool is the test that keeps such cells.
            print(
                f"[perturbo] CRT: setting aside {num_multi_dropped} of {num_chunk} chunk cells that carry "
                "more than one perturbation; the control-anchored test needs one per cell "
                "(--crt-pool all-cells keeps them)."
            )
        codes = np.where(assignments == 1, np.argmax(np.asarray(chunk_labels > 0), axis=1), -1)
        unassigned = assignments != 1
    elif chunk_labels.ndim == 1:
        chunk_labels = np.asarray(chunk_labels)
        codes = chunk_labels.astype(np.int64)
        unassigned = np.zeros(num_chunk, dtype=bool)
        num_multi_dropped = 0
    else:
        raise ValueError("chunk_data.pert_id must be a label vector or an assignment matrix.")

    # Codes are shifted by one so index 0 is the pooled control pool, matching
    # the name list below.
    labels = np.concatenate([np.zeros(num_control, dtype=np.int64), codes + 1])
    keep = np.concatenate([np.ones(num_control, dtype=bool), ~unassigned])
    if chunk_data.cell_mask is not None:
        chunk_mask = np.asarray(chunk_data.cell_mask, dtype=bool).reshape(-1)
        keep[num_control:] &= chunk_mask

    control_offsets = _cell_axis(control_data.size_factors, num_control, "control size_factors")
    chunk_offsets = _cell_axis(chunk_data.size_factors, num_chunk, "chunk size_factors")
    if control_offsets is None or chunk_offsets is None:
        raise ValueError(
            "Both the control and chunk data must carry size factors. The CRT conditions on a "
            "fixed offset, so it cannot fall back to a latent one."
        )
    control_covariates = _cell_axis(control_data.covariates, num_control, "control covariates")
    chunk_covariates = _cell_axis(chunk_data.covariates, num_chunk, "chunk covariates")
    if (control_covariates is None) != (chunk_covariates is None):
        raise ValueError("Control and chunk data must agree on whether covariates are present.")

    stacked = PerTurboData(
        counts=jnp.asarray(np.concatenate([control_counts, chunk_counts], axis=0), dtype=jnp.float32),
        pert_id=jnp.asarray(labels),
        pert_names=[CRT_CONTROL_NAME, *[str(name) for name in chunk_data.pert_names]],
        gene_names=list(chunk_data.gene_names),
        cell_mask=jnp.asarray(keep),
        size_factors=jnp.asarray(
            np.concatenate([control_offsets, chunk_offsets], axis=0), dtype=jnp.float32
        ),
        covariates=(
            None
            if control_covariates is None
            else jnp.asarray(
                np.concatenate([control_covariates, chunk_covariates], axis=0), dtype=jnp.float32
            )
        ),
        covariate_names=None if control_data.covariate_names is None else list(control_data.covariate_names),
        library_size_center_log_mean=control_data.library_size_center_log_mean,
        **_stacked_batch_coding(control_data, chunk_data, control_covariates, chunk_covariates),
    )
    theta = baseline.nuisance.dispersion if dispersion is None else np.asarray(dispersion)
    return prepare_low_moi_design(
        stacked,
        control_perturbations=[CRT_CONTROL_NAME],
        dispersion=np.asarray(theta, dtype=np.float32),
    )


def fit_tail_families(
    observed_score: np.ndarray,
    moments,
    families: Iterable[str] = CRT_TAIL_FAMILIES,
) -> dict[str, dict[str, np.ndarray]]:
    """Fit each named parametric null from one shared set of resampled moments.

    Returns ``{family: {"p_value", "log_p_value", "valid"}}``. The log scale is
    carried because the linear p-value floors at float64's ~1e-308 while the far
    tail is exactly where a parametric null earns its keep - rank on the log
    field, threshold on the linear one.
    """

    common = dict(
        observed_score=observed_score,
        count=moments.count,
        sum_score=moments.sum_score,
        sum_square=moments.sum_square,
        sum_cube=moments.sum_cube,
    )
    fitted: dict[str, dict[str, np.ndarray]] = {}
    for family in families:
        if family == "skew_normal":
            fit = fit_skew_normal_from_moments(**common)
        elif family == "student_t":
            fit = fit_student_t_from_moments(**common, sum_fourth=moments.sum_fourth)
        else:
            raise ValueError(f"Unknown tail family {family!r}; expected one of {list(CRT_TAIL_FAMILIES)}.")
        fitted[family] = {
            "p_value": np.asarray(fit.p_value, dtype=np.float64),
            "log_p_value": np.asarray(fit.log_p_value, dtype=np.float64),
            "valid": np.asarray(fit.valid, dtype=bool),
        }
    return fitted


def _null_moment_summaries(moments) -> dict[str, np.ndarray]:
    """Standardized null shape, as diagnostics rather than as a test.

    A tail extrapolation is only as good as the moments behind it, so the
    skewness and excess kurtosis that drove the fit travel with the result.
    """

    count = np.where(moments.count > 0, moments.count, np.nan)
    mean = moments.sum_score / count
    variance = moments.sum_square / count - mean**2
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(np.where(variance > 0, variance, np.nan))
        third = moments.sum_cube / count - 3 * mean * variance - mean**3
        fourth = (
            moments.sum_fourth / count
            - 4 * mean * (moments.sum_cube / count)
            + 6 * mean**2 * (moments.sum_square / count)
            - 3 * mean**4
        )
        return {
            "crt_null_mean": mean,
            "crt_null_variance": variance,
            "crt_null_skewness": third / sd**3,
            "crt_null_excess_kurtosis": fourth / variance**2 - 3.0,
        }


@dataclass(frozen=True)
class ChunkCRTResult:
    """CRT outputs for one perturbation chunk, over the whole gene panel."""

    observed_score: np.ndarray
    p_value: np.ndarray
    null_converged: np.ndarray
    target_names: tuple[str, ...]
    gene_names: tuple[str, ...]
    num_resamples: int
    parametric: dict[str, dict[str, np.ndarray]]
    null_summaries: dict[str, np.ndarray]
    resampling_mechanism: str = "permutation"
    saddlepoint_only: bool = False
    # Cells the control-anchored test set aside because they carried more than one
    # perturbation; zero on the all-cells pool and on label-vector inputs.
    num_multi_assignment_cells_dropped: int = 0


def _categorical_batch_applies(design) -> bool:
    """Whether the chunk design is exactly ``[intercept, one-hot batch]``.

    Only that shape is served by the categorical kernel; a continuous covariate
    beside the batch keeps the dense path. Split out so tests can force the
    dense path on a batch design and check the two agree.
    """
    return is_intercept_and_one_hot(np.asarray(design.nuisance_design))


SHARED_SELECTION_FIT_BLOCK = 65_536
"""Rows per block when projecting the screen-wide predictor back onto the design.

Only the projection is blocked, not the fit. A float64 copy of a 300,000-cell
design with a 48-level batch one-hot is 120 MB, and it is needed only to
accumulate a (q x q) Gram matrix.
"""


def fit_shared_propensity_coefficients(
    nuisance_design: np.ndarray,
    targeting: np.ndarray,
    *,
    max_iterations: int = 25,
    eta_clip: float = 30.0,
) -> np.ndarray:
    """Fit the control-anchored CRT's selection model once, over the whole screen.

    ``nuisance_design`` is the CRT's own nuisance matrix - ``[intercept,
    covariates]``, the layout :func:`prepare_low_moi_design` builds - with one
    row per control-pool cell and one per cell that will be tested;
    ``targeting`` is 1 on a cell carrying a targeting perturbation and 0 on a
    control cell. The result is the covariate part of the selection model that
    :func:`run_crt_for_chunk` then shares across every perturbation chunk,
    fitting only each target's intercept locally.

    The coefficients come back in that *original* parametrisation, one per
    column, and that choice is load-bearing. The fit itself runs in a
    rank-revealing orthonormal basis - with a full batch one-hot beside the
    intercept the design is rank deficient, so the original parametrisation has
    no unique maximiser - but each chunk builds its own basis from its own
    cells, and basis coordinates therefore mean different things in different
    chunks. Column identity is the one thing the screen and its chunks agree
    on, so that is what the coefficients are expressed in.

    Non-uniqueness is harmless here: two solutions of ``N beta = eta`` differ by
    a vector in ``N``'s null space, and a chunk's design is a row subset of
    ``N``, so both give the same linear predictor on every chunk's cells.
    """

    from perturbo._internal.high_moi.resampling import fit_propensity_coefficients

    design = np.asarray(nuisance_design, dtype=np.float32)
    response = np.asarray(targeting, dtype=np.float32).reshape(-1)
    if design.ndim != 2 or design.shape[0] != response.shape[0]:
        raise ValueError("nuisance_design and targeting must agree on the cell count.")
    if design.shape[0] == 0:
        raise ValueError("The screen-wide selection model needs at least one cell.")
    coefficients, basis = fit_propensity_coefficients(
        response[None, :],
        design,
        max_iterations=int(max_iterations),
        eta_clip=float(eta_clip),
    )
    # The unclipped predictor on purpose. Clipping is a step-halving stand-in
    # for separation, and a clipped vector need not lie in the design's column
    # space at all, which is exactly what the projection below relies on.
    eta = np.asarray(basis, dtype=np.float64) @ np.asarray(
        coefficients, dtype=np.float64
    ).reshape(-1)

    num_columns = design.shape[1]
    gram = np.zeros((num_columns, num_columns), dtype=np.float64)
    rhs = np.zeros(num_columns, dtype=np.float64)
    for start in range(0, design.shape[0], SHARED_SELECTION_FIT_BLOCK):
        stop = start + SHARED_SELECTION_FIT_BLOCK
        block = design[start:stop].astype(np.float64)
        gram += block.T @ block
        rhs += block.T @ eta[start:stop]
    beta = np.linalg.lstsq(gram, rhs, rcond=None)[0]

    residual = 0.0
    for start in range(0, design.shape[0], SHARED_SELECTION_FIT_BLOCK):
        stop = start + SHARED_SELECTION_FIT_BLOCK
        block = design[start:stop].astype(np.float64)
        residual = max(residual, float(np.max(np.abs(block @ beta - eta[start:stop]))))
    scale = max(1.0, float(np.max(np.abs(eta))))
    if residual > 1e-4 * scale:
        raise ValueError(
            "The screen-wide selection model's linear predictor could not be written in the "
            f"nuisance columns (residual {residual:.3e} against a scale of {scale:.3e}). "
            "The predictor is fitted in a basis for those columns' own span, so this should be "
            "exact; a failure points at a design that changed between the fit and the projection."
        )
    return beta


def run_crt_for_chunk(
    baseline: CRTBaseline,
    chunk_data: PerTurboData,
    *,
    control_data: PerTurboData,
    num_resamples: int = 999,
    strata: np.ndarray | None = None,
    seed: int = 0,
    gene_chunk_size: int | None = None,
    tail_families: Iterable[str] | None = CRT_TAIL_FAMILIES,
    jax_max_gather_gib: float | None = 4.0,
    jax_targets_per_batch: int = 4,
    resampling_mechanism: str = "permutation",
    saddlepoint_only: bool = False,
    saddlepoint_screen_p_value: float = 0.05,
    saddlepoint_two_sided: str = "equal-tail",
    shared_propensity_coefficients: np.ndarray | None = None,
) -> ChunkCRTResult:
    """Run the CRT for one perturbation chunk, looping inner gene chunks.

    ``resampling_mechanism`` chooses the null: ``"permutation"`` holds each
    stratum's selected count fixed; ``"propensity"`` draws each pool cell on
    its own fitted selection probability (model-X). ``tail_families`` may
    include ``"saddlepoint"``, evaluated in the kernel; under the propensity
    mechanism that is the exact Bernoulli-sum CGF, screened by a Pearson III
    tail at ``saddlepoint_screen_p_value``. ``saddlepoint_only`` draws no
    resamples at all - it requires the propensity mechanism and
    ``tail_families=("saddlepoint",)`` - and leaves the empirical p-value
    missing.

    Production chunks perturbations and keeps every gene, which is the opposite
    decomposition from the research driver. The score kernel gathers
    ``(targets, resamples, cells, genes)``, so a full transcriptome on the gene
    axis will not fit; the inner loop puts the gene axis back under control.
    Chunking genes is exact - every quantity here is independent across genes -
    so this changes cost, not answers.

    ``shared_propensity_coefficients`` are the screen-wide logistic selection
    coefficients, one per column of the chunk design's nuisance matrix and in
    that order. Passing them keeps the covariate part of the selection model
    fixed across chunks and fits only each target's intercept here, so a
    target's p-value does not move when the chunk-size flag changes its
    neighbours. Leaving them ``None`` fits a separate model per target on that
    target's own pool.

    ``strata`` aligns with ``chunk_data``'s cells. Control cells are given their
    own stratum values from ``control_data`` implicitly: they are prepended, so
    the caller supplies control strata via ``control_data`` ordering. Passing
    ``None`` runs an unstratified CRT.

    No q-values are produced. Benjamini-Hochberg has to see every hypothesis at
    once, and a chunk is by construction only part of the family, so the
    correction belongs to whoever concatenates the chunks.
    """

    # Counted here, on the chunk as it arrives, so the result can report how many

    # cells the control-anchored test set aside; the design builder prints it.

    _labels = chunk_data.pert_id
    if isinstance(_labels, IndexedDesignMatrix):
        _active = (np.asarray(_labels.indices) >= 0) & (np.asarray(_labels.values) > 0)
        num_multi_dropped = int(np.count_nonzero(_active.sum(axis=1) > 1))
    else:
        _labels = np.asarray(_labels)
        num_multi_dropped = (
            int(np.count_nonzero(np.asarray(_labels > 0).sum(axis=1) > 1))
            if _labels.ndim == 2
            else 0
        )

    design = build_chunk_design(baseline, chunk_data, control_data=control_data)
    categorical_batch = design.batch_codes is not None and _categorical_batch_applies(design)
    if design.batch_codes is not None and not categorical_batch:
        # A continuous covariate sits beside the batch, which the categorical
        # kernel (one intercept per batch, nothing else) cannot represent:
        # stay on the dense design.
        design = dataclasses.replace(design, batch_codes=None, batch_names=None)
    num_control = int(np.asarray(control_data.counts).shape[0])
    full_strata = None
    if categorical_batch and strata is None:
        # The categorical kernel's closed-form null holds only when the CRT
        # strata are the batch levels (each batch a draw of fixed size), so a
        # batch covariate stratifies the test by batch, as the research driver
        # does on Replogle essential. The dense path has no stratified
        # closed form and stays unstratified with the batch in the logits.
        full_strata = np.asarray(design.batch_codes).reshape(-1)
    if strata is not None:
        chunk_strata = np.asarray(strata).reshape(-1)
        if chunk_strata.shape[0] == num_control + int(np.asarray(chunk_data.counts).shape[0]):
            stacked_strata = chunk_strata
        elif chunk_strata.shape[0] == int(np.asarray(chunk_data.counts).shape[0]):
            raise ValueError(
                "strata must cover the control cells as well as the chunk cells, in that order; "
                "the CRT resamples a target's label against the control pool, so the pool needs "
                "stratum values too."
            )
        else:
            raise ValueError("strata must contain one value per stacked (control + chunk) cell.")
        full_strata = stacked_strata[np.asarray(design.source_cell_indices)]

    num_genes = design.num_genes
    width = num_genes if gene_chunk_size is None else int(gene_chunk_size)
    num_targets = design.num_targets

    families = tuple(tail_families or ())
    unknown = [family for family in families if family not in CRT_ALL_TAIL_FAMILIES]
    if unknown:
        raise ValueError(f"Unknown tail families {unknown}; expected from {list(CRT_ALL_TAIL_FAMILIES)}.")
    if resampling_mechanism not in CRT_MECHANISMS:
        raise ValueError(f"resampling_mechanism must be one of {list(CRT_MECHANISMS)}.")
    want_saddlepoint = CRT_SADDLEPOINT_FAMILY in families
    moment_families = tuple(family for family in families if family != CRT_SADDLEPOINT_FAMILY)
    if saddlepoint_only:
        if families != (CRT_SADDLEPOINT_FAMILY,):
            raise ValueError("saddlepoint_only requires tail_families=('saddlepoint',) alone.")
        if resampling_mechanism != "propensity":
            raise ValueError("saddlepoint_only requires resampling_mechanism='propensity'.")
    if not 0.0 < saddlepoint_screen_p_value <= 1.0:
        raise ValueError("saddlepoint_screen_p_value must lie in (0, 1].")

    shape = (num_targets, num_genes)
    observed = np.full(shape, np.nan, dtype=np.float64)
    p_values = np.full(shape, np.nan, dtype=np.float64)
    converged = np.zeros(shape, dtype=bool)
    parametric = {
        family: {key: np.full(shape, np.nan, dtype=np.float64) for key in ("p_value", "log_p_value")}
        | {"valid": np.zeros(shape, dtype=bool)}
        for family in families
    }
    if want_saddlepoint:
        parametric[CRT_SADDLEPOINT_FAMILY]["used_screen"] = np.zeros(shape, dtype=bool)
    null_summaries = {
        name: np.full(shape, np.nan, dtype=np.float64)
        for name in ("crt_null_mean", "crt_null_variance", "crt_null_skewness", "crt_null_excess_kurtosis")
    }

    # Resamples never depend on genes, so they are drawn once for the whole
    # chunk rather than redrawn per gene slice.
    permutations = precompute_low_moi_permutations(
        design,
        num_resamples=num_resamples,
        strata=full_strata,
        seed=seed,
        resampling_mechanism=resampling_mechanism,
        draw_resamples=not saddlepoint_only,
        shared_propensity_coefficients=shared_propensity_coefficients,
    )

    counts = np.asarray(design.counts)
    dispersion = np.asarray(design.dispersion)
    gene_names = tuple(design.gene_names)
    coefficients = baseline.nuisance.coefficients

    for gene_slice in iter_gene_chunks(num_genes, width):
        sliced = replace_gene_axis(
            design,
            counts=jnp.asarray(counts[:, gene_slice]),
            dispersion=jnp.asarray(dispersion[gene_slice]),
            gene_names=gene_names[gene_slice],
        )
        result = run_low_moi_score_permutations(
            sliced,
            num_resamples=num_resamples,
            strata=full_strata,
            seed=seed,
            backend="jax",
            null_model="control_only",
            permutations=permutations,
            jax_max_gather_gib=jax_max_gather_gib,
            jax_targets_per_batch=jax_targets_per_batch,
            # The categorical kernel parametrizes the nuisance as one intercept
            # per batch and refuses coefficients on a dense design. The two are
            # the same column space, and stage one's coefficients have been
            # polished onto the control null mode at stage one's dispersion, so
            # the kernel's own per-batch fit at that dispersion lands on the
            # same mode: letting it refit reproduces the polished baseline
            # while keeping Z'WZ as segment sums instead of a dense
            # (genes, K, K) tensor. Measured on Replogle essential (47 gem
            # groups): 85 minutes on the dense design.
            nuisance_coefficients=None if categorical_batch else coefficients[:, gene_slice],
            return_null_moments=bool(moment_families),
            tail_approximation=CRT_SADDLEPOINT_FAMILY if want_saddlepoint else None,
            saddlepoint_only=saddlepoint_only,
            saddlepoint_screen_p_value=saddlepoint_screen_p_value,
            saddlepoint_two_sided=saddlepoint_two_sided,
        )
        block_observed = np.asarray(result.observed_score, dtype=np.float64)
        observed[:, gene_slice] = block_observed
        p_values[:, gene_slice] = np.asarray(result.p_value, dtype=np.float64)
        converged[:, gene_slice] = np.asarray(result.null_converged)
        if moment_families:
            # One resampling pass feeds every family; each fit is a cheap
            # host-side pass over moments that are already reduced.
            fitted = fit_tail_families(block_observed, result.null_moments, moment_families)
            for family, columns in fitted.items():
                for key, values in columns.items():
                    parametric[family][key][:, gene_slice] = values
            for name, values in _null_moment_summaries(result.null_moments).items():
                null_summaries[name][:, gene_slice] = values
        if want_saddlepoint:
            # The kernel evaluated the saddlepoint itself. It is the only family
            # the kernel was asked for, so its screen flag is the primary
            # fallback field.
            block = parametric[CRT_SADDLEPOINT_FAMILY]
            fitted_saddlepoint = (result.tail_fits or {}).get(CRT_SADDLEPOINT_FAMILY)
            if fitted_saddlepoint is None:
                raise RuntimeError("The score kernel did not return a saddlepoint fit.")
            block["p_value"][:, gene_slice] = np.asarray(fitted_saddlepoint["p_value"], dtype=np.float64)
            block["log_p_value"][:, gene_slice] = np.asarray(fitted_saddlepoint["log_p_value"], dtype=np.float64)
            block["valid"][:, gene_slice] = np.asarray(fitted_saddlepoint["valid"], dtype=bool)
            if result.parametric_used_fallback is not None:
                block["used_screen"][:, gene_slice] = np.asarray(result.parametric_used_fallback, dtype=bool)

    return ChunkCRTResult(
        observed_score=observed,
        p_value=p_values,
        null_converged=converged,
        target_names=tuple(design.target_names),
        gene_names=gene_names,
        num_resamples=num_resamples,
        parametric=parametric,
        null_summaries=null_summaries,
        num_multi_assignment_cells_dropped=num_multi_dropped,
        resampling_mechanism=resampling_mechanism,
        saddlepoint_only=saddlepoint_only,
    )


CRT_POOLS = ("control-anchored", "all-cells")
"""Which cells a target is resampled against.

``control-anchored`` (low MOI): the null is fit on control cells and each target
is tested inside controls plus its own cells; it needs at most one perturbation
per analysed cell and a pool of unperturbed controls. ``all-cells`` (high MOI):
the null is fit on every analysed cell with every element omitted and each
element is tested as a marginal association over all cells; it needs the
guide-to-element structure and nothing else. Nothing branches on a measured
MOI; see ``docs/research/12_one_crt_two_designs.md``.
"""


def _element_membership(data: PerTurboData) -> tuple[np.ndarray, np.ndarray, int]:
    """COO ``(cell_index, element_index)`` of cells carrying each element, sorted by element.

    Built from the non-zero entries of the guide matrix joined to the
    guide-to-element map, never from a dense cells-by-elements product: NumPy
    has no BLAS path for integer or boolean matmul, and the dense product over
    the 207k-cell, 13k-guide at-scale screen ran single-threaded for hours.
    """
    if data.guide_matrix is None or data.guide_to_element is None:
        raise ValueError(
            "The all-cells CRT needs the guide-to-element structure (guide_matrix and guide_to_element); "
            "load the data with the perturbation modality and its element map."
        )
    guide_to_element = np.asarray(data.guide_to_element) > 0
    num_guides, num_elements = guide_to_element.shape
    guide_matrix = data.guide_matrix
    num_cells = int(guide_matrix.shape[0])
    if int(guide_matrix.shape[1]) != int(num_guides):
        raise ValueError(
            f"guide_matrix has {guide_matrix.shape[1]} guides but guide_to_element maps {num_guides}."
        )
    if isinstance(guide_matrix, IndexedDesignMatrix):
        indices = np.asarray(guide_matrix.indices, dtype=np.int64)
        values = np.asarray(guide_matrix.values)
        active = (indices >= 0) & (values > 0)
        cells = np.broadcast_to(np.arange(num_cells)[:, None], indices.shape)[active]
        guides = indices[active]
    else:
        cells, guides = np.nonzero(np.asarray(guide_matrix) > 0)
    # Expand each detected (cell, guide) to the guide's elements. The map's
    # non-zeros come out sorted by guide, so a guide's elements are one
    # contiguous run starting at offsets[guide].
    map_guides, map_elements = np.nonzero(guide_to_element)
    per_guide = np.bincount(map_guides, minlength=num_guides)
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(per_guide, dtype=np.int64)])
    reps = per_guide[guides]
    total = int(reps.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), int(num_elements)
    expanded_cells = np.repeat(cells.astype(np.int64), reps)
    run_start = np.repeat(np.cumsum(reps, dtype=np.int64) - reps, reps)
    within = np.arange(total, dtype=np.int64) - run_start
    expanded_elements = map_elements[np.repeat(offsets[guides], reps) + within].astype(np.int64)
    # One row per distinct (element, cell), element-major then by cell: the
    # same order the dense ``np.nonzero(membership.T)`` produced.
    keys = np.unique(expanded_elements * num_cells + expanded_cells)
    element_index = keys // num_cells
    cell_index = keys - element_index * num_cells
    return cell_index.astype(np.int64), element_index.astype(np.int64), int(num_elements)


@dataclass(frozen=True)
class AllCellsPropensityFit:
    """Gene-independent state reused across gene blocks of an all-cells CRT."""

    cell_index: np.ndarray
    element_index: np.ndarray
    element_names: tuple[str, ...]
    testable: np.ndarray
    coefficients: np.ndarray
    basis: np.ndarray
    num_cells: int


def prepare_all_cells_propensity(
    baseline: CRTBaseline,
    data: PerTurboData,
    *,
    min_cells_per_element: int = 1,
    include_guide_count: bool = True,
    max_iterations: int = 25,
    eta_clip: float = 30.0,
    element_batch_size: int = 64,
) -> AllCellsPropensityFit:
    """Fit the all-cells assignment models once for reuse over gene blocks."""

    from perturbo._internal.high_moi.resampling import (
        fit_propensity_coefficients,
        propensity_basis,
    )

    num_cells = int(np.asarray(data.counts).shape[0])
    if baseline.num_control_cells != num_cells:
        raise ValueError(
            "The all-cells CRT baseline must be prepared on the analysed cells themselves "
            f"(baseline holds {baseline.num_control_cells} cells, data holds {num_cells})."
        )
    cell_index, element_index, num_elements = _element_membership(data)
    element_names = tuple(str(name) for name in data.pert_names)
    if len(element_names) != num_elements:
        raise ValueError(
            f"pert_names has {len(element_names)} entries but guide_to_element maps to {num_elements} elements."
        )
    sizes = np.bincount(element_index, minlength=num_elements)
    testable = sizes >= int(min_cells_per_element)
    keep_pair = testable[element_index]
    cell_index, element_index = cell_index[keep_pair], element_index[keep_pair]

    nuisance_design = np.asarray(baseline.nuisance.nuisance_design, dtype=np.float32)
    propensity_design = nuisance_design
    if include_guide_count:
        if isinstance(data.guide_matrix, IndexedDesignMatrix):
            detected = np.sum(
                (np.asarray(data.guide_matrix.indices) >= 0)
                & (np.asarray(data.guide_matrix.values) > 0),
                axis=1,
                dtype=np.float64,
            )
        else:
            detected = np.asarray(np.asarray(data.guide_matrix) > 0).sum(axis=1).astype(np.float64)
        moi = np.log1p(detected)
        moi = (moi - moi.mean()) / max(float(moi.std()), 1e-8)
        propensity_design = np.concatenate(
            [nuisance_design, moi.astype(np.float32)[:, None]], axis=1
        )

    coefficients_parts = []
    basis = None
    order = np.argsort(element_index, kind="stable")
    sorted_elements = element_index[order]
    sorted_cells = cell_index[order]
    starts = np.searchsorted(sorted_elements, np.arange(num_elements + 1))
    for start in range(0, num_elements, int(element_batch_size)):
        stop = min(start + int(element_batch_size), num_elements)
        indicators = np.zeros((stop - start, num_cells), dtype=np.float32)
        for local, element in enumerate(range(start, stop)):
            indicators[local, sorted_cells[starts[element] : starts[element + 1]]] = 1.0
        coef, basis_batch = fit_propensity_coefficients(
            indicators,
            propensity_design,
            max_iterations=int(max_iterations),
            eta_clip=float(eta_clip),
            basis=basis,
        )
        coefficients_parts.append(np.asarray(coef))
        if basis is None:
            basis = np.asarray(basis_batch)
    if basis is None:
        basis = np.asarray(propensity_basis(propensity_design))
    coefficients = (
        np.concatenate(coefficients_parts, axis=0)
        if coefficients_parts
        else np.zeros((0, basis.shape[1]), dtype=np.float32)
    )
    return AllCellsPropensityFit(
        cell_index=cell_index,
        element_index=element_index,
        element_names=element_names,
        testable=testable,
        coefficients=coefficients,
        basis=basis,
        num_cells=num_cells,
    )


def run_crt_all_cells(
    baseline: CRTBaseline,
    data: PerTurboData,
    *,
    gene_chunk_size: int | None = 500,
    screen_p_value: float = 0.05,
    two_sided: str = "equal-tail",
    gene_block_size: int = 64,
    element_batch_size: int = 64,
    min_cells_per_element: int = 1,
    include_guide_count_in_propensity: bool = True,
    propensity_max_iterations: int = 25,
    eta_clip: float = 30.0,
    propensity_fit: AllCellsPropensityFit | None = None,
) -> ChunkCRTResult:
    """The propensity saddlepoint CRT with every analysed cell as the pool (high MOI).

    ``baseline`` must have been prepared on ``data`` itself
    (:func:`prepare_crt_baseline` with the analysed cells as ``control_data``):
    the null is the stage-one fit polished onto the mode over all cells with
    every element omitted, so each cell's residual is in sample and no pool
    projection is needed. Each element's selection model is a logistic
    regression of its indicator on the nuisance design, plus the standardized
    log1p count of detected guides when ``include_guide_count_in_propensity``
    (a cell that carries more guides is more likely to carry any given one; the
    outcome model need not know this, the selection model must). The statistic
    and tail are the research kernel's exact Bernoulli-sum saddlepoint, so the
    result matches ``analysis/gasperini_high_moi.py --resampling-mechanism
    propensity --tail-approximation saddlepoint --saddlepoint-only`` at the same
    dispersion.

    No resamples are drawn; the empirical p-value is left missing. Elements
    with fewer than ``min_cells_per_element`` cells are left NaN. Q-values are
    the caller's, as for the chunked low-MOI path.
    """

    from perturbo._internal.saddlepoint import fit_high_moi_propensity_saddlepoint

    num_cells = int(np.asarray(data.counts).shape[0])
    if baseline.num_control_cells != num_cells:
        raise ValueError(
            "The all-cells CRT baseline must be prepared on the analysed cells themselves "
            f"(baseline holds {baseline.num_control_cells} cells, data holds {num_cells})."
        )
    if tuple(baseline.gene_names) != tuple(str(g) for g in data.gene_names):
        raise ValueError("Baseline and data disagree on the gene axis.")
    if not 0.0 < screen_p_value <= 1.0:
        raise ValueError("screen_p_value must lie in (0, 1].")

    nuisance_design = np.asarray(baseline.nuisance.nuisance_design, dtype=np.float32)
    if propensity_fit is None:
        propensity_fit = prepare_all_cells_propensity(
            baseline,
            data,
            min_cells_per_element=int(min_cells_per_element),
            include_guide_count=bool(include_guide_count_in_propensity),
            max_iterations=int(propensity_max_iterations),
            eta_clip=float(eta_clip),
            element_batch_size=int(element_batch_size),
        )
    element_names = tuple(str(name) for name in data.pert_names)
    if propensity_fit.num_cells != num_cells or propensity_fit.element_names != element_names:
        raise ValueError("propensity_fit does not match the all-cells CRT data.")
    cell_index = propensity_fit.cell_index
    element_index = propensity_fit.element_index
    testable = propensity_fit.testable
    coefficients = propensity_fit.coefficients
    basis = propensity_fit.basis
    num_elements = len(element_names)

    num_genes = baseline.num_genes
    width = num_genes if gene_chunk_size is None else int(gene_chunk_size)
    shape = (num_elements, num_genes)
    p_value = np.full(shape, np.nan)
    log_p = np.full(shape, np.nan)
    valid = np.zeros(shape, dtype=bool)
    used_screen = np.zeros(shape, dtype=bool)
    observed = np.full(shape, np.nan)
    null_mean = np.full(shape, np.nan)
    null_variance = np.full(shape, np.nan)
    null_skewness = np.full(shape, np.nan)
    for start in range(0, num_genes, width):
        gene_slice = slice(start, min(start + width, num_genes))
        block = control_block_for_genes(baseline, gene_slice)
        information = np.asarray(block.information, dtype=np.float64)
        inverse = np.linalg.inv(information)
        fit = fit_high_moi_propensity_saddlepoint(
            score_residual=np.asarray(block.score_residual),
            observation_weight=np.asarray(block.observation_weight),
            nuisance_design=nuisance_design,
            nuisance_information_inverse=inverse,
            nuisance_score=np.asarray(block.nuisance_score),
            cell_index=cell_index,
            element_index=element_index,
            num_elements=num_elements,
            propensity_coefficients=coefficients,
            propensity_basis=basis,
            eta_clip=float(eta_clip),
            screen_p_value=float(screen_p_value),
            two_sided=two_sided,
            gene_block_size=int(gene_block_size),
            element_batch_size=int(element_batch_size),
        )
        p_value[:, gene_slice] = fit.p_value
        log_p[:, gene_slice] = fit.log_p_value
        valid[:, gene_slice] = fit.valid
        used_screen[:, gene_slice] = fit.used_fallback
        observed[:, gene_slice] = fit.observed_sum
        null_mean[:, gene_slice] = fit.null_mean
        null_variance[:, gene_slice] = fit.null_variance
        null_skewness[:, gene_slice] = fit.null_skewness
    untested = ~testable
    for array in (p_value, log_p, observed, null_mean, null_variance, null_skewness):
        array[untested] = np.nan
    valid[untested] = False
    used_screen[untested] = False
    # The reported score is standardized by the null's spread, as the chunked
    # path and the research driver report it (the CRT z-value column).
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized = (observed - null_mean) / np.sqrt(null_variance)
    return ChunkCRTResult(
        observed_score=standardized,
        p_value=np.full(shape, np.nan),
        null_converged=valid,
        target_names=element_names,
        gene_names=tuple(baseline.gene_names),
        num_resamples=0,
        parametric={
            CRT_SADDLEPOINT_FAMILY: {
                "p_value": p_value,
                "log_p_value": log_p,
                "valid": valid,
                "used_screen": used_screen,
            }
        },
        null_summaries={
            "crt_null_mean": null_mean,
            "crt_null_variance": null_variance,
            "crt_null_skewness": null_skewness,
            "crt_null_excess_kurtosis": np.full(shape, np.nan),
        },
        resampling_mechanism="propensity",
        saddlepoint_only=True,
    )


def exclude_targets(chunk_data: PerTurboData, drop_names: Iterable[str]) -> PerTurboData | None:
    """Drop named perturbations, and their cells, from a chunk.

    Used to keep control perturbations out of the CRT's target list. Their cells
    are already the control pool, and stacking them again would enter the same
    biological cells twice as distinct rows - inflating the pooled information
    and the resampling pool. The resulting test would still be *valid*, since
    the null is conditional on the residuals either way, but it would answer a
    question about a doubled design that nobody asked.

    Returns ``None`` when nothing is left to test.
    """

    drop = {str(name) for name in drop_names}
    names = [str(name) for name in chunk_data.pert_names]
    keep_targets = np.asarray([name not in drop for name in names], dtype=bool)
    if not keep_targets.any():
        return None
    if keep_targets.all():
        return chunk_data

    labels = chunk_data.pert_id
    old_to_new = np.full(len(names), -1, dtype=np.int64)
    old_to_new[np.flatnonzero(keep_targets)] = np.arange(int(keep_targets.sum()))
    if isinstance(labels, IndexedDesignMatrix):
        indices = np.asarray(labels.indices, dtype=np.int64)
        values = np.asarray(labels.values)
        safe = np.maximum(indices, 0)
        mapped = old_to_new[safe]
        active = (indices >= 0) & (values != 0) & (mapped >= 0)
        keep_cells = active.any(axis=1)
        kept_active = active[keep_cells]
        kept_mapped = mapped[keep_cells]
        kept_values = values[keep_cells]
        widths = kept_active.sum(axis=1)
        width = max(1, int(widths.max(initial=0)))
        new_indices = np.full((int(keep_cells.sum()), width), -1, dtype=np.int32)
        new_values = np.zeros((int(keep_cells.sum()), width), dtype=np.float32)
        for row in range(new_indices.shape[0]):
            row_active = kept_active[row]
            count = int(row_active.sum())
            if count:
                new_indices[row, :count] = kept_mapped[row, row_active]
                new_values[row, :count] = kept_values[row, row_active]
        new_labels = IndexedDesignMatrix(
            indices=jnp.asarray(new_indices),
            values=jnp.asarray(new_values),
            num_columns=int(keep_targets.sum()),
        )
    elif labels.ndim == 1:
        labels_array = np.asarray(labels)
        keep_cells = keep_targets[labels_array.astype(np.int64)]
        new_labels = old_to_new[labels_array.astype(np.int64)[keep_cells]]
    elif labels.ndim == 2:
        binary = np.asarray(labels > 0)
        keep_cells = binary[:, keep_targets].any(axis=1)
        new_labels = binary[keep_cells][:, keep_targets]
    else:
        raise ValueError("chunk_data.pert_id must be a label vector or an assignment matrix.")
    if not keep_cells.any():
        return None

    def _rows(values):
        if values is None:
            return None
        if isinstance(values, IndexedDesignMatrix):
            return values.take_rows(keep_cells)
        return jnp.asarray(np.asarray(values)[keep_cells])

    return PerTurboData(
        counts=jnp.asarray(np.asarray(chunk_data.counts)[keep_cells]),
        pert_id=(new_labels if isinstance(new_labels, IndexedDesignMatrix) else jnp.asarray(new_labels)),
        pert_names=[name for name, keep in zip(names, keep_targets, strict=True) if keep],
        gene_names=list(chunk_data.gene_names),
        cell_mask=_rows(chunk_data.cell_mask),
        size_factors=_rows(chunk_data.size_factors),
        covariates=_rows(chunk_data.covariates),
        covariate_names=None if chunk_data.covariate_names is None else list(chunk_data.covariate_names),
        guide_matrix=_rows(chunk_data.guide_matrix),
        guide_names=None if chunk_data.guide_names is None else list(chunk_data.guide_names),
        guide_to_element=chunk_data.guide_to_element,
        library_size_center_log_mean=chunk_data.library_size_center_log_mean,
    )


@dataclass
class CRTAccumulator:
    """Collects chunk results into one screen-wide table.

    Chunks are absorbed *by element name* rather than by position. Positional
    scatter would work, but it silently depends on the caller threading the same
    index arrays through two loops, and the whole point of the chunk-invariance
    work is that nothing downstream should care how the run was decomposed.

    Benjamini-Hochberg is deliberately not applied per chunk. The correction has
    to see every hypothesis at once, and a chunk is by construction only part of
    the family, so it happens in :meth:`finalize`.
    """

    element_names: tuple[str, ...]
    gene_names: tuple[str, ...]
    tail_families: tuple[str, ...] = CRT_TAIL_FAMILIES
    saddlepoint_only: bool = False

    def __post_init__(self) -> None:
        shape = (len(self.element_names), len(self.gene_names))
        self._index = {name: position for position, name in enumerate(self.element_names)}
        self._gene_index = {name: position for position, name in enumerate(self.gene_names)}
        self._dropped_by_target_chunk: dict[tuple[str, ...], int] = {}
        self.num_multi_assignment_cells_dropped = 0
        self.observed_score = np.full(shape, np.nan, dtype=np.float64)
        self.p_value = None if self.saddlepoint_only else np.full(shape, np.nan, dtype=np.float64)
        self.null_converged = np.zeros(shape, dtype=bool)
        self.tested = np.zeros(len(self.element_names), dtype=bool)
        self.parametric = {
            family: {key: np.full(shape, np.nan, dtype=np.float64) for key in ("p_value", "log_p_value")}
            | {"valid": np.zeros(shape, dtype=bool)}
            for family in self.tail_families
        }
        if CRT_SADDLEPOINT_FAMILY in self.parametric:
            self.parametric[CRT_SADDLEPOINT_FAMILY]["used_screen"] = np.zeros(shape, dtype=bool)
        summary_names = ["crt_null_mean", "crt_null_variance", "crt_null_skewness"]
        if not self.saddlepoint_only:
            summary_names.append("crt_null_excess_kurtosis")
        self.null_summaries = {name: np.full(shape, np.nan, dtype=np.float64) for name in summary_names}

    def absorb(self, result: ChunkCRTResult) -> None:
        unknown = [name for name in result.target_names if name not in self._index]
        if unknown:
            raise ValueError(f"Chunk reported elements absent from the screen: {unknown[:5]}")
        unknown_genes = [name for name in result.gene_names if name not in self._gene_index]
        if unknown_genes:
            raise ValueError(
                f"Chunk gene names do not match the accumulator: absent from the screen {unknown_genes[:5]}"
            )
        rows = np.asarray([self._index[name] for name in result.target_names])
        columns = np.asarray([self._gene_index[name] for name in result.gene_names])
        destination = np.ix_(rows, columns)
        target_chunk = tuple(result.target_names)
        self._dropped_by_target_chunk[target_chunk] = max(
            self._dropped_by_target_chunk.get(target_chunk, 0),
            int(getattr(result, "num_multi_assignment_cells_dropped", 0)),
        )
        self.num_multi_assignment_cells_dropped = sum(self._dropped_by_target_chunk.values())
        self.observed_score[destination] = result.observed_score
        if self.p_value is not None:
            self.p_value[destination] = result.p_value
        self.null_converged[destination] = result.null_converged
        self.tested[rows] = True
        for family, columns in result.parametric.items():
            if family not in self.parametric:
                continue
            for key, values in columns.items():
                if key in self.parametric[family]:
                    self.parametric[family][key][destination] = values
        for name, values in result.null_summaries.items():
            if name in self.null_summaries:
                self.null_summaries[name][destination] = values

    def finalize(self, *, streaming: bool = False) -> dict[str, np.ndarray]:
        """Screen-wide columns, with Benjamini-Hochberg over every tested pair.

        Each parametric family gets its own q-value, corrected independently
        over the whole screen. They are alternative nulls for the same
        statistic, not a multiple-testing family among themselves, so pooling
        them would be wrong.
        """

        shape = (len(self.element_names), len(self.gene_names))
        missing = np.broadcast_to(np.asarray(np.nan, dtype=np.float64), shape)
        empirical_p = missing if self.p_value is None else self.p_value
        columns = {
            "crt_z_value": self.observed_score,
            "crt_p_value": empirical_p,
            "crt_q_value": missing if self.p_value is None else _benjamini_hochberg(self.p_value),
        }
        for family, fitted in self.parametric.items():
            columns[f"crt_{family}_p_value"] = fitted["p_value"]
            columns[f"crt_{family}_log_p_value"] = fitted["log_p_value"]
            columns[f"crt_{family}_q_value"] = _benjamini_hochberg(fitted["p_value"])
            columns[f"crt_{family}_valid"] = (
                fitted["valid"] if streaming else fitted["valid"].astype(np.float64)
            )
            for key, values in fitted.items():
                if key not in ("p_value", "log_p_value", "valid"):
                    columns[f"crt_{family}_{key}"] = (
                        values if streaming else np.asarray(values, dtype=np.float64)
                    )
        columns.update(self.null_summaries)
        columns.setdefault("crt_null_excess_kurtosis", missing)
        return columns
