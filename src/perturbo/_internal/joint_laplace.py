"""Deterministic negative-binomial Laplace inference.

This module is an opt-in foundation for the control-anchored Laplace/CRT
research plan.  The first implementation fits a conventional joint NB GLM with
fixed dispersion.  It is intentionally smaller than the NumPyro model surface:
the purpose is to validate data design, deterministic full-data optimization,
and marginal-curvature calculations before adding modular baseline inference.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from scipy import optimize

from perturbo.core import PerTurboData
from perturbo.utils import compute_size_factors


@dataclass(frozen=True)
class JointNBDesign:
    """Explicit arrays consumed by the experimental NB Laplace backend."""

    counts: jnp.ndarray
    target_design: jnp.ndarray
    nuisance_design: jnp.ndarray
    offsets: jnp.ndarray
    dispersion: jnp.ndarray
    target_names: tuple[str, ...]
    nuisance_names: tuple[str, ...]
    gene_names: tuple[str, ...]
    control_mask: jnp.ndarray
    source_cell_indices: jnp.ndarray

    @property
    def num_cells(self) -> int:
        return int(self.counts.shape[0])

    @property
    def num_genes(self) -> int:
        return int(self.counts.shape[1])

    @property
    def num_targets(self) -> int:
        return int(self.target_design.shape[1])


@dataclass(frozen=True)
class JointLaplaceFit:
    """Marginal summaries and optimization diagnostics for a joint NB fit."""

    posterior_mean: jnp.ndarray
    posterior_scale: jnp.ndarray
    z_values: jnp.ndarray
    nuisance_mean: jnp.ndarray
    converged: jnp.ndarray
    optimizer_iterations: jnp.ndarray
    objective: jnp.ndarray
    target_names: tuple[str, ...]
    nuisance_names: tuple[str, ...]
    gene_names: tuple[str, ...]
    effect_prior_scale: float | None
    covariance_approximation: str
    max_abs_projected_effect_correlation: jnp.ndarray
    median_abs_projected_effect_correlation: jnp.ndarray
    num_projected_effect_correlation_pairs: jnp.ndarray


@dataclass(frozen=True)
class ProjectedDiagonalVariance:
    """Marginal-variance approximation after projecting out nuisance terms."""

    variance: np.ndarray
    projected_curvature: np.ndarray
    max_abs_projected_correlation: float
    median_abs_projected_correlation: float
    num_correlation_pairs: int


def _resolve_control_indices(
    perturbation_names: list[str],
    control_perturbations: Iterable[str | int],
) -> tuple[int, ...]:
    requested = tuple(control_perturbations)
    if not requested:
        raise ValueError("control_perturbations must contain at least one name or index.")
    name_to_index = {str(name): idx for idx, name in enumerate(perturbation_names)}
    indices: list[int] = []
    for value in requested:
        if isinstance(value, str):
            if value not in name_to_index:
                raise KeyError(f"Control perturbation {value!r} was not found in data.pert_names.")
            index = name_to_index[value]
        else:
            index = int(value)
            if index < 0 or index >= len(perturbation_names):
                raise IndexError(
                    f"Control perturbation index {index} is outside [0, {len(perturbation_names)})."
                )
        if index not in indices:
            indices.append(index)
    return tuple(indices)


def _active_cell_mask(data: PerTurboData) -> np.ndarray:
    n_cells = int(data.counts.shape[0])
    if data.cell_mask is None:
        return np.ones(n_cells, dtype=bool)
    mask = np.asarray(data.cell_mask, dtype=bool).reshape(-1)
    if mask.shape != (n_cells,):
        raise ValueError(
            f"data.cell_mask must contain one value per cell; got {mask.shape} for {n_cells} cells."
        )
    return mask


def _reference_coded_target_design(
    data: PerTurboData,
    *,
    control_indices: tuple[int, ...],
    allow_high_moi: bool,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    perturbation_names = list(data.pert_names)
    target_indices = tuple(idx for idx in range(len(perturbation_names)) if idx not in control_indices)
    if not target_indices:
        raise ValueError("At least one non-control perturbation is required.")

    perturbations = np.asarray(data.pert_id)
    if perturbations.ndim == 1:
        if not np.all(np.isfinite(perturbations)):
            raise ValueError("One-dimensional perturbation labels must be finite.")
        labels = perturbations.astype(np.int64, copy=False)
        if not np.array_equal(labels, perturbations):
            raise ValueError("One-dimensional perturbation labels must be integer category indices.")
        if np.any(labels < 0) or np.any(labels >= len(perturbation_names)):
            raise ValueError("One-dimensional perturbation labels reference an unknown category.")
        design = np.column_stack([labels == idx for idx in target_indices]).astype(np.float32)
        control_mask = np.isin(labels, np.asarray(control_indices, dtype=np.int64))
        assigned_mask = np.ones(labels.shape[0], dtype=bool)
    elif perturbations.ndim == 2:
        if perturbations.shape[1] != len(perturbation_names):
            raise ValueError(
                "Two-dimensional data.pert_id must have one column per perturbation name; "
                f"got {perturbations.shape[1]} columns and {len(perturbation_names)} names."
            )
        if np.any(~np.isfinite(perturbations)) or np.any(perturbations < 0):
            raise ValueError("Two-dimensional perturbation assignments must be finite and non-negative.")
        binary = np.asarray(perturbations > 0, dtype=np.float32)
        if not allow_high_moi and np.any(binary.sum(axis=1) > 1):
            raise ValueError(
                "The initial joint Laplace backend supports low-MOI assignments only "
                "(at most one active perturbation per cell); set allow_high_moi=True "
                "to use the joint high-MOI covariance approximation."
            )
        design = binary[:, target_indices]
        has_control = binary[:, control_indices].sum(axis=1) > 0
        control_mask = has_control & (design.sum(axis=1) == 0)
        assigned_mask = binary.sum(axis=1) > 0
    else:
        raise ValueError("data.pert_id must be a one-dimensional label vector or two-dimensional matrix.")

    target_names = tuple(perturbation_names[idx] for idx in target_indices)
    return design, control_mask, target_names, assigned_mask


def prepare_joint_nb_design(
    data: PerTurboData,
    *,
    control_perturbations: Iterable[str | int],
    dispersion: np.ndarray | jnp.ndarray,
    use_observed_size_factors: bool = True,
    allow_high_moi: bool = False,
    require_control_cells: bool = True,
) -> JointNBDesign:
    """Convert existing ``PerTurboData`` into a reference-coded NB design.

    Controls are pooled into the all-zero reference row of ``target_design``.
    Cells without any assignment in a matrix-based input are excluded rather
    than silently treated as controls. High-MOI input must be explicitly
    enabled; cells carrying controls and targets are targeting rows, not pure
    controls. ``require_control_cells=False`` is intended only for joint fits,
    never for the control-anchored or score-resampling backends.
    """

    counts = np.asarray(data.counts)
    if counts.ndim != 2 or np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("data.counts must be a finite, non-negative cells-by-genes matrix.")
    n_cells, n_genes = counts.shape
    if len(data.gene_names) != n_genes:
        raise ValueError("data.gene_names must contain one name per count-matrix column.")

    theta = np.asarray(dispersion, dtype=np.float32).reshape(-1)
    if theta.shape != (n_genes,) or np.any(~np.isfinite(theta)) or np.any(theta <= 0):
        raise ValueError("dispersion must contain one finite positive value per gene.")

    control_indices = _resolve_control_indices(list(data.pert_names), control_perturbations)
    target_design, control_mask, target_names, assigned_mask = _reference_coded_target_design(
        data,
        control_indices=control_indices,
        allow_high_moi=allow_high_moi,
    )
    keep = _active_cell_mask(data) & assigned_mask
    if require_control_cells and not np.any(control_mask & keep):
        raise ValueError("No active control cells remain after applying cell and assignment masks.")
    target_counts = target_design[keep].sum(axis=0)
    empty_targets = [name for name, count in zip(target_names, target_counts, strict=True) if count == 0]
    if empty_targets:
        raise ValueError(f"Target perturbations have no active cells: {empty_targets}")

    if use_observed_size_factors and data.size_factors is not None:
        offsets = np.asarray(data.size_factors, dtype=np.float32)
        if offsets.ndim == 1:
            offsets = offsets[:, None]
        if offsets.shape not in {(n_cells, 1), (n_cells, n_genes)}:
            raise ValueError(
                "Size-factor offsets must have shape (n_cells,), (n_cells, 1), or "
                f"(n_cells, n_genes); got {offsets.shape}."
            )
        offsets_kept = offsets[keep]
    else:
        offsets_kept = np.asarray(compute_size_factors(counts[keep]), dtype=np.float32)

    nuisance_parts = [np.ones((n_cells, 1), dtype=np.float32)]
    nuisance_names = ["intercept"]
    if data.covariates is not None:
        covariates = np.asarray(data.covariates, dtype=np.float32)
        if covariates.ndim != 2 or covariates.shape[0] != n_cells:
            raise ValueError("data.covariates must have shape (n_cells, n_covariates).")
        nuisance_parts.append(covariates)
        if data.covariate_names is None:
            nuisance_names.extend(f"covariate_{idx}" for idx in range(covariates.shape[1]))
        elif len(data.covariate_names) != covariates.shape[1]:
            raise ValueError("data.covariate_names must contain one name per covariate column.")
        else:
            nuisance_names.extend(str(name) for name in data.covariate_names)

    nuisance_design = np.concatenate(nuisance_parts, axis=1)
    return JointNBDesign(
        counts=jnp.asarray(counts[keep]),
        target_design=jnp.asarray(target_design[keep]),
        nuisance_design=jnp.asarray(nuisance_design[keep]),
        offsets=jnp.asarray(offsets_kept),
        dispersion=jnp.asarray(theta),
        target_names=target_names,
        nuisance_names=tuple(nuisance_names),
        gene_names=tuple(str(name) for name in data.gene_names),
        control_mask=jnp.asarray(control_mask[keep]),
        source_cell_indices=jnp.asarray(np.flatnonzero(keep), dtype=jnp.int32),
    )


def _sum_nb_log_likelihood(
    eta: jnp.ndarray,
    counts: jnp.ndarray,
    theta: jnp.ndarray,
    *,
    cell_chunk_size: int | None,
) -> jnp.ndarray:
    def chunk_log_likelihood(eta_chunk: jnp.ndarray, count_chunk: jnp.ndarray) -> jnp.ndarray:
        likelihood = dist.NegativeBinomialLogits(
            total_count=theta,
            logits=eta_chunk - jnp.log(theta),
        )
        return likelihood.log_prob(count_chunk).sum()

    n_cells = int(counts.shape[0])
    if cell_chunk_size is None or cell_chunk_size >= n_cells:
        return chunk_log_likelihood(eta, counts)
    if cell_chunk_size < 1:
        raise ValueError("cell_chunk_size must be positive when provided.")
    total = jnp.asarray(0.0, dtype=eta.dtype)
    for start in range(0, n_cells, cell_chunk_size):
        stop = min(start + cell_chunk_size, n_cells)
        total = total + chunk_log_likelihood(eta[start:stop], counts[start:stop])
    return total


def _gene_negative_log_posterior(
    parameters: jnp.ndarray,
    *,
    counts: jnp.ndarray,
    target_design: jnp.ndarray,
    nuisance_design: jnp.ndarray,
    offset: jnp.ndarray,
    theta: jnp.ndarray,
    effect_prior_scale: float | None,
    nuisance_prior_scale: float | None,
    cell_chunk_size: int | None,
) -> jnp.ndarray:
    num_nuisance = int(nuisance_design.shape[1])
    nuisance = parameters[:num_nuisance]
    effects = parameters[num_nuisance:]
    eta = offset + nuisance_design @ nuisance + target_design @ effects
    log_density = _sum_nb_log_likelihood(
        eta,
        counts,
        theta,
        cell_chunk_size=cell_chunk_size,
    )
    if effect_prior_scale is not None:
        log_density = log_density + dist.Normal(0.0, effect_prior_scale).log_prob(effects).sum()
    if nuisance_prior_scale is not None:
        log_density = log_density + dist.Normal(0.0, nuisance_prior_scale).log_prob(nuisance).sum()
    return -log_density


def _optimize_map(
    objective_jax,
    initial: np.ndarray,
    *,
    maxiter: int,
    gradient_tolerance: float,
) -> optimize.OptimizeResult:
    """Minimize a single-argument objective.

    Note that this jits ``objective_jax`` on every call, so a caller that builds
    a fresh closure per gene pays a full JAX compilation each time. Per-gene
    loops should use :func:`make_compiled_value_and_grad` with
    :func:`optimize_map_with` instead.
    """

    value_and_grad = jax.jit(jax.value_and_grad(objective_jax))

    def scipy_objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_grad(jnp.asarray(parameters, dtype=jnp.float32))
        return float(value), np.asarray(gradient, dtype=np.float64)

    return optimize.minimize(
        scipy_objective,
        np.asarray(initial, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(maxiter), "gtol": float(gradient_tolerance)},
    )


def make_compiled_value_and_grad(
    *,
    target_design: jnp.ndarray,
    nuisance_design: jnp.ndarray,
    effect_prior_scale: float | None,
    nuisance_prior_scale: float | None,
    cell_chunk_size: int | None,
):
    """Compile the per-gene NB objective exactly once.

    The designs and priors are fixed across genes and so are closed over; the
    gene-varying counts, offset, and dispersion are passed as traced arguments.
    That way every gene reuses one compiled kernel. Closing over the gene-varying
    arrays instead - the obvious way to write a per-gene loop - makes JAX
    recompile for each gene, which costs roughly 150 ms against a few ms of
    actual optimization.

    Because the offset is a traced argument, this also covers the
    control-anchored fit, where the fixed baseline enters as a per-gene offset
    and ``nuisance_design`` is empty.
    """

    def objective(
        parameters: jnp.ndarray,
        counts: jnp.ndarray,
        offset: jnp.ndarray,
        theta: jnp.ndarray,
    ) -> jnp.ndarray:
        return _gene_negative_log_posterior(
            parameters,
            counts=counts,
            target_design=target_design,
            nuisance_design=nuisance_design,
            offset=offset,
            theta=theta,
            effect_prior_scale=effect_prior_scale,
            nuisance_prior_scale=nuisance_prior_scale,
            cell_chunk_size=cell_chunk_size,
        )

    return jax.jit(jax.value_and_grad(objective))


def optimize_map_with(
    value_and_grad,
    initial: np.ndarray,
    *args: jnp.ndarray,
    maxiter: int,
    gradient_tolerance: float,
) -> optimize.OptimizeResult:
    """Run L-BFGS against a pre-compiled ``value_and_grad``.

    ``args`` are forwarded to it after the parameter vector, so the compiled
    kernel is shared across every call.
    """

    def scipy_objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_grad(jnp.asarray(parameters, dtype=jnp.float32), *args)
        return float(value), np.asarray(gradient, dtype=np.float64)

    return optimize.minimize(
        scipy_objective,
        np.asarray(initial, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(maxiter), "gtol": float(gradient_tolerance)},
    )


def _observed_nb_weights(counts: np.ndarray, eta: np.ndarray, theta: float) -> np.ndarray:
    mean = np.exp(np.clip(np.asarray(eta, dtype=np.float64), -30.0, 30.0))
    theta_value = float(theta)
    return theta_value * (np.asarray(counts, dtype=np.float64) + theta_value) * mean / np.square(
        theta_value + mean
    )


def low_moi_marginal_variances(
    *,
    counts: np.ndarray,
    eta: np.ndarray,
    theta: float,
    target_design: np.ndarray,
    nuisance_design: np.ndarray,
    effect_prior_scale: float | None,
    nuisance_prior_scale: float | None,
    curvature_jitter: float = 1e-8,
) -> np.ndarray:
    """Return exact effect marginal variances without a dense effect Hessian.

    For a low-MOI one-hot target design, the effect-effect information block is
    diagonal.  The diagonal of the inverse Schur complement is obtained through
    a nuisance-sized Woodbury solve.
    """

    x = np.asarray(target_design, dtype=np.float64)
    z = np.asarray(nuisance_design, dtype=np.float64)
    if x.ndim != 2 or np.any(np.count_nonzero(x, axis=1) > 1):
        raise ValueError("target_design must be low-MOI with at most one nonzero entry per row.")
    weights = _observed_nb_weights(counts, eta, theta)
    effect_precision = 0.0 if effect_prior_scale is None else 1.0 / float(effect_prior_scale) ** 2
    nuisance_precision = 0.0 if nuisance_prior_scale is None else 1.0 / float(nuisance_prior_scale) ** 2

    d = np.sum(weights[:, None] * np.square(x), axis=0) + effect_precision
    if np.any(~np.isfinite(d)) or np.any(d <= 0):
        raise np.linalg.LinAlgError("Effect curvature is non-positive or non-finite.")
    a = z.T @ (weights[:, None] * z)
    if nuisance_precision:
        a = a + nuisance_precision * np.eye(a.shape[0])
    c = z.T @ (weights[:, None] * x)
    d_inverse = 1.0 / d
    nuisance_schur = a - (c * d_inverse[None, :]) @ c.T
    nuisance_schur = nuisance_schur + float(curvature_jitter) * np.eye(nuisance_schur.shape[0])
    solved = np.linalg.solve(nuisance_schur, c)
    correction = np.sum(c * solved, axis=0) * np.square(d_inverse)
    variances = d_inverse + correction
    if np.any(~np.isfinite(variances)) or np.any(variances <= 0):
        raise np.linalg.LinAlgError("Marginal effect variances are non-positive or non-finite.")
    return variances


def _sample_effect_pairs(
    num_effects: int,
    *,
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_effects < 2 or max_pairs == 0:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty
    total_pairs = num_effects * (num_effects - 1) // 2
    num_pairs = min(int(max_pairs), total_pairs)
    if num_pairs == total_pairs:
        left, right = np.triu_indices(num_effects, k=1)
        return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)

    rng = np.random.default_rng(seed)
    if total_pairs <= 4 * num_pairs:
        all_left, all_right = np.triu_indices(num_effects, k=1)
        selected = rng.choice(total_pairs, size=num_pairs, replace=False)
        return (
            np.asarray(all_left[selected], dtype=np.int64),
            np.asarray(all_right[selected], dtype=np.int64),
        )

    sampled: set[tuple[int, int]] = set()
    while len(sampled) < num_pairs:
        candidates = rng.integers(0, num_effects, size=(2 * (num_pairs - len(sampled)), 2))
        for left, right in candidates:
            if left == right:
                continue
            pair = (int(min(left, right)), int(max(left, right)))
            sampled.add(pair)
            if len(sampled) == num_pairs:
                break
    ordered = np.asarray(sorted(sampled), dtype=np.int64)
    return ordered[:, 0], ordered[:, 1]


def projected_diagonal_marginal_variances(
    *,
    counts: np.ndarray,
    eta: np.ndarray,
    theta: float,
    target_design: np.ndarray,
    nuisance_design: np.ndarray,
    effect_prior_scale: float | None,
    nuisance_prior_scale: float | None,
    curvature_jitter: float = 1e-8,
    correlation_diagnostic_pairs: int = 256,
    correlation_seed: int = 0,
) -> ProjectedDiagonalVariance:
    """Approximate effect marginals from diagonal nuisance-projected curvature.

    This is intended for randomly assorted high-MOI guides. It keeps the exact
    diagonal of the effect Schur complement but ignores its off-diagonal terms.
    The returned sampled correlations diagnose that approximation without
    constructing a dense guide-by-guide information matrix.
    """

    x = np.asarray(target_design, dtype=np.float64)
    z = np.asarray(nuisance_design, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("target_design must be a non-empty cells-by-effects matrix.")
    if z.ndim != 2 or z.shape[0] != x.shape[0]:
        raise ValueError("nuisance_design must align with target_design rows.")
    if correlation_diagnostic_pairs < 0:
        raise ValueError("correlation_diagnostic_pairs must be non-negative.")

    weights = _observed_nb_weights(counts, eta, theta)
    effect_precision = 0.0 if effect_prior_scale is None else 1.0 / float(effect_prior_scale) ** 2
    nuisance_precision = 0.0 if nuisance_prior_scale is None else 1.0 / float(nuisance_prior_scale) ** 2
    weighted_x = weights[:, None] * x
    raw_effect_diagonal = np.sum(weighted_x * x, axis=0)
    nuisance_information = z.T @ (weights[:, None] * z)
    if nuisance_precision:
        nuisance_information = nuisance_information + nuisance_precision * np.eye(z.shape[1])
    nuisance_information = nuisance_information + float(curvature_jitter) * np.eye(z.shape[1])
    nuisance_cross = z.T @ weighted_x
    projected_cross = np.linalg.solve(nuisance_information, nuisance_cross)
    projected_design_diagonal = raw_effect_diagonal - np.sum(nuisance_cross * projected_cross, axis=0)
    projected_curvature = projected_design_diagonal + effect_precision
    if np.any(~np.isfinite(projected_curvature)) or np.any(projected_curvature <= 0):
        raise np.linalg.LinAlgError("Projected effect curvature is non-positive or non-finite.")

    left_indices, right_indices = _sample_effect_pairs(
        x.shape[1],
        max_pairs=correlation_diagnostic_pairs,
        seed=correlation_seed,
    )
    correlations: list[np.ndarray] = []
    for start in range(0, left_indices.size, 64):
        stop = min(start + 64, left_indices.size)
        left = left_indices[start:stop]
        right = right_indices[start:stop]
        raw_cross = np.sum(
            weights[:, None] * x[:, left] * x[:, right],
            axis=0,
        )
        nuisance_projection = np.sum(
            nuisance_cross[:, left] * projected_cross[:, right],
            axis=0,
        )
        denominator = np.sqrt(
            np.maximum(projected_design_diagonal[left], 0.0)
            * np.maximum(projected_design_diagonal[right], 0.0)
        )
        valid = np.isfinite(denominator) & (denominator > 1e-12)
        if np.any(valid):
            correlations.append(np.abs((raw_cross - nuisance_projection)[valid] / denominator[valid]))
    if correlations:
        absolute_correlations = np.concatenate(correlations)
        max_correlation = float(np.max(absolute_correlations))
        median_correlation = float(np.median(absolute_correlations))
        num_pairs = int(absolute_correlations.size)
    else:
        max_correlation = np.nan
        median_correlation = np.nan
        num_pairs = 0
    return ProjectedDiagonalVariance(
        variance=1.0 / projected_curvature,
        projected_curvature=projected_curvature,
        max_abs_projected_correlation=max_correlation,
        median_abs_projected_correlation=median_correlation,
        num_correlation_pairs=num_pairs,
    )


def fit_joint_nb_laplace(
    design: JointNBDesign,
    *,
    effect_prior_scale: float | None = 1.0,
    nuisance_prior_scale: float | None = None,
    cell_chunk_size: int | None = None,
    maxiter: int = 500,
    gradient_tolerance: float = 1e-6,
    covariance_approximation: str = "auto",
    correlation_diagnostic_pairs: int = 256,
    correlation_seed: int = 0,
    correlation_warning_threshold: float = 0.2,
) -> JointLaplaceFit:
    """Fit independent per-gene joint NB MAPs and marginal approximations.

    ``auto`` uses exact low-MOI marginals when each cell has at most one target
    and nuisance-projected diagonal curvature otherwise.
    """

    if effect_prior_scale is not None and effect_prior_scale <= 0:
        raise ValueError("effect_prior_scale must be positive or None.")
    if nuisance_prior_scale is not None and nuisance_prior_scale <= 0:
        raise ValueError("nuisance_prior_scale must be positive or None.")
    if maxiter < 1:
        raise ValueError("maxiter must be positive.")
    if cell_chunk_size is not None and cell_chunk_size < 1:
        raise ValueError("cell_chunk_size must be positive when provided.")
    valid_covariance_modes = {"auto", "exact_low_moi", "projected_diagonal"}
    if covariance_approximation not in valid_covariance_modes:
        raise ValueError(
            f"covariance_approximation must be one of {sorted(valid_covariance_modes)}."
        )
    if correlation_diagnostic_pairs < 0:
        raise ValueError("correlation_diagnostic_pairs must be non-negative.")
    if not np.isfinite(correlation_warning_threshold) or correlation_warning_threshold <= 0:
        raise ValueError("correlation_warning_threshold must be finite and positive.")

    counts = np.asarray(design.counts)
    target_design = np.asarray(design.target_design, dtype=np.float32)
    nuisance_design = np.asarray(design.nuisance_design, dtype=np.float32)
    offsets = np.asarray(design.offsets, dtype=np.float32)
    dispersion = np.asarray(design.dispersion, dtype=np.float32)
    num_nuisance = nuisance_design.shape[1]
    num_targets = target_design.shape[1]
    num_genes = counts.shape[1]
    is_low_moi = bool(np.all(np.count_nonzero(target_design, axis=1) <= 1))
    resolved_covariance_approximation = covariance_approximation
    if resolved_covariance_approximation == "auto":
        resolved_covariance_approximation = "exact_low_moi" if is_low_moi else "projected_diagonal"
    if resolved_covariance_approximation == "exact_low_moi" and not is_low_moi:
        raise ValueError("exact_low_moi covariance requires at most one active target per cell.")

    effect_mean = np.zeros((num_targets, num_genes), dtype=np.float32)
    effect_scale = np.zeros_like(effect_mean)
    nuisance_mean = np.zeros((num_nuisance, num_genes), dtype=np.float32)
    converged = np.zeros(num_genes, dtype=bool)
    iterations = np.zeros(num_genes, dtype=np.int32)
    objectives = np.zeros(num_genes, dtype=np.float64)
    max_projected_correlation = np.full(num_genes, np.nan, dtype=np.float32)
    median_projected_correlation = np.full(num_genes, np.nan, dtype=np.float32)
    num_projected_correlation_pairs = np.zeros(num_genes, dtype=np.int32)

    # Compiled once for every gene; see make_compiled_value_and_grad.
    value_and_grad = make_compiled_value_and_grad(
        target_design=jnp.asarray(target_design),
        nuisance_design=jnp.asarray(nuisance_design),
        effect_prior_scale=effect_prior_scale,
        nuisance_prior_scale=nuisance_prior_scale,
        cell_chunk_size=cell_chunk_size,
    )

    for gene_index in range(num_genes):
        y = jnp.asarray(counts[:, gene_index], dtype=jnp.float32)
        offset_column = offsets[:, 0] if offsets.shape[1] == 1 else offsets[:, gene_index]
        offset = jnp.asarray(offset_column, dtype=jnp.float32)
        theta = jnp.asarray(dispersion[gene_index], dtype=jnp.float32)

        initial = np.zeros(num_nuisance + num_targets, dtype=np.float64)
        initial[0] = float(np.log(np.mean(np.asarray(y)) + 0.1) - np.mean(offset_column))

        result = optimize_map_with(
            value_and_grad,
            initial,
            y,
            offset,
            theta,
            maxiter=maxiter,
            gradient_tolerance=gradient_tolerance,
        )
        map_parameters = np.asarray(result.x, dtype=np.float64)
        nuisance_map = map_parameters[:num_nuisance]
        effect_map = map_parameters[num_nuisance:]
        eta_map = (
            np.asarray(offset_column, dtype=np.float64)
            + nuisance_design.astype(np.float64) @ nuisance_map
            + target_design.astype(np.float64) @ effect_map
        )
        if resolved_covariance_approximation == "exact_low_moi":
            variances = low_moi_marginal_variances(
                counts=counts[:, gene_index],
                eta=eta_map,
                theta=float(dispersion[gene_index]),
                target_design=target_design,
                nuisance_design=nuisance_design,
                effect_prior_scale=effect_prior_scale,
                nuisance_prior_scale=nuisance_prior_scale,
            )
        else:
            projected = projected_diagonal_marginal_variances(
                counts=counts[:, gene_index],
                eta=eta_map,
                theta=float(dispersion[gene_index]),
                target_design=target_design,
                nuisance_design=nuisance_design,
                effect_prior_scale=effect_prior_scale,
                nuisance_prior_scale=nuisance_prior_scale,
                correlation_diagnostic_pairs=correlation_diagnostic_pairs,
                correlation_seed=correlation_seed,
            )
            variances = projected.variance
            max_projected_correlation[gene_index] = projected.max_abs_projected_correlation
            median_projected_correlation[gene_index] = projected.median_abs_projected_correlation
            num_projected_correlation_pairs[gene_index] = projected.num_correlation_pairs

        nuisance_mean[:, gene_index] = nuisance_map.astype(np.float32)
        effect_mean[:, gene_index] = effect_map.astype(np.float32)
        effect_scale[:, gene_index] = np.sqrt(variances).astype(np.float32)
        converged[gene_index] = bool(result.success and np.isfinite(result.fun))
        iterations[gene_index] = int(result.nit)
        objectives[gene_index] = float(result.fun)

    finite_diagnostic = max_projected_correlation[np.isfinite(max_projected_correlation)]
    if finite_diagnostic.size and np.max(finite_diagnostic) > correlation_warning_threshold:
        warnings.warn(
            "Nuisance-projected effect correlations exceed the requested high-MOI approximation "
            f"threshold (max={np.max(finite_diagnostic):.3f}, threshold={correlation_warning_threshold:.3f}). "
            "Inspect correlated guide blocks before treating diagonal marginals as calibrated.",
            RuntimeWarning,
            stacklevel=2,
        )

    z_values = effect_mean / np.clip(effect_scale, 1e-8, None)
    return JointLaplaceFit(
        posterior_mean=jnp.asarray(effect_mean),
        posterior_scale=jnp.asarray(effect_scale),
        z_values=jnp.asarray(z_values),
        nuisance_mean=jnp.asarray(nuisance_mean),
        converged=jnp.asarray(converged),
        optimizer_iterations=jnp.asarray(iterations),
        objective=jnp.asarray(objectives),
        target_names=design.target_names,
        nuisance_names=design.nuisance_names,
        gene_names=design.gene_names,
        effect_prior_scale=effect_prior_scale,
        covariance_approximation=resolved_covariance_approximation,
        max_abs_projected_effect_correlation=jnp.asarray(max_projected_correlation),
        median_abs_projected_effect_correlation=jnp.asarray(median_projected_correlation),
        num_projected_effect_correlation_pairs=jnp.asarray(num_projected_correlation_pairs),
    )


__all__ = [
    "JointLaplaceFit",
    "JointNBDesign",
    "ProjectedDiagonalVariance",
    "fit_joint_nb_laplace",
    "low_moi_marginal_variances",
    "prepare_joint_nb_design",
    "projected_diagonal_marginal_variances",
]
