"""Categorical low-MOI design preparation without a dense target matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import jax.numpy as jnp
import numpy as np

from perturbo.core import PerTurboData
from perturbo.utils import compute_size_factors


@dataclass(frozen=True)
class LowMOIDesign:
    """Arrays for one-control-versus-categorical-target low-MOI inference.

    ``target_codes`` is ``-1`` for controls and a zero-based target index for
    targeting cells. It deliberately replaces the former cells-by-targets
    one-hot matrix; target groups are recovered with equality/index operations.
    """

    counts: jnp.ndarray
    target_codes: jnp.ndarray
    nuisance_design: jnp.ndarray
    offsets: jnp.ndarray
    dispersion: jnp.ndarray
    target_names: tuple[str, ...]
    nuisance_names: tuple[str, ...]
    gene_names: tuple[str, ...]
    control_mask: jnp.ndarray
    source_cell_indices: jnp.ndarray
    batch_codes: jnp.ndarray | None = None
    batch_names: tuple[str, ...] | None = None
    _gene_independent_token: object | None = None

    @property
    def num_cells(self) -> int:
        return int(self.counts.shape[0])

    @property
    def num_genes(self) -> int:
        return int(self.counts.shape[1])

    @property
    def num_targets(self) -> int:
        return len(self.target_names)


def _control_indices(names: Iterable[str], requested: Iterable[str | int]) -> tuple[int, ...]:
    values = tuple(requested)
    if not values:
        raise ValueError("control_perturbations must contain at least one name or index.")
    name_to_index = {str(name): index for index, name in enumerate(names)}
    indices: list[int] = []
    for value in values:
        index = name_to_index[value] if isinstance(value, str) else int(value)
        if index < 0 or index >= len(name_to_index):
            raise IndexError("control perturbation index is outside the perturbation categories.")
        if index not in indices:
            indices.append(index)
    return tuple(indices)


def prepare_low_moi_design(
    data: PerTurboData,
    *,
    control_perturbations: Iterable[str | int],
    dispersion: np.ndarray | jnp.ndarray,
    use_observed_size_factors: bool = True,
    extra_control_cells: np.ndarray | None = None,
) -> LowMOIDesign:
    """Prepare a categorical low-MOI design from ``PerTurboData``.

    Each retained cell must carry exactly one assignment. Matrix assignments are
    accepted only when every retained row contains one active guide; unassigned
    rows are excluded and control guides are pooled into code ``-1``.

    ``extra_control_cells`` adds cells to the control pool *without* clearing
    their target assignment, so a cell can be both a member of the null pool and
    a member of the target being tested against it. That is what lets every
    control guide serve in the pool and still be tested as a pseudo-target,
    rather than half of them being spent as a permanent comparison set.

    The resulting test is a leave-one-in permutation test and remains calibrated:
    the CRT null is that a target's label is exchangeable within its pair pool,
    and resampling assignments inside the pool is valid whether or not the
    target's own cells are in it. Self-inclusion attenuates the contrast by
    roughly one part in the guide count, and attenuates the null identically.

    One caveat it cannot remove: the nuisance baseline is fit on control cells,
    so a pseudo-target inside the pool contributes to the baseline it is tested
    against, and a genuinely active control guide slightly masks itself. That
    dilution is one part in the guide count too. Leave-one-out would avoid it,
    at the cost of refitting the null once per guide.
    """

    counts = np.asarray(data.counts)
    if counts.ndim != 2 or np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("data.counts must be a finite, non-negative cells-by-genes matrix.")
    n_cells, n_genes = counts.shape
    if len(data.gene_names) != n_genes:
        raise ValueError("data.gene_names must contain one name per count column.")
    theta = np.asarray(dispersion, dtype=np.float32).reshape(-1)
    if theta.shape != (n_genes,) or np.any(~np.isfinite(theta)) or np.any(theta <= 0):
        raise ValueError("dispersion must contain one finite positive value per gene.")

    names = tuple(str(name) for name in data.pert_names)
    controls = _control_indices(names, control_perturbations)
    target_original = tuple(index for index in range(len(names)) if index not in controls)
    if not target_original:
        raise ValueError("At least one non-control perturbation is required.")
    original_to_target = {original: code for code, original in enumerate(target_original)}
    pert = np.asarray(data.pert_id)
    if pert.ndim == 1:
        labels = pert.astype(np.int64, copy=False)
        if not np.array_equal(labels, pert) or np.any(labels < 0) or np.any(labels >= len(names)):
            raise ValueError("one-dimensional perturbation labels must be valid integer categories.")
        assigned = np.ones(n_cells, dtype=bool)
    elif pert.ndim == 2:
        binary = np.asarray(pert > 0, dtype=bool)
        if binary.shape != (n_cells, len(names)):
            raise ValueError("matrix perturbation assignments must align with perturbation names.")
        assignments = binary.sum(axis=1)
        if np.any(assignments > 1):
            raise ValueError("prepare_low_moi_design requires at most one assignment per cell.")
        assigned = assignments == 1
        labels = np.full(n_cells, -1, dtype=np.int64)
        labels[assigned] = np.argmax(binary[assigned], axis=1)
    else:
        raise ValueError("data.pert_id must be a label vector or assignment matrix.")

    if data.cell_mask is None:
        active = np.ones(n_cells, dtype=bool)
    else:
        active = np.asarray(data.cell_mask, dtype=bool).reshape(-1)
        if active.shape != (n_cells,):
            raise ValueError("data.cell_mask must contain one value per cell.")
    keep = active & assigned
    control_mask = np.isin(labels, controls) & keep
    if extra_control_cells is not None:
        extra = np.asarray(extra_control_cells, dtype=bool).reshape(-1)
        if extra.shape != (n_cells,):
            raise ValueError("extra_control_cells must contain one value per cell.")
        control_mask = control_mask | (extra & keep)
    if not np.any(control_mask):
        raise ValueError("No active control cells remain after masking.")
    target_codes = np.full(n_cells, -1, dtype=np.int32)
    for original, code in original_to_target.items():
        target_codes[labels == original] = code
    target_codes = target_codes[keep]
    target_names = tuple(names[index] for index in target_original)
    target_counts = np.bincount(target_codes[target_codes >= 0], minlength=len(target_names))
    if np.any(target_counts == 0):
        missing = [name for name, count in zip(target_names, target_counts, strict=True) if count == 0]
        raise ValueError(f"Target perturbations have no active cells: {missing}")

    if use_observed_size_factors and data.size_factors is not None:
        offsets = np.asarray(data.size_factors, dtype=np.float32)
        if offsets.ndim == 1:
            offsets = offsets[:, None]
        if offsets.shape not in {(n_cells, 1), (n_cells, n_genes)}:
            raise ValueError("size factors must have one or one-per-gene offset per cell.")
        offsets = offsets[keep]
    else:
        offsets = np.asarray(compute_size_factors(counts[keep]), dtype=np.float32)
    nuisance = [np.ones((n_cells, 1), dtype=np.float32)]
    nuisance_names = ["intercept"]
    if data.covariates is not None:
        covariates = np.asarray(data.covariates, dtype=np.float32)
        if covariates.ndim != 2 or covariates.shape[0] != n_cells:
            raise ValueError("data.covariates must have shape (n_cells, n_covariates).")
        nuisance.append(covariates)
        if data.covariate_names is None:
            nuisance_names.extend(f"covariate_{index}" for index in range(covariates.shape[1]))
        elif len(data.covariate_names) == covariates.shape[1]:
            nuisance_names.extend(str(name) for name in data.covariate_names)
        else:
            raise ValueError("data.covariate_names must contain one name per covariate.")
    batch_codes = None
    batch_names = None
    if data.categorical_batch_codes is not None:
        raw_batch_codes = np.asarray(data.categorical_batch_codes, dtype=np.int32).reshape(-1)
        if raw_batch_codes.shape != (n_cells,) or np.any(raw_batch_codes < 0):
            raise ValueError("categorical_batch_codes must contain one non-negative code per cell.")
        if data.categorical_batch_names is None:
            n_batches = int(raw_batch_codes.max()) + 1
            names_for_batch = tuple(f"batch_{index}" for index in range(n_batches))
        else:
            names_for_batch = tuple(str(name) for name in data.categorical_batch_names)
            n_batches = len(names_for_batch)
        if n_batches < 1 or np.any(raw_batch_codes >= n_batches):
            raise ValueError("categorical_batch_codes contain an unknown batch.")
        batch_codes = jnp.asarray(raw_batch_codes[keep], dtype=jnp.int32)
        batch_names = names_for_batch
    return LowMOIDesign(
        counts=jnp.asarray(counts[keep], dtype=jnp.float32),
        target_codes=jnp.asarray(target_codes),
        nuisance_design=jnp.asarray(np.concatenate(nuisance, axis=1)[keep]),
        offsets=jnp.asarray(offsets),
        dispersion=jnp.asarray(theta),
        target_names=target_names,
        nuisance_names=tuple(nuisance_names),
        gene_names=tuple(str(name) for name in data.gene_names),
        control_mask=jnp.asarray(control_mask[keep]),
        source_cell_indices=jnp.asarray(np.flatnonzero(keep), dtype=jnp.int32),
        batch_codes=batch_codes,
        batch_names=batch_names,
        _gene_independent_token=data._analysis_design_token,
    )
