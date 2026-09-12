"""Command-line entrypoints for perturbo."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import warnings
from pathlib import Path
import time
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .core import *  # noqa: F403 - CLI delegates to the fitting workflow namespace.
from .core import (
    BACKED_ROW_CHUNK_SIZE,
    BetaFit,
    PerTurboData,
    SVIConfig,
    VALID_CLI_SIZE_FACTOR_MODES,
    _PerturbationChunk,
    _build_matrix_membership,
    _build_obs_membership,
    _compute_gene_outlier_thresholds,
    _construct_perturbation_chunks,
    _count_gene_outliers_per_cell,
    _dedupe_preserve_order,
    _extract_gene_names,
    _extract_pert_names,
    _fixed_zero_size_factors,
    _get_layer_matrix,
    _group_perturbation_matrix_by_element,
    _is_censored_model_name,
    _load_from_path_with_backing,
    _load_perturbation_element_mapping,
    _normalize_likelihood_name,
    _pad_cortado_data_for_chunk,
    _resolve_adata,
    _resolve_cli_size_factor_mode,
    _resolve_control_element_mask,
    _resolve_perturbation_modality,
    _save_loss_plot,
    _save_multi_loss_plot,
    _sum_matrix_rows_bounded,
    _ReusableSVIRunner,
    _validate_cli_input_keys,
    _validate_clip_percentile,
    _validate_gene_outlier_threshold_floor,
    _validate_guide_strategy,
    apply_covariate_transform,
    fit_control,
    fit_perturbation_effects,
    load_analysis_cells,
    load_controls,
    subset_control_fit_genes,
)
from .io import MuDataSetup, control_fit_arrays, save_array_bundle, save_light_fit_bundle
from .sparse_design import IndexedDesignMatrix
from .crt import (
    CRTAccumulator,
    CRT_ALL_TAIL_FAMILIES,
    CRT_MECHANISMS,
    CRT_SADDLEPOINT_FAMILY,
    DEFAULT_NEWTON_STEP_TOLERANCE,
    exclude_targets,
    fit_shared_propensity_coefficients,
    prepare_all_cells_propensity,
    prepare_crt_baseline,
    run_crt_all_cells,
    run_crt_for_chunk,
    validate_crt_config,
)
from .core import measure_realized_moi
from .results import (
    build_guide_efficiency_df,
    build_standard_element_effects_df,
    load_pairs_to_test,
    restrict_effects_to_pairs,
)
from .training_schedule import resolve_training_schedule


def _beta_fit_arrays(beta_fit: BetaFit) -> dict[str, Any]:
    return {
        "posterior_mean": beta_fit.posterior_mean,
        "posterior_scale": beta_fit.posterior_scale,
        "z_values": beta_fit.z_values,
        "losses": beta_fit.losses,
        "dispersion_excess_inverse": beta_fit.dispersion_excess_inverse,
    }


def _guide_posterior_arrays(beta_fit: BetaFit) -> dict[str, Any]:
    return {
        "guide_effect_mean": beta_fit.guide_effect_mean,
        "guide_effect_scale": beta_fit.guide_effect_scale,
        "guide_effect_z_values": beta_fit.guide_effect_z_values,
        "guide_relative_efficiency_mean": beta_fit.guide_relative_efficiency_mean,
        "guide_relative_efficiency_scale": beta_fit.guide_relative_efficiency_scale,
        "guide_offset_mean": beta_fit.guide_offset_mean,
        "guide_offset_scale": beta_fit.guide_offset_scale,
        "guide_dispersion_excess_inverse": beta_fit.guide_dispersion_excess_inverse,
    }


def _guide_efficacy_for_cli(
    beta_fit: BetaFit,
    *,
    guide_effect_strategy: str,
    n_guides: int,
) -> np.ndarray | None:
    if str(guide_effect_strategy).lower() == "shared":
        return np.ones((int(n_guides),), dtype=np.float32)
    relative = beta_fit.guide_relative_efficiency_mean
    if relative is None:
        return None
    arr = np.asarray(relative, dtype=np.float32)
    if arr.shape[0] != int(n_guides):
        return None
    if arr.ndim == 1:
        return np.clip(arr, a_min=0.0, a_max=None)
    return np.clip(arr.reshape(int(n_guides), -1).mean(axis=1), a_min=0.0, a_max=None).astype(np.float32)


_GUIDE_POSTERIOR_FIELDS = (
    "guide_effect_mean",
    "guide_effect_scale",
    "guide_effect_z_values",
    "guide_relative_efficiency_mean",
    "guide_relative_efficiency_scale",
    "guide_offset_mean",
    "guide_offset_scale",
    "guide_dispersion_excess_inverse",
)


def _merge_chunk_guide_posteriors(
    buffers: dict[str, np.ndarray],
    seen: dict[str, np.ndarray],
    chunk_fit: BetaFit,
    *,
    chunk_guide_names: list[str],
    global_guide_names: list[str],
) -> None:
    """Merge guide-indexed chunk results by guide name, never by local row."""
    if len(set(global_guide_names)) != len(global_guide_names):
        raise ValueError("Guide names must be unique to combine chunked posterior arrays.")
    global_index = {name: index for index, name in enumerate(global_guide_names)}
    try:
        target_rows = np.asarray([global_index[name] for name in chunk_guide_names], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Chunk returned an unknown guide name: {error.args[0]!r}.") from error

    for field in _GUIDE_POSTERIOR_FIELDS:
        value = getattr(chunk_fit, field)
        if value is None:
            continue
        array = np.asarray(value)
        if array.shape[0] != len(chunk_guide_names):
            raise ValueError(
                f"Chunk guide posterior {field!r} has {array.shape[0]} rows for "
                f"{len(chunk_guide_names)} guide names."
            )
        if field not in buffers:
            buffers[field] = np.full(
                (len(global_guide_names), *array.shape[1:]),
                np.nan,
                dtype=array.dtype,
            )
            seen[field] = np.zeros(len(global_guide_names), dtype=bool)
        elif buffers[field].shape[1:] != array.shape[1:]:
            raise ValueError(
                f"Chunk guide posterior {field!r} changed trailing shape from "
                f"{buffers[field].shape[1:]} to {array.shape[1:]}."
            )
        buffers[field][target_rows] = array
        seen[field][target_rows] = True


def _finalize_chunk_guide_posteriors(
    buffers: dict[str, np.ndarray],
    seen: dict[str, np.ndarray],
    *,
    global_guide_names: list[str],
) -> dict[str, jnp.ndarray]:
    for field, mask in seen.items():
        if not np.all(mask):
            missing = [name for name, present in zip(global_guide_names, mask, strict=True) if not present]
            raise ValueError(
                f"Chunked fitting did not return {field!r} for {len(missing)} guide(s): "
                + ", ".join(missing[:5])
                + ("..." if len(missing) > 5 else "")
            )
    return {field: jnp.asarray(array) for field, array in buffers.items()}


def _bundle_size_factors(
    adata: Any,
    *,
    selected_row_indices: np.ndarray | None = None,
    size_factor_mode: str,
    size_factor_key: str | None,
    library_size_key: str | None,
    library_size_center_log_mean: float | None,
    gene_clip_thresholds: np.ndarray | None,
    winsorize_gene_expression: bool,
) -> tuple[np.ndarray | None, str]:
    """Materialize the exact per-cell offset needed by a simulation bundle."""
    source_n_obs = int(adata.n_obs)
    if selected_row_indices is None:
        selected_rows = np.arange(source_n_obs, dtype=np.int64)
    else:
        selected_rows = np.asarray(selected_row_indices, dtype=np.int64)
        if selected_rows.ndim != 1:
            raise ValueError("selected_row_indices must be one-dimensional.")
        if selected_rows.size and (
            np.any(selected_rows < 0) or np.any(selected_rows >= source_n_obs)
        ):
            raise IndexError("selected_row_indices contains a row outside the source data.")
    n_obs = int(selected_rows.size)
    selected_obs = adata.obs.iloc[selected_rows]
    if size_factor_mode == "infer":
        return None, "latent"
    if size_factor_mode == "none":
        return np.zeros((n_obs, 1), dtype=np.float32), "fixed_zero"
    if size_factor_key is not None:
        values = pd.to_numeric(selected_obs[size_factor_key], errors="coerce").to_numpy(dtype=np.float32)
        return values.reshape(-1, 1), "obs_size_factor"
    if library_size_key is not None:
        library_size = pd.to_numeric(selected_obs[library_size_key], errors="coerce").to_numpy(dtype=np.float64)
        center = float(library_size_center_log_mean or 0.0)
        return (np.log1p(library_size) - center).astype(np.float32)[:, None], "obs_library_size"

    totals = np.empty(n_obs, dtype=np.float64)
    row_chunk_size = min(BACKED_ROW_CHUNK_SIZE, max(n_obs, 1))
    for start in range(0, n_obs, row_chunk_size):
        stop = min(start + row_chunk_size, n_obs)
        block = adata.X[selected_rows[start:stop], :]
        if winsorize_gene_expression and gene_clip_thresholds is not None:
            if hasattr(block, "toarray"):
                block = block.toarray()
            block = np.asarray(block)
            block = np.minimum(block, np.asarray(gene_clip_thresholds)[None, :])
            block_totals = block.sum(axis=1, dtype=np.float64)
        else:
            block_totals = block.sum(axis=1, dtype=np.float64)
        totals[start:stop] = np.asarray(block_totals, dtype=np.float64).reshape(-1)
    center = float(library_size_center_log_mean or 0.0)
    return (np.log1p(totals) - center).astype(np.float32)[:, None], "analyzed_count_total"


def _gene_chunk_configuration_problem(args, size_factor_mode: str) -> str | None:
    """Gene blocks are independent only under the supported fixed-offset model."""
    if args.likelihood != "negbin":
        return "gene chunking currently requires --likelihood negbin"
    if size_factor_mode not in {"observed", "none"}:
        return "gene chunking requires observed or fixed-zero size factors"
    if args.num_factors or args.guide_random_effects or args.propagate_baseline_uncertainty:
        return "gene chunking requires zero latent factors and no guide random effects or baseline uncertainty propagation"
    if args.guide_effect_strategy != "shared" or args.guide_activity_mode != "always_on":
        return "gene chunking currently requires shared, always-on guide effects"
    if args.fit_perturbation_dispersion:
        return "gene chunking currently requires fixed perturbation dispersion"
    return None


def _crt_configuration_problem(args, size_factor_mode) -> str | None:
    """Why this run cannot carry the CRT, or None when it can.

    The same conditions ``validate_crt_config`` enforces, checked before any
    fitting so that a default-on test can decline quietly instead of wasting a
    stage-one fit to discover the conflict.
    """
    if args.likelihood not in ("nb", "negbin"):
        return f"--likelihood {args.likelihood} is not the plain negative binomial the test requires."
    if size_factor_mode not in ("observed", "none"):
        return (
            f"size_factor_mode={size_factor_mode!r} is not supported: the test conditions on a fixed "
            "offset, and a per-cell offset fit jointly with the effect would depend on the cell's own label."
        )
    if args.num_factors:
        return f"--num-factors {args.num_factors} adds latent factors, which can absorb perturbation signal."
    if args.guide_random_effects:
        return "--guide-random-effects adds a per-guide latent outside the nuisance design."
    return None


def _crt_tested_cell_rows(
    element_names: list[str],
    membership: list[np.ndarray],
    *,
    control_selector,
    include_control_elements: bool,
) -> np.ndarray:
    """Analysis rows the control-anchored CRT will test, across every chunk.

    Perturbation chunking splits these rows between chunks, but the shared
    selection model has to see all of them at once - that is the whole point of
    fitting it here rather than inside a chunk. Control elements are left out
    unless the run tests them as targets too, matching what each chunk does.
    """

    control_mask = _resolve_control_element_mask(
        [str(name) for name in element_names],
        control_selector,
        infer_control_elements=False,
    )
    tested = [
        np.asarray(rows, dtype=np.int64)
        for index, rows in enumerate(membership)
        if include_control_elements or not control_mask[index]
    ]
    if not tested:
        return np.zeros(0, dtype=np.int64)
    # Sorted and de-duplicated: a cell carries at most one element on this
    # path, but the union is taken over element lists all the same.
    return np.unique(np.concatenate(tested))


def _crt_tested_cell_mask(
    chunk_data: PerTurboData,
    *,
    control_selector,
    include_control_elements: bool,
) -> np.ndarray:
    """Cells of ``chunk_data`` the control-anchored CRT will actually test.

    The same rules the chunk itself applies: control elements are dropped
    unless the run tests them as targets (``exclude_targets``), a cell carrying
    two elements is set aside rather than reinterpreted, and a masked-out cell
    stays out (``build_chunk_design``). The shared selection model has to be
    fitted on these rows and no others, or it would describe a different
    population than the one each chunk resamples in.
    """

    names = [str(name) for name in chunk_data.pert_names]
    control = _resolve_control_element_mask(
        names, control_selector, infer_control_elements=False
    )
    labels = chunk_data.pert_id
    if isinstance(labels, IndexedDesignMatrix):
        indices = np.asarray(labels.indices, dtype=np.int64)
        active = (indices >= 0) & (np.asarray(labels.values) > 0)
        first = np.argmax(active, axis=1)
        codes = np.where(
            active.sum(axis=1) == 1, indices[np.arange(indices.shape[0]), first], -1
        )
    else:
        labels = np.asarray(labels)
        if labels.ndim == 2:
            binary = np.asarray(labels > 0)
            codes = np.where(binary.sum(axis=1) == 1, np.argmax(binary, axis=1), -1)
        else:
            codes = labels.astype(np.int64)
    keep = codes >= 0
    if not include_control_elements:
        keep = keep & ~control[np.maximum(codes, 0)]
    if chunk_data.cell_mask is not None:
        keep = keep & np.asarray(chunk_data.cell_mask, dtype=bool).reshape(-1)
    return keep


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run PerTurbo end-to-end on an AnnData/MuData file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to .h5ad or .h5mu file")
    parser.add_argument(
        "--pairs-to-test",
        default=None,
        help=(
            "Optional CSV/TSV/Parquet with columns element,gene. The run is unchanged - every "
            "pair is still fitted and tested - but a second effect table restricted to these "
            "pairs is written beside the transcriptome-wide one, with Benjamini-Hochberg "
            "recomputed within that family. Use it to obtain a cis-scale comparison and the "
            "transcriptome-wide analysis from a single run."
        ),
    )
    parser.add_argument("--out-dir", required=True, help="Directory to write outputs")
    parser.add_argument("--modality-key", default=None, help="MuData modality key (e.g., rna)")
    parser.add_argument(
        "--perturbation-key",
        default=None,
        help="Obs field for perturbation labels (required for low-MOI inputs).",
    )
    parser.add_argument(
        "--perturbation-modality-key",
        default=None,
        help="MuData modality key containing the perturbation matrix (high-MOI inputs).",
    )
    parser.add_argument(
        "--perturbation-layer",
        default=None,
        help="Layer in the perturbation modality containing the binarized matrix (defaults to .X).",
    )
    parser.add_argument(
        "--perturbation-element-varm-key",
        default=None,
        help=(
            "Optional key in perturbation modality .varm containing a perturbation-by-element "
            "binary map. When set, beta effects are fit at element level."
        ),
    )
    parser.add_argument(
        "--perturbation-element-names-uns-key",
        default=None,
        help=(
            "Optional key in perturbation modality .uns containing element names. "
            "Required when --perturbation-element-varm-key has no column labels."
        ),
    )
    parser.add_argument(
        "--control-substring",
        default=None,
        help="Substring identifying controls (required for low-MOI inputs).",
    )
    parser.add_argument(
        "--max-control-cells",
        type=int,
        default=10000,
        help="Maximum number of control cells to use (randomly subsampled).",
    )
    parser.add_argument(
        "--continuous-covariates",
        nargs="+",
        default=None,
        help="Continuous obs covariates to regress (auto log1p+zscore for count-like, otherwise zscore).",
    )
    parser.add_argument(
        "--batch-covariate",
        default=None,
        help="Optional categorical obs covariate (one-hot with most-frequent reference dropped).",
    )
    parser.add_argument(
        "--size-factor-key",
        default=None,
        help=(
            "Obs field containing precomputed size factors (already transformed; typically centered around zero). "
            "Must not be raw count/library-size data."
        ),
    )
    parser.add_argument(
        "--library-size-key",
        default=None,
        help=(
            "Obs field containing raw library-size/count data (for example UMI counts). "
            "This is transformed internally to centered log size factors."
        ),
    )
    parser.add_argument(
        "--size-factor-mode",
        default="observed",
        choices=VALID_CLI_SIZE_FACTOR_MODES,
        help=(
            "How to handle per-cell size factors: condition on observed values (observed, the "
            "default; --library-size-key names the column, otherwise each cell's total count over "
            "the analysed genes is used), infer latent size factors (infer; incompatible with the "
            "conditional randomization test), or fix them to zero (none)."
        ),
    )
    parser.add_argument(
        "--crt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Run the conditional randomization test against the stage-1 baseline, adding "
            "crt_p_value/crt_q_value/crt_z_value to element_effects.parquet. On by default; "
            "--no-crt fits the effects alone. It needs --size-factor-mode observed (or none), "
            "--likelihood nb, --num-factors 0, no --guide-random-effects, and one perturbation "
            "per cell: left to the default it steps aside with a message when the configuration "
            "is unsupported, while an explicit --crt reports the problem and stops."
        ),
    )
    parser.add_argument("--crt-num-resamples", type=int, default=999)
    parser.add_argument("--crt-seed", type=int, default=0)
    parser.add_argument(
        "--crt-gene-chunk-size",
        type=int,
        default=2000,
        help=(
            "Genes per inner CRT block. The score gather scales with this times the cells per "
            "target times the resample block, so it is the main memory knob."
        ),
    )
    parser.add_argument("--crt-max-gather-gib", type=float, default=8.0)
    parser.add_argument(
        "--crt-tail-families",
        nargs="*",
        default=["saddlepoint"],
        choices=list(CRT_ALL_TAIL_FAMILIES),
        help=(
            "Continuous nulls so p-values can resolve below the empirical floor of 1/(resamples+1). "
            "saddlepoint (the default) is evaluated in the kernel and, under --crt-mechanism "
            "propensity, uses the exact Bernoulli-sum CGF; skew_normal and student_t are fitted "
            "from resampled moments and need resamples. Pass with no values to skip them."
        ),
    )
    parser.add_argument(
        "--crt-only",
        action="store_true",
        help=(
            "Stop after stage one and the CRT: skip the stage-two effect fit. The element table "
            "then carries the CRT columns with missing effect estimates. This is the fast path when "
            "only the test is wanted (power calculations, calibration checks)."
        ),
    )
    parser.add_argument(
        "--crt-auto-moi-threshold",
        type=float,
        default=3.0,
        help=(
            "With --crt-pool auto, the screen is high MOI (all-cells pool) when the median guides per "
            "cell exceeds this, low MOI (control-anchored pool) otherwise. Default 3: a screen whose "
            "constructs carry two guides each still reads as one perturbation per cell, while a screen "
            "at a true high MOI (tens of guides per cell) does not."
        ),
    )
    parser.add_argument(
        "--crt-min-control-cells",
        type=int,
        default=1000,
        help=(
            "Warn when fewer cells than this carry nothing but control guides, or when they are under "
            "1%% of all cells. The pool choice does not depend on it; a thin control population makes "
            "the control-anchored null noisy and the all-cells null uninformative. Default 1000."
        ),
    )
    parser.add_argument(
        "--crt-test-control-elements",
        action="store_true",
        help=(
            "Also test the control elements themselves, against the same control pool. They are "
            "skipped by default because their cells are that pool, so a control element would be "
            "tested against a set containing itself: the test stays valid but is conservative. "
            "Turn it on to get p-values for a run's own negative controls as a calibration "
            "diagnostic; a clean calibration needs controls held out of the pool."
        ),
    )
    parser.add_argument(
        "--crt-pool",
        choices=("auto", "control-anchored", "all-cells"),
        default="auto",
        help=(
            "Cells a target is resampled against. 'control-anchored' (low MOI) tests each target "
            "inside the control cells plus its own cells, with the null fit on controls. 'all-cells' "
            "(high MOI) tests each element as a marginal association over every analysed cell, with "
            "the null fit on all cells; it needs the guide-to-element map "
            "(--perturbation-element-varm-key) and no control cells, and runs the exact "
            "Bernoulli-sum saddlepoint with no resamples. 'auto' picks all-cells when the element "
            "map is given, control-anchored otherwise."
        ),
    )
    parser.add_argument(
        "--crt-mechanism",
        choices=list(CRT_MECHANISMS),
        default="propensity",
        help=(
            "How a target's label is resampled inside its pool of controls plus its own cells: "
            "a stratified permutation holding the count fixed, or model-X Bernoulli draws at each "
            "cell's fitted selection probability (propensity)."
        ),
    )
    parser.add_argument(
        "--crt-saddlepoint-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Draw no resamples: fit the propensity model, compute observed scores, and evaluate the "
            "exact-CGF saddlepoint alone. On by default whenever the mechanism is propensity and "
            "the only tail family is saddlepoint (the production configuration); "
            "--no-crt-saddlepoint-only adds the resampling pass. Requires those two settings; "
            "crt_p_value stays missing without resamples."
        ),
    )
    parser.add_argument(
        "--crt-screen-p-value",
        type=float,
        default=0.05,
        help="Evaluate the saddlepoint only where the Pearson III screen is at or below this p-value.",
    )
    parser.add_argument(
        "--crt-two-sided",
        choices=["symmetric", "equal-tail"],
        default="equal-tail",
        help=(
            "Two-sided convention for the saddlepoint p-value: 'symmetric' is P(|S| >= |observed|); "
            "'equal-tail' is twice the tail on the observed side, which rejects each tail equally under a skewed null."
        ),
    )
    parser.add_argument(
        "--crt-baseline-step-tolerance",
        type=float,
        default=DEFAULT_NEWTON_STEP_TOLERANCE,
        help=(
            "Maximum per-gene Newton step, in nats, from the stage-1 baseline to the control-cell "
            "null mode. The CRT reuses that baseline rather than refitting it, so this is the only "
            "check that it is fit well enough for the score test to mean what it says."
        ),
    )
    parser.add_argument(
        "--crt-allow-unconverged-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Warn instead of failing when the stage-1 baseline is not at the control-cell null mode. "
            "On by default: after the polish the guard's percentile is dominated by genes with almost "
            "no control counts, and every production run passes it. --no-crt-allow-unconverged-baseline "
            "makes the guard fatal."
        ),
    )
    parser.add_argument(
        "--crt-polish-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before the CRT, move the stage-1 nuisance coefficients onto the control-cell null "
            "mode by Fisher scoring at the stage-1 dispersion. The resampling null is exact either "
            "way; this restores the efficiency of the score statistic when stage one sits off the "
            "mode, as it does with a batch covariate, and it is what makes the test insensitive to "
            "how long stage one trained. On by default; --no-crt-polish-baseline tests the "
            "stage-one coefficients as they came out of SVI."
        ),
    )
    parser.add_argument("--gene-name-key", default=None, help="Var field for gene names")
    parser.add_argument(
        "--clip-gene-expression-percentile",
        type=float,
        default=100.0,
        help=(
            "Percentile used to derive per-gene outlier thresholds across the full RNA matrix. "
            "How those thresholds are applied is controlled by --gene-outlier-action and "
            "--winsorize-gene-expression-outliers. "
            "Use 100 to disable threshold-based outlier handling. "
            "--likelihood=censored_nb requires a value strictly below 100."
        ),
    )
    parser.add_argument(
        "--gene-outlier-action",
        default="none",
        choices=("none", "filter_cells"),
        help=(
            "How to apply the gene-wise outlier thresholds: do nothing, or drop cells with many outlier genes."
        ),
    )
    parser.add_argument(
        "--gene-outlier-threshold-floor",
        type=int,
        default=2,
        help="Minimum per-gene outlier threshold after percentile estimation.",
    )
    parser.add_argument(
        "--winsorize-gene-expression-outliers",
        action="store_true",
        help="Winsorize remaining gene counts down to their per-gene outlier thresholds after optional cell filtering.",
    )
    parser.add_argument(
        "--outlier-cell-min-genes",
        type=int,
        default=0,
        help=(
            "When --gene-outlier-action includes filter_cells, drop cells with at least this many "
            "outlier genes. Set >0 to enable a burden-based cell filter."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="JAX device spec: platform ('cpu'/'gpu') or indexed device ('gpu:1').",
    )
    parser.add_argument("--prior", default="normal", help="Prior for beta (normal or cauchy)")
    parser.add_argument(
        "--likelihood",
        default="negbin",
        choices=("nb", "negbin", "censored_nb", "lognormal_nb", "mixture_nb"),
        help="Observation model (nb/negbin, censored_nb, lognormal_nb, or mixture_nb)",
    )
    parser.add_argument(
        "--guide-effect-strategy",
        default="shared",
        choices=("shared", "relative"),
        help="How guides targeting the same element share information during stage-2 fitting.",
    )
    parser.add_argument(
        "--guide-activity-mode",
        default="always_on",
        choices=("always_on", "absolute"),
        help="Guide activity model for stage-2 fitting.",
    )
    parser.add_argument(
        "--guide-random-effects",
        action="store_true",
        help=(
            "Enable guide-level random effects with hierarchical gene-wise shrinkage in stage-1, "
            "and carry that calibration into stage-2."
        ),
    )
    parser.add_argument(
        "--fit-perturbation-dispersion",
        action="store_true",
        help=(
            "Fit a non-negative perturbation-by-gene excess inverse dispersion in stage 2. "
            "Currently supported for low-MOI negative-binomial fits only."
        ),
    )
    parser.add_argument(
        "--perturbation-dispersion-prior-rate",
        type=float,
        default=10.0,
        help="Rate of the exponential prior on stage-2 excess inverse dispersion.",
    )
    parser.add_argument(
        "--save-model-params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write a light, simulation-ready fitted model parameter bundle into --out-dir by default. "
            "The bundle can be loaded with PERTURBO.load(--out-dir) while the original input data remain available."
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Shared raw SVI step count for both control and perturbation-effect fits.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Shared dataset passes for both control and perturbation-effect fits.",
    )
    parser.add_argument(
        "--num-steps-control",
        type=int,
        default=None,
        help="Stage-specific raw SVI step count for the control fit.",
    )
    parser.add_argument(
        "--num-epochs-control",
        type=int,
        default=None,
        help="Stage-specific dataset passes for the control fit.",
    )
    parser.add_argument(
        "--num-steps-betas",
        type=int,
        default=None,
        help="Stage-specific raw SVI step count for perturbation-effect fitting.",
    )
    parser.add_argument(
        "--num-epochs-betas",
        type=int,
        default=None,
        help="Stage-specific dataset passes for perturbation-effect fitting.",
    )
    parser.add_argument(
        "--num-particles",
        type=int,
        default=1,
        help="Number of ELBO particles for Monte Carlo integration.",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.01,
        help=(
            "Adam learning rate for both SVI stages. Convergence depends on the data, effect sizes, and "
            "minibatch size. Compare longer fits before interpreting effect magnitudes; the default "
            "500 steps can underestimate strong knockdowns."
        ),
    )
    parser.add_argument(
        "--num-factors",
        type=int,
        default=0,
        help="If >0, fit this many latent factors (shared across controls and betas)",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=0,
        help=(
            "If >0, subsample cells per step for minibatch SVI "
            "(applies to control and beta fits unless overridden)."
        ),
    )
    parser.add_argument(
        "--minibatch-size-control",
        type=int,
        default=0,
        help="If >0, subsample cells per step for control minibatch SVI",
    )
    parser.add_argument(
        "--minibatch-size-betas",
        type=int,
        default=0,
        help="If >0, subsample cells per step for beta minibatch SVI",
    )
    parser.add_argument(
        "--gene-chunk-size",
        type=int,
        default=0,
        help=(
            "Genes loaded and fit together in stage two; keeps every cell and perturbation predictor. "
            "0 uses perturbation chunks for mutually exclusive assignments and 256-gene blocks "
            "when co-occurring perturbations would otherwise be split. Requires plain NB with fixed "
            "size factors, shared guide effects, and no latent factors or extra dispersion. "
            "Control fitting remains capped by --max-control-cells."
        ),
    )
    parser.add_argument(
        "--perturbation-chunk-size",
        type=int,
        default=0,
        help=(
            "Maximum perturbations per chunk when fitting betas. "
            "If 0, chunk size is chosen automatically from --max-chunk-size. "
            "Co-occurring assignments use gene blocks to retain the joint model."
        ),
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=50000,
        help="Maximum number of cells per perturbation chunk",
    )
    parser.add_argument(
        "--backed",
        action="store_true",
        help="Enable disk-backed reads for AnnData/MuData inputs",
    )
    parser.add_argument(
        "--use-observed-size-factors",
        action="store_true",
        help=(
            "Treat size factors as observed instead of latent. "
            "Backward-compatible alias for --size-factor-mode=observed."
        ),
    )
    parser.add_argument(
        "--propagate-baseline-uncertainty",
        action="store_true",
        help=(
            "In stage-2 beta fitting, marginalize over stage-1 baseline uncertainty "
            "for beta_0 and theta instead of conditioning on their point estimates."
        ),
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_false",
        dest="progress",
        help="Disable tqdm progress bar during SVI",
    )
    parser.set_defaults(progress=True)
    parser.add_argument(
        "--progress-chunk-size",
        type=int,
        default=100,
        help="Number of SVI steps per progress bar update",
    )
    parser.add_argument("--single-frame", action="store_true", help="Output long dataframe")
    args = parser.parse_args(argv)
    args.likelihood = _normalize_likelihood_name(args.likelihood)
    size_factor_mode = _resolve_cli_size_factor_mode(
        size_factor_mode=args.size_factor_mode,
        use_observed_size_factors=args.use_observed_size_factors,
    )
    use_observed_size_factors = size_factor_mode in {"observed", "none"}
    if args.size_factor_key is not None and args.library_size_key is not None:
        raise ValueError(
            "--size-factor-key and --library-size-key are mutually exclusive. "
            "Provide only one."
        )
    if size_factor_mode == "none":
        size_factor_key_for_loading = None
        library_size_key_for_loading = None
    else:
        size_factor_key_for_loading = args.size_factor_key
        library_size_key_for_loading = args.library_size_key

    _backed_note = " (backed mode — data stays on disk)" if args.backed else " (loading fully into memory — may take several minutes for large files)"
    print(f"[perturbo] Reading input{_backed_note}: {args.input}")
    _t0 = time.monotonic()
    data = _load_from_path_with_backing(args.input, backed=args.backed)
    _load_elapsed = time.monotonic() - _t0
    _loaded_adata = data if hasattr(data, "n_obs") else next(iter(data.mod.values())) if hasattr(data, "mod") else data
    print(
        f"[perturbo] Input loaded in {_load_elapsed:.1f}s: "
        f"{getattr(_loaded_adata, 'n_obs', '?')} cells × {getattr(_loaded_adata, 'n_vars', '?')} genes"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[perturbo] Outputs will be written to: {out_dir}")

    continuous_covariates = _dedupe_preserve_order(args.continuous_covariates)
    batch_covariate = None if args.batch_covariate in (None, "", "None") else str(args.batch_covariate)
    if batch_covariate is not None and batch_covariate in continuous_covariates:
        raise ValueError(
            f"Covariate column '{batch_covariate}' was provided as both continuous and batch. "
            "Select it in only one role."
        )

    if args.perturbation_modality_key is None and args.perturbation_key is None:
        raise ValueError(
            "--perturbation-key is required unless --perturbation-modality-key is provided."
        )
    if args.perturbation_modality_key is None and args.control_substring is None:
        raise ValueError("--control-substring is required for low-MOI inputs.")
    if args.perturbation_element_varm_key is not None and args.perturbation_modality_key is None:
        raise ValueError("--perturbation-element-varm-key requires --perturbation-modality-key.")
    if args.perturbation_element_names_uns_key is not None and args.perturbation_element_varm_key is None:
        raise ValueError(
            "--perturbation-element-names-uns-key requires --perturbation-element-varm-key."
        )
    guide_effect_strategy, guide_activity_mode = _validate_guide_strategy(
        args.guide_effect_strategy,
        args.guide_activity_mode,
    )
    if (
        guide_effect_strategy != "shared" or guide_activity_mode != "always_on"
    ) and args.perturbation_element_varm_key is None:
        raise ValueError(
            "Guide-sharing modes require --perturbation-element-varm-key so guides can be matched to elements."
        )
    if (
        guide_effect_strategy != "shared" or guide_activity_mode != "always_on"
    ) and args.perturbation_modality_key is None:
        raise ValueError(
            "Guide-sharing modes require --perturbation-modality-key with a guide-level perturbation matrix."
        )
    if guide_activity_mode == "absolute" and args.likelihood not in {"negbin", "censored_nb"}:
        raise ValueError(
            "--guide-activity-mode=absolute is currently only compatible with negative-binomial likelihoods."
        )

    clip_percentile = _validate_clip_percentile(args.clip_gene_expression_percentile)
    if _is_censored_model_name(args.likelihood) and clip_percentile >= 100.0:
        raise ValueError("--likelihood=censored_nb requires --clip-gene-expression-percentile < 100.")

    if args.crt_only and args.crt is False:
        raise ValueError("--crt-only cannot be combined with --no-crt.")
    # Tri-state: True asked for it, False refused it, None is the default. On the
    # default an unsupported configuration steps aside rather than failing a run
    # that never mentioned the test; asked for explicitly, it is an error.
    if args.crt_saddlepoint_only is None:
        # The production configuration draws no resamples; any other mechanism or
        # tail family needs them.
        args.crt_saddlepoint_only = args.crt_mechanism == "propensity" and list(args.crt_tail_families) == [
            CRT_SADDLEPOINT_FAMILY
        ]
    crt_requested = args.crt is True or bool(args.crt_only)
    if args.crt is False:
        args.crt = False
    else:
        unsupported = _crt_configuration_problem(args, size_factor_mode)
        if unsupported is None:
            args.crt = True
        elif crt_requested:
            raise ValueError(unsupported)
        else:
            print(f"[perturbo] Skipping the conditional randomization test: {unsupported}")
            args.crt = False
    if args.crt:
        # Deliberately before any fitting: a CRT run sits behind a full stage-1
        # fit, so surfacing an unsupported flag afterwards would cost a training
        # run to discover.
        crt_pool = args.crt_pool
        if crt_pool == "auto":
            # Decide on the design the data actually has, not on how the file was
            # written: high MOI when cells carry more than a few guides each, low MOI
            # otherwise. The control population does not enter the decision - a thin
            # one is reported as a warning either way, because it makes the
            # control-anchored null noisy and the all-cells null uninformative.
            moi = measure_realized_moi(
                data,
                perturbation_modality_key=args.perturbation_modality_key,
                perturbation_layer=args.perturbation_layer,
                perturbation_key=args.perturbation_key,
                control_substring=args.control_substring,
                perturbation_element_varm_key=args.perturbation_element_varm_key,
                perturbation_element_names_uns_key=args.perturbation_element_names_uns_key,
                modality_key=args.modality_key,
            )
            high_moi = moi["median_guides_per_cell"] > args.crt_auto_moi_threshold
            crt_pool = "all-cells" if high_moi else "control-anchored"
            print(
                f"[perturbo] --crt-pool auto resolved to '{crt_pool}': median {moi['median_guides_per_cell']:.2f} "
                f"guides per cell ({'>' if high_moi else '<='} {args.crt_auto_moi_threshold:g}), measured from "
                f"{moi['source']}. Pass --crt-pool to decide explicitly."
            )
            if high_moi and args.perturbation_element_varm_key is None:
                raise ValueError(
                    "--crt-pool auto chose the all-cells pool (high MOI) but there is no guide-to-element "
                    "map to run it with; pass --perturbation-element-varm-key so each element can be tested "
                    "as a marginal association over all cells, or --crt-pool control-anchored."
                )
        else:
            moi = measure_realized_moi(
                data,
                perturbation_modality_key=args.perturbation_modality_key,
                perturbation_layer=args.perturbation_layer,
                perturbation_key=args.perturbation_key,
                control_substring=args.control_substring,
                perturbation_element_varm_key=args.perturbation_element_varm_key,
                perturbation_element_names_uns_key=args.perturbation_element_names_uns_key,
                modality_key=args.modality_key,
            )
        n_controls = int(moi["n_control_cells"])
        control_fraction = n_controls / max(float(moi["n_cells"]), 1.0)
        print(
            f"[perturbo] CRT pool '{crt_pool}': {n_controls:,} of {int(moi['n_cells']):,} cells "
            f"({100 * control_fraction:.2f}%) carry only control guides."
        )
        if n_controls < args.crt_min_control_cells or control_fraction < 0.01:
            warnings.warn(
                f"Only {n_controls:,} cells ({100 * control_fraction:.2f}%) carry nothing but control guides "
                f"(threshold {args.crt_min_control_cells:,} cells or 1%). "
                + (
                    "The control-anchored null is fit on these cells alone, so it will be noisy."
                    if crt_pool == "control-anchored"
                    else "Control elements are the only calibration negatives the all-cells test has."
                ),
                stacklevel=1,
            )
        if crt_pool == "all-cells":
            if args.perturbation_element_varm_key is None:
                raise ValueError("--crt-pool all-cells needs the guide-to-element map (--perturbation-element-varm-key).")
            if not args.crt_saddlepoint_only or args.crt_mechanism != "propensity":
                raise ValueError(
                    "--crt-pool all-cells runs the propensity saddlepoint with no resamples; pass "
                    "--crt-mechanism propensity --crt-tail-families saddlepoint --crt-saddlepoint-only."
                )
        validate_crt_config(
            likelihood=args.likelihood,
            size_factor_mode=size_factor_mode,
            num_factors=args.num_factors,
            guide_random_effects=args.guide_random_effects,
            retain_guide_structure=args.perturbation_element_varm_key is not None and crt_pool == "control-anchored",
        )
        if args.crt_num_resamples < 1:
            raise ValueError("--crt-num-resamples must be >= 1.")
        if args.crt_saddlepoint_only:
            if list(args.crt_tail_families) != [CRT_SADDLEPOINT_FAMILY]:
                raise ValueError("--crt-saddlepoint-only requires --crt-tail-families saddlepoint alone.")
            if args.crt_mechanism != "propensity":
                raise ValueError("--crt-saddlepoint-only requires --crt-mechanism propensity.")
        if not 0.0 < args.crt_screen_p_value <= 1.0:
            raise ValueError("--crt-screen-p-value must lie in (0, 1].")
        if args.crt_gene_chunk_size < 1:
            raise ValueError("--crt-gene-chunk-size must be >= 1.")
        if args.control_substring is None and crt_pool == "control-anchored":
            raise ValueError(
                "--crt requires --control-substring: the test resamples each perturbation's label "
                "within a pool of control cells, so it needs to know which cells those are."
            )

    _validate_cli_input_keys(
        data,
        perturbation_key=args.perturbation_key,
        modality_key=args.modality_key,
        perturbation_modality_key=args.perturbation_modality_key,
        perturbation_layer=args.perturbation_layer,
        perturbation_element_varm_key=args.perturbation_element_varm_key,
        perturbation_element_names_uns_key=args.perturbation_element_names_uns_key,
        size_factor_key=size_factor_key_for_loading,
        library_size_key=library_size_key_for_loading,
        gene_name_key=args.gene_name_key,
        continuous_covariates=continuous_covariates,
        batch_covariate=batch_covariate,
    )

    chunks: list[_PerturbationChunk] | None = None
    all_perturbation_names = None
    all_guide_names: list[str] | None = None
    analysis_gene_names = None
    # The control-anchored propensity CRT shares one screen-wide selection
    # model across perturbation chunks; these are the analysis rows it is fitted
    # on beside the control pool, captured while the chunk membership is still
    # in hand.
    crt_shares_selection_model = bool(
        args.crt and crt_pool == "control-anchored" and args.crt_mechanism == "propensity"
    )
    crt_tested_rows: np.ndarray | None = None
    threshold_floor = _validate_gene_outlier_threshold_floor(args.gene_outlier_threshold_floor)
    apply_filter_cells = args.gene_outlier_action == "filter_cells"
    apply_winsorize = args.winsorize_gene_expression_outliers
    if args.perturbation_chunk_size < 0:
        raise ValueError("--perturbation-chunk-size must be >= 0.")
    if args.gene_chunk_size < 0:
        raise ValueError("--gene-chunk-size must be >= 0.")
    if args.max_chunk_size < 1:
        raise ValueError("--max-chunk-size must be >= 1.")
    if args.outlier_cell_min_genes < 0:
        raise ValueError("--outlier-cell-min-genes must be >= 0.")
    if apply_filter_cells and args.outlier_cell_min_genes < 1:
        raise ValueError("--outlier-cell-min-genes must be >= 1 when --gene-outlier-action includes filter_cells.")
    if not apply_filter_cells and args.outlier_cell_min_genes != 0:
        raise ValueError("--outlier-cell-min-genes requires --gene-outlier-action=filter_cells or both.")
    analysis_adata = _resolve_adata(data, args.modality_key)
    _row_chunk_size = BACKED_ROW_CHUNK_SIZE if args.backed else None
    gene_clip_thresholds = None
    if (apply_filter_cells or apply_winsorize) and clip_percentile is not None and clip_percentile < 100.0:
        _chunked_note = ", row-chunked for backed data" if _row_chunk_size is not None else ""
        print(
            "[perturbo] Computing per-gene outlier thresholds at "
            f"the {clip_percentile:g}th percentile{_chunked_note}..."
        )
        gene_clip_thresholds = _compute_gene_outlier_thresholds(
            analysis_adata.X,
            percentile=clip_percentile,
            threshold_floor=threshold_floor,
            row_chunk_size=_row_chunk_size,
        )
        print("[perturbo] Per-gene outlier thresholds ready.")
    elif apply_filter_cells:
        raise ValueError("--gene-outlier-action=filter_cells requires --clip-gene-expression-percentile < 100.")

    cell_keep_mask = None
    if apply_filter_cells:
        outlier_gene_counts = _count_gene_outliers_per_cell(
            analysis_adata.X, gene_clip_thresholds, row_chunk_size=_row_chunk_size
        )
        cell_keep_mask = outlier_gene_counts < args.outlier_cell_min_genes
        removed_cells = int(np.count_nonzero(~cell_keep_mask))
        kept_cells = int(np.count_nonzero(cell_keep_mask))
        print(
            "[perturbo] Cell outlier filter removed "
            f"{removed_cells} cells with >= {args.outlier_cell_min_genes} outlier genes; "
            f"{kept_cells} cells remain."
        )
        if kept_cells == 0:
            raise ValueError("Cell outlier filter removed all analysis cells.")
        analysis_adata_for_workflow = analysis_adata[cell_keep_mask]
    else:
        analysis_adata_for_workflow = analysis_adata

    should_chunk = False
    retain_guide_structure = args.perturbation_element_varm_key is not None
    n_analysis_cells = getattr(analysis_adata_for_workflow, "n_obs", None)
    if args.pairs_to_test is not None:
        # Say this before the expensive work, not after. On the perturbo branch this
        # flag restricted the fit itself, so a caller who has not read the release
        # notes would otherwise wait out a transcriptome-wide run expecting a small one.
        print(
            "[perturbo] --pairs-to-test given: the fit and the test still cover every "
            "perturbation and gene. The file selects the rows of a second table, "
            "element_effects_requested_pairs.parquet, whose q-values are corrected "
            "within that set alone."
        )
    if args.perturbation_chunk_size > 0:
        should_chunk = True
    elif n_analysis_cells is not None and n_analysis_cells > args.max_chunk_size:
        should_chunk = True
        print(
            "[perturbo] Auto chunking enabled because the analysis dataset has "
            f"{n_analysis_cells} cells; chunk cell counts will be capped at "
            f"--max-chunk-size={args.max_chunk_size}."
        )

    gene_chunk_size = args.gene_chunk_size or None
    if gene_chunk_size is not None:
        problem = _gene_chunk_configuration_problem(args, size_factor_mode)
        if problem is not None:
            raise ValueError(problem)
        should_chunk = False

    if should_chunk:
        analysis_gene_names = _extract_gene_names(analysis_adata_for_workflow, args.gene_name_key)
        if args.perturbation_modality_key is not None:
            pert_adata = _resolve_perturbation_modality(data, args.perturbation_modality_key)
            pert_adata = pert_adata[analysis_adata_for_workflow.obs_names]
            all_guide_names = _extract_pert_names(pert_adata)
            pert_matrix = _get_layer_matrix(pert_adata, args.perturbation_layer)
            if args.perturbation_element_varm_key is not None:
                element_mapping, element_names = _load_perturbation_element_mapping(
                    data,
                    perturbation_modality_key=args.perturbation_modality_key,
                    perturbation_element_varm_key=args.perturbation_element_varm_key,
                    perturbation_element_names_uns_key=args.perturbation_element_names_uns_key,
                    preserve_sparse=True,
                )
                pert_matrix = _group_perturbation_matrix_by_element(pert_matrix, element_mapping, preserve_sparse=True)
                all_perturbation_names = element_names
            else:
                all_perturbation_names = _extract_pert_names(pert_adata)
            # Dropping a co-occurring predictor changes the fitted model. For
            # overlapping assignments block genes instead, keeping all columns.
            active_per_cell = np.asarray((pert_matrix > 0).sum(axis=1)).reshape(-1)
            if np.any(active_per_cell > 1):
                problem = _gene_chunk_configuration_problem(args, size_factor_mode)
                if problem is not None:
                    raise ValueError(
                        "Perturbation chunks would omit co-occurring predictors, and " + problem
                        + ". Use a supported gene-block configuration, or raise --max-chunk-size "
                        "to fit all cells jointly with --perturbation-chunk-size 0."
                    )
                gene_chunk_size = 256
                print("[perturbo] Co-occurring perturbations detected: using 256-gene blocks with every predictor retained.")
            else:
                membership = _build_matrix_membership(
                    pert_matrix,
                    num_perts=len(all_perturbation_names),
                )
                chunks = _construct_perturbation_chunks(
                    all_perturbation_names,
                    membership,
                    max_chunk_size=args.max_chunk_size,
                    max_perturbations_per_chunk=args.perturbation_chunk_size or None,
                )
                if crt_shares_selection_model:
                    crt_tested_rows = _crt_tested_cell_rows(
                        all_perturbation_names,
                        membership,
                        control_selector=args.control_substring,
                        include_control_elements=args.crt_test_control_elements,
                    )
                del membership
            del pert_matrix, pert_adata
            if args.perturbation_element_varm_key is not None:
                del element_mapping
        else:
            pert_series = analysis_adata_for_workflow.obs[args.perturbation_key].astype(str)
            all_perturbation_names = [str(x) for x in pd.Categorical(pert_series).categories.tolist()]
            all_guide_names = list(all_perturbation_names)
            membership = _build_obs_membership(pert_series, all_perturbation_names)
            chunks = _construct_perturbation_chunks(
                all_perturbation_names,
                membership,
                max_chunk_size=args.max_chunk_size,
                max_perturbations_per_chunk=args.perturbation_chunk_size or None,
            )
            if crt_shares_selection_model:
                crt_tested_rows = _crt_tested_cell_rows(
                    all_perturbation_names,
                    membership,
                    control_selector=args.control_substring,
                    include_control_elements=args.crt_test_control_elements,
                )

    controls_loaded = load_controls(
        data,
        perturbation_key=args.perturbation_key,
        control_selector=args.control_substring,
        modality_key=args.modality_key,
        perturbation_modality_key=args.perturbation_modality_key,
        perturbation_layer=args.perturbation_layer,
        perturbation_element_varm_key=args.perturbation_element_varm_key,
        perturbation_element_names_uns_key=args.perturbation_element_names_uns_key,
        max_control_cells=args.max_control_cells,
        size_factor_key=size_factor_key_for_loading,
        library_size_key=library_size_key_for_loading,
        gene_name_key=args.gene_name_key,
        device=args.device,
        cell_keep_mask=cell_keep_mask,
        clip_gene_expression_percentile=clip_percentile,
        winsorize_gene_expression=apply_winsorize,
        gene_outlier_threshold_floor=threshold_floor,
        gene_clip_thresholds=gene_clip_thresholds if apply_winsorize else None,
        continuous_covariates=continuous_covariates,
        batch_covariate=batch_covariate,
        return_covariate_transform_state=True,
        infer_control_guides=bool(args.guide_random_effects and args.perturbation_modality_key is not None),
        only_control_guides=bool(args.crt and crt_pool == "control-anchored" and args.perturbation_modality_key is not None),
    )
    if isinstance(controls_loaded, tuple):
        controls, covariate_transform_state = controls_loaded
    else:
        controls = controls_loaded
        covariate_transform_state = None
    if size_factor_mode == "none":
        controls.size_factors = _fixed_zero_size_factors(controls.counts)
    if args.num_particles < 1:
        raise ValueError("--num-particles must be >= 1.")
    svi_cfg = SVIConfig(step_size=args.step_size, num_particles=args.num_particles)
    minibatch_control = args.minibatch_size_control or args.minibatch_size or None
    minibatch_betas = args.minibatch_size_betas or args.minibatch_size or None
    if gene_chunk_size is not None and minibatch_betas is None:
        minibatch_betas = min(1024, int(n_analysis_cells))
        print(f"[perturbo] Gene blocks use {minibatch_betas} cells per SVI step; override with --minibatch-size-betas.")
    schedule = resolve_training_schedule(
        shared_steps=args.num_steps,
        shared_epochs=args.num_epochs,
        control_steps=args.num_steps_control,
        beta_steps=args.num_steps_betas,
        control_epochs=args.num_epochs_control,
        beta_epochs=args.num_epochs_betas,
        default_control_steps=500,
        default_beta_steps=500,
    )
    if gene_chunk_size is not None and not args.crt_only:
        planned_steps = schedule.resolve_stage_steps(
            stage="beta", num_cells=int(n_analysis_cells), minibatch_size=minibatch_betas,
        )
        expected_passes = planned_steps * min(int(minibatch_betas), int(n_analysis_cells)) / int(n_analysis_cells)
        print(
            f"[perturbo] Each gene block: {planned_steps} SVI steps, {expected_passes:.2f} expected cell passes. "
            "Set --num-epochs-betas to specify training coverage."
        )

    if args.num_factors < 0:
        raise ValueError("--num-factors must be >= 0.")
    num_factors = args.num_factors or None

    control_fit = fit_control(
        controls,
        num_steps=schedule.resolve_stage_steps(
            stage="control",
            num_cells=int(controls.counts.shape[0]),
            minibatch_size=minibatch_control,
        ),
        prior=args.prior,
        svi_config=svi_cfg,
        model_name=args.likelihood,
        num_factors=num_factors,
        use_observed_size_factors=use_observed_size_factors,
        count_censoring_percentile=clip_percentile if _is_censored_model_name(args.likelihood) else None,
        minibatch_size=minibatch_control,
        progress=args.progress,
        progress_chunk_size=args.progress_chunk_size,
        guide_random_effects=args.guide_random_effects,
    )
    _save_loss_plot(
        control_fit.losses,
        out_dir / "control_loss_curve.png",
        title="Control loss curve",
    )
    print(f"[perturbo] Wrote {out_dir / 'control_loss_curve.png'}")
    control_fit_path = out_dir / "control_fit.npz"
    save_array_bundle(control_fit_path, control_fit_arrays(control_fit))
    print(f"[perturbo] Wrote {control_fit_path}")

    crt_shared_propensity: np.ndarray | None = None

    def _fit_crt_shared_selection_model(
        target_covariates: np.ndarray | None, num_target: int
    ) -> np.ndarray | None:
        """The selection slopes every control-anchored CRT chunk will share.

        Fitted once here, over the control pool and every cell that will be
        tested, rather than inside each chunk: a chunk sees only its own
        targets' cells, so an in-chunk fit would estimate the covariate slopes
        from whichever neighbours the chunk-size flag happened to group
        together, and a target's p-value would move with that flag.

        The design is assembled as ``[intercept, covariates]`` with the control
        pool first, exactly the parametrisation and column order
        ``prepare_low_moi_design`` gives each chunk, because the coefficients
        are interpreted in those columns downstream.
        """

        if not crt_shares_selection_model or num_target == 0:
            return None
        num_control = int(np.asarray(controls.counts).shape[0])
        blocks = [np.ones((num_control + num_target, 1), dtype=np.float32)]
        control_covariates = (
            None if controls.covariates is None else np.asarray(controls.covariates, dtype=np.float32)
        )
        if control_covariates is not None:
            if target_covariates is None:
                raise RuntimeError(
                    "The control pool carries covariates but the tested cells do not; the CRT's "
                    "nuisance designs would not share a parametrisation."
                )
            blocks.append(
                np.concatenate(
                    [control_covariates, np.asarray(target_covariates, dtype=np.float32)], axis=0
                )
            )
        targeting = np.concatenate(
            [np.zeros(num_control, dtype=np.float32), np.ones(num_target, dtype=np.float32)]
        )
        coefficients = fit_shared_propensity_coefficients(
            np.concatenate(blocks, axis=1), targeting
        )
        print(
            f"[perturbo] CRT: shared selection model fit over {num_control + num_target:,} cells "
            f"({num_target:,} targeting); every chunk reuses these covariate slopes."
        )
        return coefficients

    if crt_shares_selection_model and crt_tested_rows is not None:
        _tested_covariates = None
        if covariate_transform_state is not None:
            _tested_obs = analysis_adata_for_workflow.obs.iloc[crt_tested_rows]
            _tested_covariates, _ = apply_covariate_transform(_tested_obs, covariate_transform_state)
        crt_shared_propensity = _fit_crt_shared_selection_model(
            _tested_covariates, int(crt_tested_rows.size)
        )

    crt_baseline = None
    crt_accumulator: CRTAccumulator | None = None
    if args.crt and crt_pool == "control-anchored" and gene_chunk_size is None:
        print("[perturbo] Preparing CRT baseline from the stage-1 fit...")
        crt_baseline = prepare_crt_baseline(
            controls,
            control_fit,
            step_tolerance=args.crt_baseline_step_tolerance,
            strict=not args.crt_allow_unconverged_baseline,
            polish=args.crt_polish_baseline,
        )
        if crt_baseline.pre_polish_check is not None:
            print(f"[perturbo] CRT baseline before polishing: {crt_baseline.pre_polish_check.describe()}")
        print(f"[perturbo] CRT baseline: {crt_baseline.null_check.describe()}")
    def _run_crt_on_chunk(
        chunk_source: PerTurboData,
        accumulator: CRTAccumulator,
        *,
        baseline_source=None,
        control_source=None,
    ) -> None:
        """Test one chunk's perturbations, skipping the control elements.

        Control cells are already the pool, so testing them as targets from the
        chunk side would stack the same biological cells twice as distinct rows.
        """

        control_names = [
            name
            for name, is_control in zip(
                list(chunk_source.pert_names),
                _resolve_control_element_mask(
                    list(chunk_source.pert_names),
                    args.control_substring,
                    infer_control_elements=False,
                ),
                strict=True,
            )
            if is_control
        ]
        if args.crt_test_control_elements:
            testable = chunk_source
            if control_names:
                print(
                    f"[perturbo] CRT: testing {len(control_names)} control element(s) as targets too; each is"
                    " tested against a pool that contains its own cells, so its p-value is conservative."
                )
        else:
            testable = exclude_targets(chunk_source, control_names)
            if testable is None:
                print("[perturbo] CRT: chunk holds only control elements; nothing to test.")
                return
            if control_names:
                print(f"[perturbo] CRT: skipping {len(control_names)} control element(s) as targets.")
        accumulator.absorb(
            run_crt_for_chunk(
                crt_baseline if baseline_source is None else baseline_source,
                testable,
                control_data=controls if control_source is None else control_source,
                num_resamples=args.crt_num_resamples,
                seed=args.crt_seed,
                gene_chunk_size=args.crt_gene_chunk_size,
                tail_families=args.crt_tail_families,
                jax_max_gather_gib=args.crt_max_gather_gib,
                resampling_mechanism=args.crt_mechanism,
                saddlepoint_only=args.crt_saddlepoint_only,
                saddlepoint_screen_p_value=args.crt_screen_p_value,
                saddlepoint_two_sided=args.crt_two_sided,
                shared_propensity_coefficients=crt_shared_propensity,
            )
        )

    def _nan_beta_fit(n_perts: int, n_genes: int):
        """A stage-two result with every estimate missing, for --crt-only runs."""
        shape = (int(n_perts), int(n_genes))
        return BetaFit(
            posterior_mean=jnp.full(shape, jnp.nan, dtype=jnp.float32),
            posterior_scale=jnp.full(shape, jnp.nan, dtype=jnp.float32),
            z_values=jnp.full(shape, jnp.nan, dtype=jnp.float32),
            losses=jnp.zeros((0,), dtype=jnp.float32),
            svi_result=None,
        )

    all_cells_propensity = None

    def _run_all_cells_crt(
        full_data: PerTurboData,
        *,
        block_control_fit=None,
        accumulator=None,
    ) -> CRTAccumulator:
        """The high-MOI CRT over every analysed cell at once, independent of stage-two chunking."""
        nonlocal all_cells_propensity
        started = time.perf_counter()
        print("[perturbo] Preparing the all-cells CRT baseline: stage-1 fit polished over every analysed cell...")
        baseline = prepare_crt_baseline(
            full_data,
            control_fit if block_control_fit is None else block_control_fit,
            step_tolerance=args.crt_baseline_step_tolerance,
            strict=not args.crt_allow_unconverged_baseline,
            polish=True,
        )
        print(f"[perturbo] CRT baseline: {baseline.null_check.describe()}")
        if gene_chunk_size is not None and all_cells_propensity is None:
            all_cells_propensity = prepare_all_cells_propensity(baseline, full_data)
        if accumulator is None:
            accumulator = CRTAccumulator(
                element_names=tuple(str(name) for name in full_data.pert_names),
                gene_names=tuple(str(name) for name in full_data.gene_names),
                tail_families=tuple(args.crt_tail_families),
            )
        accumulator.absorb(
            run_crt_all_cells(
                baseline,
                full_data,
                gene_chunk_size=args.crt_gene_chunk_size,
                screen_p_value=args.crt_screen_p_value,
                two_sided=args.crt_two_sided,
                **({"propensity_fit": all_cells_propensity} if all_cells_propensity is not None else {}),
            )
        )
        print(f"[perturbo] CRT (all-cells pool) complete in {time.perf_counter() - started:.0f}s")
        return accumulator

    def _load_all_analysis_cells(*, selected_gene_indices=None, full_panel_library_sizes=None) -> PerTurboData:
        return load_analysis_cells(
            data,
            perturbation_key=args.perturbation_key,
            modality_key=args.modality_key,
            perturbation_modality_key=args.perturbation_modality_key,
            perturbation_layer=args.perturbation_layer,
            perturbation_element_varm_key=args.perturbation_element_varm_key,
            perturbation_element_names_uns_key=args.perturbation_element_names_uns_key,
            size_factor_key=size_factor_key_for_loading,
            library_size_key=library_size_key_for_loading,
            gene_name_key=args.gene_name_key,
            device=args.device,
            cell_keep_mask=cell_keep_mask,
            clip_gene_expression_percentile=clip_percentile,
            winsorize_gene_expression=apply_winsorize,
            gene_outlier_threshold_floor=threshold_floor,
            gene_clip_thresholds=gene_clip_thresholds if apply_winsorize else None,
            continuous_covariates=continuous_covariates,
            batch_covariate=batch_covariate,
            covariate_transform_state=covariate_transform_state,
            retain_guide_structure=retain_guide_structure and (
                gene_chunk_size is None
                or (args.crt and crt_pool == "all-cells" and all_cells_propensity is None)
            ),
            library_size_center_log_mean=controls.library_size_center_log_mean,
            selected_gene_indices=selected_gene_indices,
            full_panel_library_sizes=full_panel_library_sizes,
            indexed_perturbation_design=gene_chunk_size is not None,
        )

    chunk_losses: list[jnp.ndarray] = []
    guide_efficiency_frames: list[pd.DataFrame] = []
    chunk_guide_posteriors: dict[str, np.ndarray] = {}
    chunk_guide_posterior_seen: dict[str, np.ndarray] = {}
    analysis_data: PerTurboData | None = None
    stage_two_skipped = False
    if gene_chunk_size is not None:
        analysis_gene_names = _extract_gene_names(analysis_adata, args.gene_name_key)
        n_genes = len(analysis_gene_names)
        full_panel_library_sizes = np.zeros(int(n_analysis_cells)) if size_factor_mode == "none" else None
        if size_factor_mode == "observed" and size_factor_key_for_loading is None and library_size_key_for_loading is None:
            selected_rows = (
                np.arange(analysis_adata.n_obs) if cell_keep_mask is None else np.flatnonzero(cell_keep_mask)
            )
            print("[perturbo] Computing full-panel library sizes once before reading gene blocks...")
            if not apply_winsorize or gene_clip_thresholds is None:
                full_panel_library_sizes = _sum_matrix_rows_bounded(analysis_adata.X, selected_rows)
            else:
                full_panel_library_sizes = np.empty(selected_rows.size, dtype=np.float64)
                for start in range(0, selected_rows.size, BACKED_ROW_CHUNK_SIZE):
                    stop = min(start + BACKED_ROW_CHUNK_SIZE, selected_rows.size)
                    block = analysis_adata.X[selected_rows[start:stop], :]
                    if hasattr(block, "toarray"):
                        block = block.toarray()
                    full_panel_library_sizes[start:stop] = np.minimum(
                        np.asarray(block), gene_clip_thresholds[None, :]
                    ).sum(axis=1, dtype=np.float64)
        gene_posteriors: dict[str, np.ndarray] = {}
        for start in range(0, n_genes, gene_chunk_size):
            stop = min(start + gene_chunk_size, n_genes)
            gene_slice = slice(start, stop)
            print(f"[perturbo] Gene block {start + 1}–{stop}/{n_genes}: every cell and perturbation predictor retained")
            analysis_data = _load_all_analysis_cells(
                selected_gene_indices=gene_slice,
                full_panel_library_sizes=full_panel_library_sizes,
            )
            if size_factor_mode == "none":
                analysis_data.size_factors = _fixed_zero_size_factors(analysis_data.counts)
            all_perturbation_names = list(analysis_data.pert_names)
            block_control_fit = subset_control_fit_genes(control_fit, gene_slice)
            if args.crt:
                if crt_accumulator is None:
                    crt_accumulator = CRTAccumulator(
                        element_names=tuple(all_perturbation_names),
                        gene_names=tuple(analysis_gene_names),
                        tail_families=tuple(args.crt_tail_families),
                    )
                if crt_pool == "all-cells":
                    _run_all_cells_crt(
                        analysis_data, block_control_fit=block_control_fit, accumulator=crt_accumulator
                    )
                else:
                    block_controls = replace(
                        controls,
                        counts=controls.counts[:, gene_slice],
                        gene_names=analysis_gene_names[gene_slice],
                    )
                    block_baseline = prepare_crt_baseline(
                        block_controls,
                        block_control_fit,
                        step_tolerance=args.crt_baseline_step_tolerance,
                        strict=not args.crt_allow_unconverged_baseline,
                        polish=args.crt_polish_baseline,
                    )
                    if crt_shares_selection_model and crt_shared_propensity is None:
                        # Gene blocks keep every cell and every perturbation, so
                        # this sees the whole screen already; it is fitted once
                        # and reused, the selection model having no gene axis.
                        _tested = _crt_tested_cell_mask(
                            analysis_data,
                            control_selector=args.control_substring,
                            include_control_elements=args.crt_test_control_elements,
                        )
                        crt_shared_propensity = _fit_crt_shared_selection_model(
                            None
                            if analysis_data.covariates is None
                            else np.asarray(analysis_data.covariates, dtype=np.float32)[_tested],
                            int(np.count_nonzero(_tested)),
                        )
                    _run_crt_on_chunk(
                        analysis_data, crt_accumulator,
                        baseline_source=block_baseline, control_source=block_controls,
                    )
            if analysis_data.guide_matrix is not None:
                analysis_data = replace(analysis_data, guide_matrix=None, guide_to_element=None)
            if args.crt_only:
                block_fit = _nan_beta_fit(len(all_perturbation_names), stop - start)
            else:
                block_fit = fit_perturbation_effects(
                    analysis_data,
                    block_control_fit,
                    num_steps=schedule.resolve_stage_steps(
                        stage="beta", num_cells=int(analysis_data.counts.shape[0]), minibatch_size=minibatch_betas,
                    ),
                    prior=args.prior,
                    svi_config=svi_cfg,
                    model_name=args.likelihood,
                    use_observed_size_factors=True,
                    minibatch_size=minibatch_betas,
                    progress=args.progress,
                    progress_chunk_size=args.progress_chunk_size,
                )
            for field in ("posterior_mean", "posterior_scale", "z_values", *_GUIDE_POSTERIOR_FIELDS):
                value = getattr(block_fit, field)
                if value is None:
                    continue
                values = np.asarray(value)
                if values.ndim != 2 or values.shape[1] != stop - start:
                    raise ValueError(f"Gene-block posterior {field!r} must have its gene axis last.")
                if field not in gene_posteriors:
                    gene_posteriors[field] = np.empty((values.shape[0], n_genes), dtype=values.dtype)
                gene_posteriors[field][:, gene_slice] = values
            if block_fit.losses.size:
                chunk_losses.append(block_fit.losses)
            # A combined posterior has no single SVI state. Retaining a block's
            # optimizer state would also keep its device buffers resident.
            del block_fit, block_control_fit
        beta_fit = BetaFit(
            **gene_posteriors,
            losses=jnp.concatenate(chunk_losses) if chunk_losses else jnp.array([]),
            svi_result=None,
        )
        stage_two_skipped = True
        if args.crt_only:
            print("[perturbo] --crt-only: stage two skipped; effect estimates are missing in the element table.")
    if chunks is not None and args.crt and crt_pool == "all-cells":
        # The all-cells pool needs every cell at once, whatever the stage-two
        # chunking does: load the full analysis data, test, and release it.
        full_data = _load_all_analysis_cells()
        if size_factor_mode == "none":
            full_data.size_factors = _fixed_zero_size_factors(full_data.counts)
        crt_accumulator = _run_all_cells_crt(full_data)
        all_perturbation_names = list(full_data.pert_names)
        analysis_gene_names = list(full_data.gene_names)
        del full_data
    if chunks is not None and args.crt_only:
        # No stage two: the CRT already ran (all cells) or runs chunk by chunk
        # below without any effect fit.
        n_genes = len(analysis_gene_names)
        n_perts = len(all_perturbation_names)
        if crt_pool == "control-anchored":
            crt_accumulator = CRTAccumulator(
                element_names=tuple(str(name) for name in all_perturbation_names),
                gene_names=tuple(str(name) for name in analysis_gene_names),
                tail_families=tuple(args.crt_tail_families),
            )
            for chunk_i, chunk_info in enumerate(chunks):
                chunk_data = load_analysis_cells(
                    data,
                    perturbation_key=args.perturbation_key,
                    modality_key=args.modality_key,
                    perturbation_modality_key=args.perturbation_modality_key,
                    perturbation_layer=args.perturbation_layer,
                    perturbation_element_varm_key=args.perturbation_element_varm_key,
                    perturbation_element_names_uns_key=args.perturbation_element_names_uns_key,
                    size_factor_key=size_factor_key_for_loading,
                    library_size_key=library_size_key_for_loading,
                    gene_name_key=args.gene_name_key,
                    device=args.device,
                    selected_perturbations=chunk_info.pert_names,
                    cell_keep_mask=cell_keep_mask,
                    clip_gene_expression_percentile=clip_percentile,
                    winsorize_gene_expression=apply_winsorize,
                    gene_outlier_threshold_floor=threshold_floor,
                    gene_clip_thresholds=gene_clip_thresholds if apply_winsorize else None,
                    continuous_covariates=continuous_covariates,
                    batch_covariate=batch_covariate,
                    covariate_transform_state=covariate_transform_state,
                    retain_guide_structure=retain_guide_structure,
                    library_size_center_log_mean=controls.library_size_center_log_mean,
                )
                if size_factor_mode == "none":
                    chunk_data.size_factors = _fixed_zero_size_factors(chunk_data.counts)
                crt_started = time.perf_counter()
                _run_crt_on_chunk(chunk_data, crt_accumulator)
                print(f"[perturbo] CRT chunk {chunk_i + 1}/{len(chunks)} in {time.perf_counter() - crt_started:.0f}s")
        beta_fit = _nan_beta_fit(n_perts, n_genes)
        # Stage two and the unchunked branch below are both skipped: the CRT
        # has already run over every chunk (or over all cells), and falling
        # into the unchunked branch would reload every cell and run it again.
        stage_two_skipped = True
        analysis_data = None
        print("[perturbo] --crt-only: stage two skipped; effect estimates are missing in the element table.")
    if stage_two_skipped:
        pass
    elif chunks is not None:
        if all_perturbation_names is None or analysis_gene_names is None:
            raise RuntimeError("Chunk metadata was not initialized.")
        n_genes = len(analysis_gene_names)
        n_perts = len(all_perturbation_names)
        if crt_baseline is not None:
            crt_accumulator = CRTAccumulator(
                element_names=tuple(str(name) for name in all_perturbation_names),
                gene_names=tuple(str(name) for name in analysis_gene_names),
                tail_families=tuple(args.crt_tail_families),
            )
        posterior_mean = np.zeros((n_perts, n_genes), dtype=np.float32)
        posterior_scale = np.zeros((n_perts, n_genes), dtype=np.float32)
        z_values = np.zeros((n_perts, n_genes), dtype=np.float32)
        dispersion_excess_inverse = (
            np.zeros((n_perts, n_genes), dtype=np.float32) if args.fit_perturbation_dispersion else None
        )
        last_state = None
        # Keep the standard (non-guide-aware) chunked path on one reusable
        # shape. Guide-aware chunks retain their current variable-shape path
        # until their guide buffers can be handled separately.
        padded_chunk_cell_capacity = max(int(chunk.cell_indices.size) for chunk in chunks)
        padded_chunk_pert_capacity = max(len(chunk.pert_names) for chunk in chunks)
        chunk_runner_cache: dict[str, _ReusableSVIRunner] | None = (
            {} if not retain_guide_structure and minibatch_betas is None else None
        )
        for chunk_i, chunk_info in enumerate(chunks):
            chunk_names = chunk_info.pert_names
            chunk_indices = chunk_info.pert_indices
            print(
                f"[perturbo] Chunk {chunk_i + 1}/{len(chunks)}: "
                f"{len(chunk_names)} perturbations, {chunk_info.cell_indices.size} cells"
            )
            print(
                "[perturbo] Processing perturbation chunk indices "
                f"{int(chunk_indices.min())}..{int(chunk_indices.max())}"
            )
            chunk_data = load_analysis_cells(
                data,
                perturbation_key=args.perturbation_key,
                modality_key=args.modality_key,
                perturbation_modality_key=args.perturbation_modality_key,
                perturbation_layer=args.perturbation_layer,
                perturbation_element_varm_key=args.perturbation_element_varm_key,
                perturbation_element_names_uns_key=args.perturbation_element_names_uns_key,
                size_factor_key=size_factor_key_for_loading,
                library_size_key=library_size_key_for_loading,
                gene_name_key=args.gene_name_key,
                device=args.device,
                selected_perturbations=chunk_names,
                cell_keep_mask=cell_keep_mask,
                clip_gene_expression_percentile=clip_percentile,
                winsorize_gene_expression=apply_winsorize,
                gene_outlier_threshold_floor=threshold_floor,
                gene_clip_thresholds=gene_clip_thresholds if apply_winsorize else None,
                continuous_covariates=continuous_covariates,
                batch_covariate=batch_covariate,
                covariate_transform_state=covariate_transform_state,
                retain_guide_structure=retain_guide_structure,
                library_size_center_log_mean=controls.library_size_center_log_mean,
            )
            if size_factor_mode == "none":
                chunk_data.size_factors = _fixed_zero_size_factors(chunk_data.counts)
            fit_data = chunk_data
            if chunk_runner_cache is not None:
                fit_data = _pad_cortado_data_for_chunk(
                    chunk_data,
                    cell_capacity=padded_chunk_cell_capacity,
                    pert_capacity=padded_chunk_pert_capacity,
                )
            chunk_fit = fit_perturbation_effects(
                fit_data,
                control_fit,
                num_steps=schedule.resolve_stage_steps(
                    stage="beta",
                    num_cells=int(chunk_data.counts.shape[0]),
                    minibatch_size=minibatch_betas,
                ),
                prior=args.prior,
                svi_config=svi_cfg,
                model_name=args.likelihood,
                num_factors=num_factors,
                propagate_baseline_uncertainty=args.propagate_baseline_uncertainty,
                use_observed_size_factors=use_observed_size_factors,
                count_censoring_percentile=clip_percentile if _is_censored_model_name(args.likelihood) else None,
                minibatch_size=minibatch_betas,
                progress=args.progress,
                progress_chunk_size=args.progress_chunk_size,
                guide_effect_strategy=guide_effect_strategy,
                guide_activity_mode=guide_activity_mode,
                guide_random_effects=args.guide_random_effects,
                fit_perturbation_dispersion=args.fit_perturbation_dispersion,
                perturbation_dispersion_prior_rate=args.perturbation_dispersion_prior_rate,
                _runner_cache=chunk_runner_cache,
            )
            if crt_accumulator is not None and crt_baseline is not None:
                # Control-anchored only: the all-cells pool was tested once on
                # the full data before this loop.
                crt_started = time.perf_counter()
                # The unpadded chunk, deliberately: padding rows are all-zero
                # cells the model drops via cell_mask, and they would otherwise
                # join the pooled null as legitimate zero-count observations.
                _run_crt_on_chunk(chunk_data, crt_accumulator)
                print(f"[perturbo] CRT chunk {chunk_i + 1}/{len(chunks)} in {time.perf_counter() - crt_started:.0f}s")
            n_chunk_perts = len(chunk_names)
            posterior_mean[chunk_indices] = np.asarray(chunk_fit.posterior_mean)[:n_chunk_perts]
            posterior_scale[chunk_indices] = np.asarray(chunk_fit.posterior_scale)[:n_chunk_perts]
            z_values[chunk_indices] = np.asarray(chunk_fit.z_values)[:n_chunk_perts]
            if dispersion_excess_inverse is not None and chunk_fit.dispersion_excess_inverse is not None:
                dispersion_excess_inverse[chunk_indices] = np.asarray(chunk_fit.dispersion_excess_inverse)[:n_chunk_perts]
            last_state = chunk_fit.svi_result
            if chunk_fit.losses.size > 0:
                chunk_losses.append(chunk_fit.losses)
            if retain_guide_structure:
                if all_guide_names is None:
                    raise RuntimeError("Global guide names were not initialized for chunked fitting.")
                _merge_chunk_guide_posteriors(
                    chunk_guide_posteriors,
                    chunk_guide_posterior_seen,
                    chunk_fit,
                    chunk_guide_names=list(chunk_data.guide_names or []),
                    global_guide_names=all_guide_names,
                )
            if guide_effect_strategy == "relative" and chunk_fit.guide_relative_efficiency_mean is not None:
                mapping = np.asarray(chunk_data.guide_to_element, dtype=np.float32)
                guide_names = list(chunk_data.guide_names or [])
                parent_elements: list[str] = []
                for row in np.asarray(mapping, dtype=bool):
                    idx = np.flatnonzero(row)
                    if idx.size == 0:
                        parent_elements.append("")
                    elif idx.size == 1:
                        parent_elements.append(str(chunk_data.pert_names[int(idx[0])]))
                    else:
                        parent_elements.append("|".join(str(chunk_data.pert_names[int(i)]) for i in idx.tolist()))
                guide_efficiency_frames.append(
                    build_guide_efficiency_df(
                        method="perturbo",
                        guide_names=guide_names,
                        guide_parent_elements=parent_elements,
                        guide_efficiency_mean=np.asarray(chunk_fit.guide_relative_efficiency_mean),
                        gene_names=list(analysis_gene_names),
                    )
                )

        combined_guide_posteriors = _finalize_chunk_guide_posteriors(
            chunk_guide_posteriors,
            chunk_guide_posterior_seen,
            global_guide_names=list(all_guide_names or []),
        )
        beta_fit = BetaFit(
            posterior_mean=jnp.asarray(posterior_mean),
            posterior_scale=jnp.asarray(posterior_scale),
            z_values=jnp.asarray(z_values),
            losses=jnp.concatenate(chunk_losses) if chunk_losses else jnp.array([]),
            svi_result=last_state,
            dispersion_excess_inverse=(
                None if dispersion_excess_inverse is None else jnp.asarray(dispersion_excess_inverse)
            ),
            **combined_guide_posteriors,
        )
        # full = PerTurboData(
        #     counts=jnp.empty((0, len(analysis_gene_names))),
        #     pert_id=jnp.empty((0,), dtype=jnp.int32),
        #     pert_names=all_perturbation_names,
        #     gene_names=analysis_gene_names,
        #     size_factors=None,
        # )
    else:
        analysis_data = load_analysis_cells(
            data,
            perturbation_key=args.perturbation_key,
            modality_key=args.modality_key,
            perturbation_modality_key=args.perturbation_modality_key,
            perturbation_layer=args.perturbation_layer,
            perturbation_element_varm_key=args.perturbation_element_varm_key,
            perturbation_element_names_uns_key=args.perturbation_element_names_uns_key,
            size_factor_key=size_factor_key_for_loading,
            library_size_key=library_size_key_for_loading,
            gene_name_key=args.gene_name_key,
            device=args.device,
            cell_keep_mask=cell_keep_mask,
            clip_gene_expression_percentile=clip_percentile,
            winsorize_gene_expression=apply_winsorize,
            gene_outlier_threshold_floor=threshold_floor,
            gene_clip_thresholds=gene_clip_thresholds if apply_winsorize else None,
            continuous_covariates=continuous_covariates,
            batch_covariate=batch_covariate,
            covariate_transform_state=covariate_transform_state,
            retain_guide_structure=retain_guide_structure,
            library_size_center_log_mean=controls.library_size_center_log_mean,
        )
        if size_factor_mode == "none":
            analysis_data.size_factors = _fixed_zero_size_factors(analysis_data.counts)
        all_perturbation_names = analysis_data.pert_names
        analysis_gene_names = analysis_data.gene_names
        if args.crt and crt_pool == "all-cells":
            crt_accumulator = _run_all_cells_crt(analysis_data)
        elif crt_baseline is not None:
            crt_accumulator = CRTAccumulator(
                element_names=tuple(str(name) for name in all_perturbation_names),
                gene_names=tuple(str(name) for name in analysis_gene_names),
                tail_families=tuple(args.crt_tail_families),
            )
            if crt_shares_selection_model:
                # Nothing is chunked here, so "over the screen" and "inside the
                # chunk" are the same cells and the fit could equally have
                # happened downstream. It runs here anyway, so that a screen
                # gets the same selection model however it was decomposed.
                _tested = _crt_tested_cell_mask(
                    analysis_data,
                    control_selector=args.control_substring,
                    include_control_elements=args.crt_test_control_elements,
                )
                crt_shared_propensity = _fit_crt_shared_selection_model(
                    None
                    if analysis_data.covariates is None
                    else np.asarray(analysis_data.covariates, dtype=np.float32)[_tested],
                    int(np.count_nonzero(_tested)),
                )
            crt_started = time.perf_counter()
            _run_crt_on_chunk(analysis_data, crt_accumulator)
            print(f"[perturbo] CRT complete in {time.perf_counter() - crt_started:.0f}s")
        if args.crt_only:
            beta_fit = _nan_beta_fit(len(all_perturbation_names), len(analysis_gene_names))
            print("[perturbo] --crt-only: stage two skipped; effect estimates are missing in the element table.")
        else:
            beta_fit = fit_perturbation_effects(
            analysis_data,
            control_fit,
            num_steps=schedule.resolve_stage_steps(
                stage="beta",
                num_cells=int(analysis_data.counts.shape[0]),
                minibatch_size=minibatch_betas,
            ),
            prior=args.prior,
            svi_config=svi_cfg,
            model_name=args.likelihood,
            num_factors=num_factors,
            propagate_baseline_uncertainty=args.propagate_baseline_uncertainty,
            use_observed_size_factors=use_observed_size_factors,
            count_censoring_percentile=clip_percentile if _is_censored_model_name(args.likelihood) else None,
            minibatch_size=minibatch_betas,
            progress=args.progress,
            progress_chunk_size=args.progress_chunk_size,
            guide_effect_strategy=guide_effect_strategy,
            guide_activity_mode=guide_activity_mode,
            guide_random_effects=args.guide_random_effects,
            fit_perturbation_dispersion=args.fit_perturbation_dispersion,
            perturbation_dispersion_prior_rate=args.perturbation_dispersion_prior_rate,
        )
    if all_perturbation_names is None or analysis_gene_names is None:
        raise RuntimeError("Missing perturbation/gene names for element-level output.")
    control_mask = _resolve_control_element_mask(
        list(all_perturbation_names),
        args.control_substring,
        infer_control_elements=bool(args.guide_random_effects and args.perturbation_modality_key is not None),
    )
    null_z = np.asarray(beta_fit.z_values)[control_mask].reshape(-1)
    if null_z.size == 0:
        print("[perturbo] Warning: no control-matched element rows for empirical null; empirical_p_value set to NA.")
        null_z_values: np.ndarray | None = None
    else:
        null_z_values = null_z
    crt_columns = None
    if crt_accumulator is not None:
        # Benjamini-Hochberg happens here and only here: it has to see every
        # hypothesis at once, and each chunk was by construction only part of
        # the family.
        crt_columns = crt_accumulator.finalize()
        if crt_accumulator.num_multi_assignment_cells_dropped:
            print(
                f"[perturbo] CRT: {crt_accumulator.num_multi_assignment_cells_dropped:,} analysed cells carried more "
                "than one perturbation and were set aside by the control-anchored test."
            )
        # A small record of how the test was configured and what it measured, so a
        # pipeline can see which pool ran without parsing the log.
        crt_metadata = {
            "pool": crt_pool,
            "pool_requested": args.crt_pool,
            "auto_moi_threshold": args.crt_auto_moi_threshold,
            "min_control_cells_warning": args.crt_min_control_cells,
            "measured": {key: value for key, value in moi.items()},
            "mechanism": args.crt_mechanism,
            "tail_families": list(args.crt_tail_families),
            "saddlepoint_only": bool(args.crt_saddlepoint_only),
            "tested_control_elements": bool(args.crt_test_control_elements),
            "two_sided": args.crt_two_sided,
            "multi_assignment_cells_set_aside": int(crt_accumulator.num_multi_assignment_cells_dropped),
        }
        (out_dir / "crt_metadata.json").write_text(json.dumps(crt_metadata, indent=2, default=str))
        tested = int(crt_accumulator.tested.sum())
        primary = (
            f"crt_{CRT_SADDLEPOINT_FAMILY}_p_value" if args.crt_saddlepoint_only else "crt_p_value"
        )
        finite = np.isfinite(crt_columns[primary])
        print(
            f"[perturbo] CRT ({args.crt_mechanism}): {tested}/{len(all_perturbation_names)} elements "
            f"tested, {int(finite.sum())} pairs with a p-value."
        )
        if args.crt_saddlepoint_only:
            print("[perturbo] CRT: saddlepoint-only run; no resamples were drawn and crt_p_value is missing.")
        else:
            # The empirical p-value floors at 1/(resamples+1); over a large
            # screen that floor can sit above the Benjamini-Hochberg cutoff
            # entirely, so report each family beside it rather than letting a
            # q of zero discoveries read as "nothing is there".
            floor = 1.0 / (args.crt_num_resamples + 1)
            at_floor = int(np.count_nonzero(crt_columns["crt_p_value"][finite] <= floor + 1e-12))
            print(
                f"[perturbo] CRT empirical: {int(np.count_nonzero(crt_columns['crt_q_value'][finite] < 0.05))} "
                f"at q<0.05; {at_floor} pairs tied at the p floor of {floor:.3g}."
            )
        for family in crt_accumulator.tail_families:
            q = crt_columns[f"crt_{family}_q_value"]
            valid = crt_columns[f"crt_{family}_valid"] > 0
            print(
                f"[perturbo] CRT {family}: {int(np.count_nonzero(q[finite] < 0.05))} at q<0.05, "
                f"{int(np.count_nonzero(valid & finite))}/{int(finite.sum())} fits valid."
            )
    element_effects = build_standard_element_effects_df(
        method="perturbo",
        effect_loc=np.asarray(beta_fit.posterior_mean),
        effect_scale=np.asarray(beta_fit.posterior_scale),
        element_names=list(all_perturbation_names),
        gene_names=list(analysis_gene_names),
        null_z_values=null_z_values,
        extra_columns=crt_columns,
    )
    element_effects_path = out_dir / "element_effects.parquet"
    element_effects.to_parquet(element_effects_path, index=False)
    print(f"[perturbo] Wrote {element_effects_path}")

    if args.pairs_to_test is not None:
        # One fit and one test, two tables. The restricted table exists because
        # the multiple-testing family differs, not because the analysis does.
        requested_pairs = load_pairs_to_test(args.pairs_to_test)
        restricted = restrict_effects_to_pairs(element_effects, requested_pairs)
        missing = len(requested_pairs) - len(restricted)
        restricted_path = out_dir / "element_effects_requested_pairs.parquet"
        restricted.to_parquet(restricted_path, index=False)
        print(
            f"[perturbo] Wrote {restricted_path}: {len(restricted):,} of {len(requested_pairs):,} requested pairs"
            + (f" ({missing:,} not present in the analysed grid)" if missing else "")
            + "; q-values recomputed within this family."
        )

    if guide_effect_strategy == "relative":
        guide_efficiency_df: pd.DataFrame | None = None
        if chunks is not None and guide_efficiency_frames:
            guide_efficiency_df = pd.concat(guide_efficiency_frames, ignore_index=True)
        elif (
            analysis_data is not None
            and beta_fit.guide_relative_efficiency_mean is not None
            and analysis_data.guide_to_element is not None
            and analysis_data.guide_names is not None
        ):
            mapping = np.asarray(analysis_data.guide_to_element, dtype=np.float32)
            guide_names = list(analysis_data.guide_names)
            parent_elements: list[str] = []
            for row in np.asarray(mapping, dtype=bool):
                idx = np.flatnonzero(row)
                if idx.size == 0:
                    parent_elements.append("")
                elif idx.size == 1:
                    parent_elements.append(str(analysis_data.pert_names[int(idx[0])]))
                else:
                    parent_elements.append("|".join(str(analysis_data.pert_names[int(i)]) for i in idx.tolist()))
            guide_efficiency_df = build_guide_efficiency_df(
                method="perturbo",
                guide_names=guide_names,
                guide_parent_elements=parent_elements,
                guide_efficiency_mean=np.asarray(beta_fit.guide_relative_efficiency_mean),
                gene_names=list(analysis_gene_names),
            )
        if guide_efficiency_df is None:
            print("[perturbo] Warning: guide efficiency strategy is relative but no estimates were available.")
        else:
            guide_efficiency_path = out_dir / "guide_efficiency.parquet"
            guide_efficiency_df.to_parquet(guide_efficiency_path, index=False)
            print(f"[perturbo] Wrote {guide_efficiency_path}")

    if covariate_transform_state is not None:
        covariate_metadata_path = out_dir / "covariate_metadata.json"
        metadata = asdict(covariate_transform_state)
        metadata["n_covariate_features"] = len(covariate_transform_state.feature_names)
        metadata["covariate_names"] = list(covariate_transform_state.feature_names)
        metadata["continuous_covariates_requested"] = continuous_covariates
        metadata["batch_covariate_requested"] = batch_covariate
        covariate_metadata_path.write_text(json.dumps(metadata, indent=2))
        print(f"[perturbo] Wrote {covariate_metadata_path}")

    if (chunks is not None or gene_chunk_size is not None) and chunk_losses:
        _save_multi_loss_plot(
            chunk_losses,
            out_dir / "beta_loss_curve.png",
            title="Beta loss curve (chunked)",
            labels=[f"chunk {i + 1}" for i in range(len(chunk_losses))],
        )
        print(f"[perturbo] Wrote {out_dir / 'beta_loss_curve.png'}")
    elif beta_fit.losses.size > 0:
        _save_loss_plot(
            beta_fit.losses,
            out_dir / "beta_loss_curve.png",
            title="Beta loss curve",
        )
        print(f"[perturbo] Wrote {out_dir / 'beta_loss_curve.png'}")
    else:
        print("[perturbo] Skipping beta loss plot (no losses recorded).")

    if args.save_model_params:
        source_path = Path(args.input).expanduser().resolve(strict=False)
        if args.perturbation_modality_key is not None:
            if hasattr(data, "mod") and args.perturbation_modality_key in data.mod:
                pert_source = _resolve_perturbation_modality(data, args.perturbation_modality_key)
                guide_names = _extract_pert_names(pert_source)
            else:
                if analysis_data is not None and analysis_data.guide_names is not None:
                    guide_names = list(analysis_data.guide_names)
                else:
                    guide_names = list(all_perturbation_names)
            perturbation_modality_for_setup = args.perturbation_modality_key
        else:
            guide_names = list(all_perturbation_names)
            perturbation_modality_for_setup = "grna"
        setup = MuDataSetup(
            rna_modality=args.modality_key or "rna",
            perturbation_modality=perturbation_modality_for_setup,
            perturbation_layer=args.perturbation_layer,
            batch_key=batch_covariate,
            library_size_key=library_size_key_for_loading,
            size_factor_key=size_factor_key_for_loading,
            continuous_covariates_keys=list(continuous_covariates or []),
            gene_by_element_key=None,
            guide_by_element_key=args.perturbation_element_varm_key,
            rna_element_uns_key=None,
            guide_element_uns_key=args.perturbation_element_names_uns_key,
            gene_name_key=args.gene_name_key,
            control_substring=args.control_substring,
            size_factor_mode=size_factor_mode,
        )
        guide_posteriors = {k: v for k, v in _guide_posterior_arrays(beta_fit).items() if v is not None}
        guide_efficacy = _guide_efficacy_for_cli(
            beta_fit,
            guide_effect_strategy=guide_effect_strategy,
            n_guides=len(guide_names),
        )
        saved_size_factors, size_factor_provenance = _bundle_size_factors(
            analysis_adata,
            selected_row_indices=np.flatnonzero(cell_keep_mask) if cell_keep_mask is not None else None,
            size_factor_mode=size_factor_mode,
            size_factor_key=size_factor_key_for_loading,
            library_size_key=library_size_key_for_loading,
            library_size_center_log_mean=controls.library_size_center_log_mean,
            gene_clip_thresholds=gene_clip_thresholds,
            winsorize_gene_expression=apply_winsorize,
        )
        metadata = {
            "likelihood": args.likelihood,
            "effect_prior_dist": args.prior,
            "efficiency_mode": guide_effect_strategy,
            "guide_effect_strategy": guide_effect_strategy,
            "guide_activity_mode": guide_activity_mode,
            "n_factors": num_factors,
            "clip_gene_expression_percentile": clip_percentile,
            "winsorize_gene_expression": apply_winsorize,
            "gene_outlier_threshold_floor": threshold_floor,
            "count_censoring_percentile": clip_percentile if _is_censored_model_name(args.likelihood) else None,
            "guide_random_effects": bool(args.guide_random_effects),
            "fit_perturbation_dispersion": bool(args.fit_perturbation_dispersion),
            "perturbation_dispersion_prior_rate": float(args.perturbation_dispersion_prior_rate),
            "fit_guide_efficacy": None,
            "library_size_center_log_mean": controls.library_size_center_log_mean,
            "size_factor_mode": size_factor_mode,
            "size_factor_provenance": size_factor_provenance,
            "setup": setup.to_json_dict(),
            "svi_config": asdict(svi_cfg),
            "covariate_transform_state": (
                asdict(covariate_transform_state) if covariate_transform_state is not None else None
            ),
            "source": {
                "path": str(source_path),
                "kind": "h5mu" if str(args.input).endswith(".h5mu") else "h5ad",
                "backed": bool(args.backed),
            },
            "input_mode": "high_moi" if args.perturbation_modality_key is not None else "low_moi",
            "perturbation_key": args.perturbation_key,
            "gene_names": list(analysis_gene_names),
            "perturbation_names": list(all_perturbation_names),
            "guide_names": list(guide_names),
            "cli_args": {
                "modality_key": args.modality_key,
                "perturbation_key": args.perturbation_key,
                "perturbation_modality_key": args.perturbation_modality_key,
                "perturbation_layer": args.perturbation_layer,
                "perturbation_element_varm_key": args.perturbation_element_varm_key,
                "perturbation_element_names_uns_key": args.perturbation_element_names_uns_key,
                "control_substring": args.control_substring,
                "crt_pool": (crt_pool if args.crt else None),
                "batch_covariate": batch_covariate,
                "size_factor_key": size_factor_key_for_loading,
                "library_size_key": library_size_key_for_loading,
                "size_factor_mode": size_factor_mode,
                "gene_name_key": args.gene_name_key,
                "gene_outlier_action": args.gene_outlier_action,
                "outlier_cell_min_genes": args.outlier_cell_min_genes,
                "max_control_cells": args.max_control_cells,
                "max_chunk_size": args.max_chunk_size,
                "perturbation_chunk_size": args.perturbation_chunk_size,
                "gene_chunk_size": gene_chunk_size,
            },
        }
        save_light_fit_bundle(
            out_dir,
            metadata=metadata,
            control_arrays=control_fit_arrays(control_fit),
            beta_arrays=_beta_fit_arrays(beta_fit),
            guide_posteriors=guide_posteriors or None,
            guide_efficacy=guide_efficacy,
            size_factors=saved_size_factors,
            cell_keep_indices=np.flatnonzero(cell_keep_mask) if cell_keep_mask is not None else None,
        )
        print(f"[perturbo] Wrote simulation-ready model parameter bundle to {out_dir}")



if __name__ == "__main__":
    main()
