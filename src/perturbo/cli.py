"""Command-line entrypoints for perturbo."""

from __future__ import annotations

from dataclasses import asdict
import json
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
    _resolve_adata,
    _resolve_cli_size_factor_mode,
    _resolve_control_element_mask,
    _resolve_perturbation_modality,
    _save_loss_plot,
    _save_multi_loss_plot,
    _validate_cli_input_keys,
    _validate_clip_percentile,
    _validate_gene_outlier_threshold_floor,
    _validate_guide_strategy,
    fit_control,
    fit_perturbation_effects,
    load_analysis_cells,
    load_controls,
)
from .io import MuDataSetup, control_fit_arrays, save_array_bundle, save_light_fit_bundle
from .results import build_guide_efficiency_df, build_standard_element_effects_df
from .training_schedule import resolve_training_schedule


def _beta_fit_arrays(beta_fit: BetaFit) -> dict[str, Any]:
    return {
        "posterior_mean": beta_fit.posterior_mean,
        "posterior_scale": beta_fit.posterior_scale,
        "z_values": beta_fit.z_values,
        "losses": beta_fit.losses,
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


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run PerTurbo end-to-end on an AnnData/MuData file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to .h5ad or .h5mu file")
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
        default="infer",
        choices=VALID_CLI_SIZE_FACTOR_MODES,
        help=(
            "How to handle per-cell size factors: infer latent size factors (infer), "
            "condition on observed/computed values (observed), or fix size factors to zero (none)."
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
    parser.add_argument("--step-size", type=float, default=0.003, help="SVI step size / learning rate")
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
        "--perturbation-chunk-size",
        type=int,
        default=0,
        help=(
            "Maximum perturbations per chunk when fitting betas. "
            "If 0, chunk size is chosen automatically from --max-chunk-size."
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
    analysis_gene_names = None
    threshold_floor = _validate_gene_outlier_threshold_floor(args.gene_outlier_threshold_floor)
    apply_filter_cells = args.gene_outlier_action == "filter_cells"
    apply_winsorize = args.winsorize_gene_expression_outliers
    if args.perturbation_chunk_size < 0:
        raise ValueError("--perturbation-chunk-size must be >= 0.")
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
    if args.perturbation_chunk_size > 0:
        should_chunk = True
    elif n_analysis_cells is not None and n_analysis_cells > args.max_chunk_size:
        should_chunk = True
        print(
            "[perturbo] Auto chunking enabled because the analysis dataset has "
            f"{n_analysis_cells} cells; chunk cell counts will be capped at "
            f"--max-chunk-size={args.max_chunk_size}."
        )

    if should_chunk:
        analysis_gene_names = _extract_gene_names(analysis_adata_for_workflow, args.gene_name_key)
        if args.perturbation_modality_key is not None:
            pert_adata = _resolve_perturbation_modality(data, args.perturbation_modality_key)
            pert_adata = pert_adata[analysis_adata_for_workflow.obs_names]
            pert_matrix = _get_layer_matrix(pert_adata, args.perturbation_layer)
            if args.perturbation_element_varm_key is not None:
                element_mapping, element_names = _load_perturbation_element_mapping(
                    data,
                    perturbation_modality_key=args.perturbation_modality_key,
                    perturbation_element_varm_key=args.perturbation_element_varm_key,
                    perturbation_element_names_uns_key=args.perturbation_element_names_uns_key,
                )
                pert_matrix = _group_perturbation_matrix_by_element(pert_matrix, element_mapping)
                all_perturbation_names = element_names
            else:
                all_perturbation_names = _extract_pert_names(pert_adata)
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
        else:
            pert_series = analysis_adata_for_workflow.obs[args.perturbation_key].astype(str)
            all_perturbation_names = [str(x) for x in pd.Categorical(pert_series).categories.tolist()]
            membership = _build_obs_membership(pert_series, all_perturbation_names)
            chunks = _construct_perturbation_chunks(
                all_perturbation_names,
                membership,
                max_chunk_size=args.max_chunk_size,
                max_perturbations_per_chunk=args.perturbation_chunk_size or None,
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
    schedule = resolve_training_schedule(
        shared_steps=args.num_steps,
        shared_epochs=args.num_epochs,
        control_steps=args.num_steps_control,
        beta_steps=args.num_steps_betas,
        control_epochs=args.num_epochs_control,
        beta_epochs=args.num_epochs_betas,
        default_control_steps=2500,
        default_beta_steps=2500,
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
    chunk_losses: list[jnp.ndarray] = []
    guide_efficiency_frames: list[pd.DataFrame] = []
    analysis_data: PerTurboData | None = None
    if chunks is not None:
        if all_perturbation_names is None or analysis_gene_names is None:
            raise RuntimeError("Chunk metadata was not initialized.")
        n_genes = len(analysis_gene_names)
        n_perts = len(all_perturbation_names)
        posterior_mean = np.zeros((n_perts, n_genes), dtype=np.float32)
        posterior_scale = np.zeros((n_perts, n_genes), dtype=np.float32)
        z_values = np.zeros((n_perts, n_genes), dtype=np.float32)
        last_state = None
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
            chunk_fit = fit_perturbation_effects(
                chunk_data,
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
            )
            posterior_mean[chunk_indices] = np.asarray(chunk_fit.posterior_mean)
            posterior_scale[chunk_indices] = np.asarray(chunk_fit.posterior_scale)
            z_values[chunk_indices] = np.asarray(chunk_fit.z_values)
            last_state = chunk_fit.svi_result
            if chunk_fit.losses.size > 0:
                chunk_losses.append(chunk_fit.losses)
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

        beta_fit = BetaFit(
            posterior_mean=jnp.asarray(posterior_mean),
            posterior_scale=jnp.asarray(posterior_scale),
            z_values=jnp.asarray(z_values),
            losses=jnp.concatenate(chunk_losses) if chunk_losses else jnp.array([]),
            svi_result=last_state,
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
    element_effects = build_standard_element_effects_df(
        method="perturbo",
        effect_loc=np.asarray(beta_fit.posterior_mean),
        effect_scale=np.asarray(beta_fit.posterior_scale),
        element_names=list(all_perturbation_names),
        gene_names=list(analysis_gene_names),
        null_z_values=null_z_values,
    )
    element_effects_path = out_dir / "element_effects.parquet"
    element_effects.to_parquet(element_effects_path, index=False)
    print(f"[perturbo] Wrote {element_effects_path}")

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

    if chunks is not None and chunk_losses:
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
        )
        guide_posteriors = {k: v for k, v in _guide_posterior_arrays(beta_fit).items() if v is not None}
        guide_efficacy = _guide_efficacy_for_cli(
            beta_fit,
            guide_effect_strategy=guide_effect_strategy,
            n_guides=len(guide_names),
        )
        metadata = {
            "producer": "perturbo",
            "bundle_version": 2,
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
            "fit_guide_efficacy": None,
            "library_size_center_log_mean": controls.library_size_center_log_mean,
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
            },
        }
        save_light_fit_bundle(
            out_dir,
            metadata=metadata,
            control_arrays=control_fit_arrays(control_fit),
            beta_arrays=_beta_fit_arrays(beta_fit),
            guide_posteriors=guide_posteriors or None,
            guide_efficacy=guide_efficacy,
            cell_keep_indices=np.flatnonzero(cell_keep_mask) if cell_keep_mask is not None else None,
        )
        print(f"[perturbo] Wrote simulation-ready model parameter bundle to {out_dir}")



if __name__ == "__main__":
    main()
