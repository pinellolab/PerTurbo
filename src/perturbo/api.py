"""Public perturbo API facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import cli, core
from .inference import PerTurboModel, PERTURBO
from .io import MuDataSetup, get_mudata_setup, load_fit_bundle, save_fit_bundle, setup_mudata
from .preprocessing import compute_gene_clip_thresholds, count_gene_outliers_per_cell
from .results import (
    PosteriorMedians,
    PosteriorParameter,
    build_element_effects_df,
    build_guide_effects_df,
    extract_parameter_table,
)
from .simulation import save_simulated_mudata, simulate_data_from_trained_model

jax = core.jax

BaselinePosteriorSummary = core.BaselinePosteriorSummary
BetaFit = core.BetaFit
ControlFit = core.ControlFit
PerTurboData = core.PerTurboData
CovariateTransformState = core.CovariateTransformState
SVIConfig = core.SVIConfig

apply_covariate_transform = core.apply_covariate_transform
fit_control = core.fit_control
fit_covariate_transform = core.fit_covariate_transform
fit_perturbation_effects = core.fit_perturbation_effects
load_analysis_cells = core.load_analysis_cells
load_controls = core.load_controls
summarize_betas = core.summarize_betas

_construct_perturbation_chunks = core._construct_perturbation_chunks
_get_device = core._get_device
_load_from_path_with_backing = core._load_from_path_with_backing
_compute_gene_outlier_thresholds = compute_gene_clip_thresholds
_count_gene_outliers_per_cell = count_gene_outliers_per_cell
_resolve_adata = core._resolve_adata
_validate_cli_input_keys = core._validate_cli_input_keys
_save_loss_plot = core._save_loss_plot
_save_multi_loss_plot = core._save_multi_loss_plot


def main(argv: list[str] | None = None) -> None:
    patched = {
        "_construct_perturbation_chunks": _construct_perturbation_chunks,
        "_get_device": _get_device,
        "_load_from_path_with_backing": _load_from_path_with_backing,
        "_compute_gene_outlier_thresholds": _compute_gene_outlier_thresholds,
        "_count_gene_outliers_per_cell": _count_gene_outliers_per_cell,
        "_resolve_adata": _resolve_adata,
        "_validate_cli_input_keys": _validate_cli_input_keys,
        "_save_loss_plot": _save_loss_plot,
        "_save_multi_loss_plot": _save_multi_loss_plot,
        "load_controls": load_controls,
        "load_analysis_cells": load_analysis_cells,
        "fit_control": fit_control,
        "fit_perturbation_effects": fit_perturbation_effects,
        "summarize_betas": summarize_betas,
    }
    modules_to_patch = (cli, core)
    originals = {
        module: {name: getattr(module, name) for name in patched if hasattr(module, name)}
        for module in modules_to_patch
    }
    try:
        for module in modules_to_patch:
            for name, value in patched.items():
                if hasattr(module, name):
                    setattr(module, name, value)
        return cli.main(argv)
    finally:
        for module, module_originals in originals.items():
            for name, value in module_originals.items():
                setattr(module, name, value)


def _append_cli_arg(argv: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return
        argv.append(flag)
        argv.extend(str(item) for item in value)
        return
    argv.extend([flag, str(value)])


def fit_from_path(
    input_path: str | Path,
    out_dir: str | Path,
    *,
    modality_key: str | None = None,
    perturbation_key: str | None = None,
    perturbation_modality_key: str | None = None,
    perturbation_layer: str | None = None,
    perturbation_element_varm_key: str | None = None,
    perturbation_element_names_uns_key: str | None = None,
    control_substring: str | None = None,
    max_control_cells: int = 10000,
    continuous_covariates: list[str] | tuple[str, ...] | None = None,
    batch_covariate: str | None = None,
    size_factor_key: str | None = None,
    library_size_key: str | None = None,
    size_factor_mode: str = "infer",
    gene_name_key: str | None = None,
    pairs_to_test: str | Path | None = None,
    clip_gene_expression_percentile: float = 100.0,
    gene_outlier_action: str = "none",
    gene_outlier_threshold_floor: int = 2,
    winsorize_gene_expression_outliers: bool = False,
    outlier_cell_min_genes: int = 0,
    device: str | None = None,
    prior: str = "normal",
    likelihood: str = "negbin",
    guide_effect_strategy: str = "shared",
    guide_activity_mode: str = "always_on",
    guide_random_effects: bool = False,
    num_steps: int | None = None,
    num_epochs: int | None = None,
    num_steps_control: int | None = None,
    num_epochs_control: int | None = None,
    num_steps_betas: int | None = None,
    num_epochs_betas: int | None = None,
    num_particles: int = 1,
    step_size: float = 0.003,
    num_factors: int = 0,
    minibatch_size: int = 0,
    minibatch_size_control: int = 0,
    minibatch_size_betas: int = 0,
    perturbation_chunk_size: int = 0,
    max_chunk_size: int = 50000,
    backed: bool = False,
    use_observed_size_factors: bool = False,
    propagate_baseline_uncertainty: bool = False,
    progress: bool = True,
    progress_chunk_size: int = 100,
    single_frame: bool = False,
    save_model_params: bool = True,
    return_model: bool = False,
) -> PerTurboModel | None:
    """Run the CLI-equivalent end-to-end fit from Python.

    Keyword names are snake_case versions of the CLI flags. The implementation
    delegates to ``perturbo.api.main`` so Python and CLI fits share the same
    validation, loading, chunking, fitting, output, and light-bundle behavior.
    """
    argv = ["--input", str(input_path), "--out-dir", str(out_dir)]
    _append_cli_arg(argv, "--modality-key", modality_key)
    _append_cli_arg(argv, "--perturbation-key", perturbation_key)
    _append_cli_arg(argv, "--perturbation-modality-key", perturbation_modality_key)
    _append_cli_arg(argv, "--perturbation-layer", perturbation_layer)
    _append_cli_arg(argv, "--perturbation-element-varm-key", perturbation_element_varm_key)
    _append_cli_arg(argv, "--perturbation-element-names-uns-key", perturbation_element_names_uns_key)
    _append_cli_arg(argv, "--control-substring", control_substring)
    _append_cli_arg(argv, "--max-control-cells", max_control_cells)
    _append_cli_arg(argv, "--continuous-covariates", continuous_covariates)
    _append_cli_arg(argv, "--batch-covariate", batch_covariate)
    _append_cli_arg(argv, "--size-factor-key", size_factor_key)
    _append_cli_arg(argv, "--library-size-key", library_size_key)
    _append_cli_arg(argv, "--size-factor-mode", size_factor_mode)
    _append_cli_arg(argv, "--gene-name-key", gene_name_key)
    _append_cli_arg(argv, "--pairs-to-test", pairs_to_test)
    _append_cli_arg(argv, "--clip-gene-expression-percentile", clip_gene_expression_percentile)
    _append_cli_arg(argv, "--gene-outlier-action", gene_outlier_action)
    _append_cli_arg(argv, "--gene-outlier-threshold-floor", gene_outlier_threshold_floor)
    if winsorize_gene_expression_outliers:
        argv.append("--winsorize-gene-expression-outliers")
    _append_cli_arg(argv, "--outlier-cell-min-genes", outlier_cell_min_genes)
    _append_cli_arg(argv, "--device", device)
    _append_cli_arg(argv, "--prior", prior)
    _append_cli_arg(argv, "--likelihood", likelihood)
    _append_cli_arg(argv, "--guide-effect-strategy", guide_effect_strategy)
    _append_cli_arg(argv, "--guide-activity-mode", guide_activity_mode)
    if guide_random_effects:
        argv.append("--guide-random-effects")
    _append_cli_arg(argv, "--num-steps", num_steps)
    _append_cli_arg(argv, "--num-epochs", num_epochs)
    _append_cli_arg(argv, "--num-steps-control", num_steps_control)
    _append_cli_arg(argv, "--num-epochs-control", num_epochs_control)
    _append_cli_arg(argv, "--num-steps-betas", num_steps_betas)
    _append_cli_arg(argv, "--num-epochs-betas", num_epochs_betas)
    _append_cli_arg(argv, "--num-particles", num_particles)
    _append_cli_arg(argv, "--step-size", step_size)
    _append_cli_arg(argv, "--num-factors", num_factors)
    _append_cli_arg(argv, "--minibatch-size", minibatch_size)
    _append_cli_arg(argv, "--minibatch-size-control", minibatch_size_control)
    _append_cli_arg(argv, "--minibatch-size-betas", minibatch_size_betas)
    _append_cli_arg(argv, "--perturbation-chunk-size", perturbation_chunk_size)
    _append_cli_arg(argv, "--max-chunk-size", max_chunk_size)
    if backed:
        argv.append("--backed")
    if use_observed_size_factors:
        argv.append("--use-observed-size-factors")
    if propagate_baseline_uncertainty:
        argv.append("--propagate-baseline-uncertainty")
    if not progress:
        argv.append("--no-progress-bar")
    _append_cli_arg(argv, "--progress-chunk-size", progress_chunk_size)
    if single_frame:
        argv.append("--single-frame")
    if save_model_params:
        argv.append("--save-model-params")
    else:
        argv.append("--no-save-model-params")

    main(argv)
    if return_model:
        if not save_model_params:
            raise ValueError("return_model=True requires save_model_params=True.")
        return PerTurboModel.load(out_dir)
    return None

__all__ = [
    "BaselinePosteriorSummary",
    "BetaFit",
    "ControlFit",
    "PerTurboData",
    "PerTurboModel",
    "CovariateTransformState",
    "MuDataSetup",
    "PERTURBO",
    "PosteriorMedians",
    "PosteriorParameter",
    "SVIConfig",
    "jax",
    "apply_covariate_transform",
    "build_element_effects_df",
    "build_guide_effects_df",
    "extract_parameter_table",
    "fit_control",
    "fit_from_path",
    "fit_covariate_transform",
    "fit_perturbation_effects",
    "get_mudata_setup",
    "load_analysis_cells",
    "load_controls",
    "load_fit_bundle",
    "main",
    "save_fit_bundle",
    "save_simulated_mudata",
    "setup_mudata",
    "simulate_data_from_trained_model",
    "summarize_betas",
]


if __name__ == "__main__":
    main()
