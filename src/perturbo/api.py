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
    pairs_to_test: str | Path | None = None,
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
    crt: bool = False,
    crt_num_resamples: int = 999,
    crt_seed: int = 0,
    crt_gene_chunk_size: int = 2000,
    crt_max_gather_gib: float = 8.0,
    crt_tail_families: list[str] | tuple[str, ...] | None = None,
    crt_mechanism: str = "permutation",
    crt_saddlepoint_only: bool = False,
    crt_screen_p_value: float = 0.05,
    crt_two_sided: str = "equal-tail",
    crt_baseline_step_tolerance: float | None = None,
    crt_allow_unconverged_baseline: bool = False,
    crt_polish_baseline: bool = False,
    crt_pool: str | None = None,
    crt_only: bool = False,
) -> PerTurboModel | None:
    """Run the file-oriented fitting workflow from Python.

    Keyword names are snake_case versions of the CLI flags. The Python defaults
    are not identical to the CLI signature defaults; in particular, this
    function declares ``crt=False``, ``step_size=0.003``, and
    ``size_factor_mode="infer"``. A false ``crt`` value leaves the CLI's
    automatic CRT selection active rather than explicitly passing ``--no-crt``.
    The implementation delegates to ``perturbo.api.main``, so Python and CLI
    fits share validation, loading, chunking, fitting, output, and light-bundle
    behavior.

    ``crt=True`` also runs the conditional randomization test against the
    stage-one baseline and adds ``crt_*`` columns to ``element_effects.parquet``.
    ``crt_tail_families=None`` keeps the CLI default tail family; pass
    ``("saddlepoint",)`` with ``crt_mechanism="propensity"`` and
    ``crt_saddlepoint_only=True`` for the exact-CGF saddlepoint without any
    resampling. An empty tuple switches the continuous tails off. ``crt_pool``
    selects the resampling pool (``'control-anchored'`` for low MOI,
    ``'all-cells'`` for high MOI; ``None`` lets the CLI choose from the design).
    ``crt_only=True`` stops after stage one and the CRT, skipping the stage-two
    fit: the path for power calculations, where the posterior is not needed.

    ``pairs_to_test`` names a two-column ``element,gene`` table. It does not
    change the run: every pair is still fitted and tested. A second effect
    table restricted to those pairs is written beside the transcriptome-wide
    one, with Benjamini-Hochberg recomputed within that smaller family, so a
    single run yields both a cis-scale comparison and the full analysis.

    Parameters
    ----------
    input_path
        Path to an AnnData ``.h5ad`` or MuData ``.h5mu`` file.
    out_dir
        Output directory. It is created if necessary; existing files with the
        standard output names may be replaced.
    pairs_to_test
        Optional CSV, TSV, or Parquet file with ``element`` and ``gene``
        columns. It requests a second restricted table without restricting the
        fit or transcriptome-wide hypothesis family.
    modality_key
        MuData modality containing expression counts. Leave unset for AnnData.
    perturbation_key
        Expression ``obs`` column with one perturbation label per cell. Required
        when ``perturbation_modality_key`` is absent.
    perturbation_modality_key
        MuData modality containing a cell-by-guide or cell-by-perturbation
        matrix.
    perturbation_layer
        Optional layer in the perturbation modality. ``None`` uses its ``X``.
    perturbation_element_varm_key
        Optional perturbation ``varm`` key containing a guide-by-element binary
        map. When present, effects are fit and reported at element level.
    perturbation_element_names_uns_key
        Perturbation ``uns`` key containing element labels when the mapping has
        no labeled columns.
    control_substring
        Regular-expression substring identifying control labels, guides, or
        mapped elements. It is required for one-label-per-cell input and for a
        control-anchored CRT.
    max_control_cells
        Maximum control cells used in stage one. Larger pools are subsampled
        without replacement with a fixed seed.
    continuous_covariates
        Expression ``obs`` columns adjusted as continuous covariates. Count-like
        values receive ``log1p`` before z-scoring; other values are z-scored.
    batch_covariate
        Optional categorical expression ``obs`` column, one-hot encoded after
        dropping the most frequent control level.
    size_factor_key
        Expression ``obs`` column containing already transformed, centered log
        offsets. Do not use this for raw library counts.
    library_size_key
        Expression ``obs`` column containing raw positive integer library sizes.
        PerTurbo applies ``log1p`` and a shared control-derived center.
    size_factor_mode
        ``"infer"`` estimates latent size factors, ``"observed"`` conditions on
        supplied or count-derived offsets, and ``"none"`` fixes offsets to zero.
        This facade defaults to ``"infer"``.
    gene_name_key
        Optional expression ``var`` column used as unique gene labels;
        otherwise ``var_names`` are used.
    clip_gene_expression_percentile
        Percentile used to derive per-gene count thresholds. ``100`` disables
        threshold-based handling. ``censored_nb`` requires a value below 100.
    gene_outlier_action
        ``"none"`` or ``"filter_cells"``. Filtering removes cells whose number
        of genes above their thresholds reaches ``outlier_cell_min_genes``.
    gene_outlier_threshold_floor
        Minimum integer threshold after per-gene percentile estimation.
    winsorize_gene_expression_outliers
        Cap remaining counts at the per-gene thresholds after optional cell
        filtering.
    outlier_cell_min_genes
        Minimum outlier-gene burden for ``gene_outlier_action="filter_cells"``.
        It must be positive when filtering and zero otherwise.
    device
        JAX device specification such as ``"cpu"``, ``"gpu"``, or ``"gpu:1"``.
        ``None`` uses JAX's default device.
    prior
        Perturbation-effect prior, ``"normal"`` or ``"cauchy"``.
    likelihood
        Observation model: ``"nb"``/``"negbin"``, ``"censored_nb"``,
        ``"lognormal_nb"``, or ``"mixture_nb"``.
    guide_effect_strategy
        ``"shared"`` gives guides targeting one element the same effect;
        ``"relative"`` learns guide-by-gene relative efficiencies and requires
        a guide-to-element map.
    guide_activity_mode
        Guide activity interpretation. ``"always_on"`` is supported;
        ``"absolute"`` is accepted by argument parsing but rejected by the
        current stage-two SVI implementation.
    guide_random_effects
        Fit hierarchical gene-wise guide variability in stage one and carry
        that calibration into stage two.
    num_steps
        Shared raw SVI update count for both stages. Step- and epoch-based
        schedules cannot be mixed. If every schedule argument is ``None``, the
        CLI uses 500 control updates and 500 effect updates.
    num_epochs
        Shared number of data passes for both stages. With full-batch SVI, one
        epoch is one update; with minibatching, steps use ceiling division by
        the effective batch size.
    num_steps_control, num_steps_betas
        Stage-specific raw update counts. Both must be provided together and
        cannot be combined with a shared or epoch-based schedule.
    num_epochs_control, num_epochs_betas
        Stage-specific data-pass counts. Both must be provided together and
        cannot be combined with a shared or step-based schedule.
    num_particles
        Monte Carlo particles in the ELBO estimate. Must be at least one.
    step_size
        Adam learning rate for both stages. This facade defaults to ``0.003``.
    num_factors
        Number of shared latent factors. Zero disables factors.
    minibatch_size
        Shared cells per SVI update when positive. Zero requests full batch
        unless a stage-specific value overrides it.
    minibatch_size_control
        Control-stage cells per update when positive; zero falls back to
        ``minibatch_size``.
    minibatch_size_betas
        Effect-stage cells per update when positive; zero falls back to
        ``minibatch_size``.
    perturbation_chunk_size
        Maximum perturbation elements per stage-two chunk. Zero chooses a size
        automatically from ``max_chunk_size``. Co-occurring predictors cannot
        be split across element chunks.
    max_chunk_size
        Maximum cells in an automatically constructed perturbation chunk.
    backed
        Open AnnData/MuData in disk-backed mode and use bounded reads where the
        execution path supports them.
    use_observed_size_factors
        Backward-compatible alias forcing ``size_factor_mode="observed"``. It
        cannot be combined with ``size_factor_mode="none"``.
    propagate_baseline_uncertainty
        Marginalize stage-two baseline intercept and dispersion over the
        stage-one variational posterior rather than conditioning on medians.
        This is unsupported for ``mixture_nb``.
    progress
        Show SVI progress bars. False forwards the CLI's no-progress flag.
    progress_chunk_size
        Positive number of full-batch SVI updates between progress refreshes.
    single_frame
        Compatibility option forwarded as ``--single-frame``. The current CLI
        does not read this parsed value; standard outputs remain Parquet tables.
    save_model_params
        Write the light, simulation-ready model bundle in ``out_dir``.
    return_model
        Load and return :class:`perturbo.PerTurboModel` after the files are
        written. This requires ``save_model_params=True``.
    crt
        If true, explicitly request the conditional randomization test and
        forward the CRT options below. If false, no ``--no-crt`` flag is sent;
        the CLI may still enable the CRT automatically when its configuration
        is supported.
    crt_num_resamples
        Resampled null assignments for empirical and moment-fitted tails. Must
        be at least one, including when the saddlepoint-only path draws none.
    crt_seed
        Random seed for CRT permutation or propensity resampling.
    crt_gene_chunk_size
        Genes per inner CRT score block. Smaller blocks reduce peak score-gather
        memory without changing the hypothesis family.
    crt_max_gather_gib
        Maximum estimated GiB for a CRT score gather before the implementation
        reduces its internal gene or resample block.
    crt_tail_families
        Continuous-null families. ``None`` keeps the CLI default
        (``"saddlepoint"``); an empty tuple disables continuous tails. Other
        supported families include moment-fitted ``"skew_normal"`` and
        ``"student_t"``.
    crt_mechanism
        ``"permutation"`` keeps the target count fixed within its pool;
        ``"propensity"`` uses Bernoulli draws from fitted cell selection
        probabilities. This facade defaults to ``"permutation"``.
    crt_saddlepoint_only
        Request no resampling pass and report the propensity-model saddlepoint
        tail alone. A false value is not forwarded explicitly, so the CLI may
        still activate its automatic saddlepoint-only default.
    crt_screen_p_value
        Evaluate the saddlepoint only for pairs whose Pearson-III screen
        p-value is at or below this threshold. Must lie in ``(0, 1]``.
    crt_two_sided
        Saddlepoint two-sided convention: ``"equal-tail"`` doubles the tail on
        the observed side; ``"symmetric"`` uses an absolute-score event.
    crt_baseline_step_tolerance
        Maximum permitted per-gene Newton step, in nats, between the supplied
        stage-one baseline and the control-cell null mode. ``None`` keeps the
        CLI default.
    crt_allow_unconverged_baseline
        Request warning instead of failure when the baseline null-mode check
        exceeds tolerance. False is not forwarded explicitly in this facade,
        so the CLI's true default remains in effect.
    crt_polish_baseline
        Request Fisher-scoring refinement of stage-one nuisance coefficients at
        fixed dispersion before the CRT. False is not forwarded explicitly, so
        the CLI's true default remains in effect.
    crt_pool
        ``"control-anchored"`` tests each low-MOI target among its cells and
        controls; ``"all-cells"`` runs the high-MOI marginal propensity
        saddlepoint; ``None`` lets the CLI choose from realized guide load.
    crt_only
        Skip stage-two effect fitting and emit CRT columns with missing effect
        estimates. This option is currently forwarded only when ``crt=True``.

    Returns
    -------
    PerTurboModel or None
        A loaded model when ``return_model=True``; otherwise ``None`` after all
        requested files are written.

    Raises
    ------
    FileNotFoundError
        If an input or requested-pairs path does not exist.
    KeyError
        If a requested modality, layer, metadata column, or mapping key is
        absent.
    ValueError
        If input counts, names, shapes, schedules, or option combinations fail
        validation, or if ``return_model`` is requested without saved params.

    Notes
    -----
    Boolean CRT options are not tri-state in this Python signature. Only
    ``crt=True`` forwards the CRT configuration block, and several false values
    omit a negative CLI flag. The descriptions above state the resulting
    behavior; set scientifically important options explicitly and inspect the
    recorded run metadata.

    This facade does not expose every current CLI switch. In particular,
    gene-axis stage-two chunking and perturbation-specific dispersion are CLI
    options in this release.
    """
    argv = ["--input", str(input_path), "--out-dir", str(out_dir)]
    _append_cli_arg(argv, "--pairs-to-test", pairs_to_test)
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
    if crt:
        argv.append("--crt")
        _append_cli_arg(argv, "--crt-num-resamples", crt_num_resamples)
        _append_cli_arg(argv, "--crt-seed", crt_seed)
        _append_cli_arg(argv, "--crt-gene-chunk-size", crt_gene_chunk_size)
        _append_cli_arg(argv, "--crt-max-gather-gib", crt_max_gather_gib)
        if crt_tail_families is not None:
            # nargs="*": the flag alone switches the continuous tails off.
            argv.append("--crt-tail-families")
            argv.extend(str(family) for family in crt_tail_families)
        _append_cli_arg(argv, "--crt-mechanism", crt_mechanism)
        if crt_saddlepoint_only:
            argv.append("--crt-saddlepoint-only")
        _append_cli_arg(argv, "--crt-screen-p-value", crt_screen_p_value)
        _append_cli_arg(argv, "--crt-two-sided", crt_two_sided)
        _append_cli_arg(argv, "--crt-baseline-step-tolerance", crt_baseline_step_tolerance)
        if crt_allow_unconverged_baseline:
            argv.append("--crt-allow-unconverged-baseline")
        if crt_polish_baseline:
            argv.append("--crt-polish-baseline")
        if crt_pool is not None:
            _append_cli_arg(argv, "--crt-pool", crt_pool)
        if crt_only:
            argv.append("--crt-only")

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
