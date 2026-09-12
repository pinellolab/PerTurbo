"""Core PerTurbo model fitting and data-loading routines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import partial
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from numpyro.infer.autoguide import AutoNormal
import pandas as pd
import matplotlib
import hdf5plugin  # noqa: F401 (needed for file reading, ignore unused import warning)
import anndata as ad
import mudata as md

import matplotlib.pyplot as plt
import scipy.sparse as sp
from tqdm.auto import tqdm

from perturbo.model import (
    GuideSharedCensoredNegativeBinomialModel,
    GuideSharedLogNormalNegativeBinomialModel,
    GuideSharedMixtureNegativeBinomialModel,
    GuideSharedNegativeBinomialModel,
    CensoredNegativeBinomialModel,
    LogNormalNegativeBinomialModel,
    MixtureNegativeBinomialModel,
    NegBinModel,
    create_plates,
)
from perturbo.censored_negative_binomial import (
    compute_gene_count_censoring_thresholds,
    require_count_censoring_percentile,
)
from perturbo.io import control_fit_arrays, save_array_bundle
from perturbo.results import build_guide_efficiency_df, build_standard_element_effects_df
from perturbo.preprocessing.counts import (
    BACKED_ROW_CHUNK_SIZE,
    _validate_clip_percentile,
    _validate_gene_outlier_threshold_floor,
    compute_gene_clip_thresholds,
    count_gene_outliers_per_cell,
    to_dense_array,
    winsorize_counts_to_gene_thresholds,
)
from perturbo.training_schedule import resolve_training_schedule
from perturbo.utils import compute_size_factors
from perturbo.sparse_design import (
    IndexedDesignMatrix,
    design_values_are_finite_nonnegative,
    indexed_design_from_matrix,
)

VALID_CLI_SIZE_FACTOR_MODES = ("infer", "observed", "none")


def _normalize_likelihood_name(likelihood: str) -> str:
    name = str(likelihood).lower()
    if name == "nb":
        return "negbin"
    return name


@dataclass
class SVIConfig:
    elbo: str = "meanfield"  # "trace" or "meanfield"
    num_particles: int = 1
    vectorize_particles: bool = True
    step_size: float = 0.01


@dataclass
class PerTurboData:
    counts: jnp.ndarray
    pert_id: jnp.ndarray | IndexedDesignMatrix
    pert_names: list[str]
    gene_names: list[str]
    cell_mask: jnp.ndarray | None = None
    size_factors: jnp.ndarray | None = None
    covariates: jnp.ndarray | None = None
    covariate_names: list[str] | None = None
    guide_matrix: jnp.ndarray | IndexedDesignMatrix | None = None
    guide_names: list[str] | None = None
    guide_to_element: jnp.ndarray | sp.csr_matrix | None = None
    library_size_center_log_mean: float | None = None
    categorical_batch_codes: jnp.ndarray | None = None
    categorical_batch_names: list[str] | None = None
    _analysis_design_token: object | None = None


@dataclass(frozen=True)
class _AnalysisDesignCache:
    """Gene-independent analysis inputs reused by the CLI's outer gene loop."""

    source_adata_id: int
    source_data_id: int
    source_obs_names: pd.Index
    source_var_names: pd.Index
    source_perturbation_adata_id: int | None
    source_perturbation_obs_names: pd.Index | None
    source_perturbation_var_names: pd.Index | None
    obs_indices: np.ndarray
    configuration: tuple[Any, ...]
    pert_id: jnp.ndarray | IndexedDesignMatrix
    pert_names: tuple[str, ...]
    covariates: jnp.ndarray | None
    covariate_names: tuple[str, ...] | None
    guide_matrix: jnp.ndarray | IndexedDesignMatrix | None
    guide_names: tuple[str, ...] | None
    guide_to_element: jnp.ndarray | sp.csr_matrix | None
    categorical_batch_codes: jnp.ndarray | None
    categorical_batch_names: tuple[str, ...] | None
    token: object


@dataclass
class BaselinePosteriorSummary:
    beta_0_loc: jnp.ndarray
    beta_0_scale: jnp.ndarray
    theta_log_loc: jnp.ndarray
    theta_log_scale: jnp.ndarray


@dataclass
class ControlFit:
    beta_0: jnp.ndarray
    theta: jnp.ndarray
    noise_scale: jnp.ndarray
    factor_loadings: jnp.ndarray | None
    factor_scores: jnp.ndarray | None
    factor_center: jnp.ndarray | None
    pca_loadings: jnp.ndarray | None
    size_factors: jnp.ndarray
    losses: jnp.ndarray
    svi_result: numpyro.infer.svi.SVIState
    baseline_posterior: BaselinePosteriorSummary | None = None
    pi_outlier: jnp.ndarray | None = None
    theta_outlier: jnp.ndarray | None = None
    outlier_mean_shift: jnp.ndarray | None = None
    covariate_coef: jnp.ndarray | None = None
    guide_random_effect_tau: jnp.ndarray | None = None
    guide_random_effect_log_tau_loc: jnp.ndarray | None = None
    guide_random_effect_log_tau_scale: jnp.ndarray | None = None
    count_censoring_threshold: jnp.ndarray | None = None


@dataclass
class BetaFit:
    posterior_mean: jnp.ndarray
    posterior_scale: jnp.ndarray
    z_values: jnp.ndarray
    losses: jnp.ndarray
    svi_result: numpyro.infer.svi.SVIState
    guide_effect_mean: jnp.ndarray | None = None
    guide_effect_scale: jnp.ndarray | None = None
    guide_effect_z_values: jnp.ndarray | None = None
    guide_relative_efficiency_mean: jnp.ndarray | None = None
    guide_relative_efficiency_scale: jnp.ndarray | None = None
    guide_offset_mean: jnp.ndarray | None = None
    guide_offset_scale: jnp.ndarray | None = None
    dispersion_excess_inverse: jnp.ndarray | None = None
    guide_dispersion_excess_inverse: jnp.ndarray | None = None


@dataclass
class _SVIRunResult:
    params: dict[str, Any]
    state: numpyro.infer.svi.SVIState
    losses: jnp.ndarray


@dataclass
class _ReusableSVIRunner:
    """Model/guide and compiled update loop shared by same-shaped chunk fits."""

    svi: SVI
    auto_guide: AutoNormal
    run_steps: Callable[..., tuple[numpyro.infer.svi.SVIState, jnp.ndarray]] | None = None


@dataclass
class _PerturbationChunk:
    pert_names: list[str]
    pert_indices: np.ndarray
    cell_indices: np.ndarray


@dataclass
class CovariateTransformState:
    continuous_covariates: list[str]
    batch_covariate: str | None
    continuous_medians: dict[str, float]
    continuous_transforms: dict[str, str]
    continuous_means: dict[str, float]
    continuous_stds: dict[str, float]
    batch_reference: str | None
    batch_levels: list[str]
    all_feature_names: list[str]
    feature_names: list[str]
    dropped_features: list[str]


def subset_control_fit_genes(control_fit: ControlFit, gene_indices: slice | Iterable[int]) -> ControlFit:
    """Return stage-one quantities aligned to one expression-gene chunk."""
    index = gene_indices

    def take(values, axis: int = 0):
        if values is None:
            return None
        return jnp.take(jnp.asarray(values), np.arange(values.shape[axis])[index], axis=axis)

    baseline = control_fit.baseline_posterior
    if baseline is not None:
        baseline = BaselinePosteriorSummary(
            beta_0_loc=take(baseline.beta_0_loc),
            beta_0_scale=take(baseline.beta_0_scale),
            theta_log_loc=take(baseline.theta_log_loc),
            theta_log_scale=take(baseline.theta_log_scale),
        )
    return replace(
        control_fit,
        beta_0=take(control_fit.beta_0),
        theta=take(control_fit.theta),
        noise_scale=take(control_fit.noise_scale),
        factor_loadings=take(control_fit.factor_loadings, axis=-1),
        factor_center=take(control_fit.factor_center),
        pca_loadings=take(control_fit.pca_loadings, axis=-1),
        baseline_posterior=baseline,
        pi_outlier=take(control_fit.pi_outlier),
        theta_outlier=take(control_fit.theta_outlier),
        outlier_mean_shift=take(control_fit.outlier_mean_shift),
        covariate_coef=take(control_fit.covariate_coef, axis=-1),
        guide_random_effect_tau=take(control_fit.guide_random_effect_tau),
        count_censoring_threshold=take(control_fit.count_censoring_threshold),
    )


def _resolve_cli_size_factor_mode(*, size_factor_mode: str, use_observed_size_factors: bool) -> str:
    mode = str(size_factor_mode).lower()
    if mode not in VALID_CLI_SIZE_FACTOR_MODES:
        raise ValueError(
            f"size_factor_mode must be one of {list(VALID_CLI_SIZE_FACTOR_MODES)}; got {size_factor_mode!r}."
        )
    if use_observed_size_factors:
        if mode == "none":
            raise ValueError(
                "--use-observed-size-factors cannot be combined with --size-factor-mode=none. "
                "Use only --size-factor-mode=none."
            )
        return "observed"
    return mode


def _fixed_zero_size_factors(counts: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros((int(counts.shape[0]), 1), dtype=jnp.float32)


def _build_elbo(config: SVIConfig):
    if config.elbo == "meanfield":
        return TraceMeanField_ELBO(
            num_particles=config.num_particles,
            vectorize_particles=config.vectorize_particles,
        )
    return Trace_ELBO(
        num_particles=config.num_particles,
        vectorize_particles=config.vectorize_particles,
    )


def _build_optimizer(config: SVIConfig):
    return numpyro.optim.Adam(step_size=config.step_size)


def _run_svi(
    svi: SVI,
    rng_key: jax.Array,
    num_steps: int,
    counts: jnp.ndarray | None,
    pert_id: jnp.ndarray | None,
    *,
    size_factors: jnp.ndarray | None = None,
    covariates: jnp.ndarray | None = None,
    guide_matrix: jnp.ndarray | None = None,
    cell_mask: jnp.ndarray | None = None,
    init_state: numpyro.infer.svi.SVIState | None = None,
    init_params: dict[str, Any] | None = None,
    stable_update: bool = False,
    forward_mode_differentiation: bool = False,
    progress: bool = False,
    progress_chunk_size: int = 100,
    reusable_runner: _ReusableSVIRunner | None = None,
    **static_kwargs: Any,
) -> _SVIRunResult:
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1.")

    if init_state is None:
        svi_state = svi.init(
            rng_key,
            counts,
            pert_id,
            init_params=init_params,
            size_factors=size_factors,
            covariates=covariates,
            guide_matrix=guide_matrix,
            cell_mask=cell_mask,
            skip_obs_sampling=True,
            **static_kwargs,
        )
        svi_state = _pin_svi_params_float32(svi, svi_state)
    else:
        svi_state = init_state

    run_steps = None if reusable_runner is None else reusable_runner.run_steps
    if run_steps is None:
        update_impl = svi.stable_update if stable_update else svi.update

        def run_steps(
            state: numpyro.infer.svi.SVIState,
            counts_arg: jnp.ndarray | None,
            pert_id_arg: jnp.ndarray | None,
            size_factors_arg: jnp.ndarray | None,
            covariates_arg: jnp.ndarray | None,
            guide_matrix_arg: jnp.ndarray | None,
            cell_mask_arg: jnp.ndarray | None,
            length: int,
        ):
            def body_fn(carry, _):
                return update_impl(
                    carry,
                    counts_arg,
                    pert_id_arg,
                    size_factors=size_factors_arg,
                    covariates=covariates_arg,
                    guide_matrix=guide_matrix_arg,
                    cell_mask=cell_mask_arg,
                    forward_mode_differentiation=forward_mode_differentiation,
                    **static_kwargs,
                )

            return jax.lax.scan(body_fn, state, None, length=length)

        run_steps = jax.jit(run_steps, static_argnums=(7,))
        if reusable_runner is not None:
            reusable_runner.run_steps = run_steps

    if not progress:
        svi_state, losses = run_steps(
            svi_state, counts, pert_id, size_factors, covariates, guide_matrix, cell_mask, length=num_steps
        )
        return _SVIRunResult(params=svi.get_params(svi_state), state=svi_state, losses=losses)

    if progress_chunk_size < 1:
        raise ValueError("progress_chunk_size must be >= 1.")

    losses_chunks = []
    remaining = num_steps
    with tqdm(total=num_steps, desc="[perturbo] SVI", unit="step") as pbar:
        while remaining > 0:
            step_count = min(progress_chunk_size, remaining)
            svi_state, losses = run_steps(
                svi_state,
                counts,
                pert_id,
                size_factors,
                covariates,
                guide_matrix,
                cell_mask,
                length=step_count,
            )
            losses = jax.block_until_ready(losses)
            losses_chunks.append(losses)
            last_loss = float(np.asarray(losses[-1]))
            pbar.set_postfix(loss=last_loss)
            pbar.update(step_count)
            remaining -= step_count

    all_losses = jnp.concatenate(losses_chunks) if len(losses_chunks) > 1 else losses_chunks[0]
    return _SVIRunResult(params=svi.get_params(svi_state), state=svi_state, losses=all_losses)


def _run_svi_minibatch(
    svi: SVI,
    rng_key: jax.Array,
    num_steps: int,
    counts: jnp.ndarray | None,
    pert_id: jnp.ndarray | None,
    *,
    size_factors: jnp.ndarray | None = None,
    covariates: jnp.ndarray | None = None,
    guide_matrix: jnp.ndarray | None = None,
    init_state: numpyro.infer.svi.SVIState | None = None,
    init_params: dict[str, Any] | None = None,
    stable_update: bool = False,
    forward_mode_differentiation: bool = False,
    progress: bool = False,
    progress_chunk_size: int = 100,
    batch_size: int,
    num_cells: int,
    **static_kwargs: Any,
) -> _SVIRunResult:
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if num_cells < 1:
        raise ValueError("num_cells must be >= 1.")

    if batch_size > num_cells:
        batch_size = num_cells

    seed = int(jax.random.randint(rng_key, (), minval=0, maxval=2**31 - 1))
    rng = np.random.default_rng(seed)
    def sample_indices() -> np.ndarray:
        if batch_size == num_cells:
            return np.arange(num_cells, dtype=np.int32)
        return rng.choice(num_cells, size=batch_size, replace=False).astype(np.int32)

    update_impl = svi.stable_update if stable_update else svi.update

    def update_step(
        state: numpyro.infer.svi.SVIState,
        counts_full: jnp.ndarray | None,
        pert_full: jnp.ndarray | None,
        size_full: jnp.ndarray | None,
        covariate_full: jnp.ndarray | None,
        guide_matrix_full: jnp.ndarray | None,
        idx: np.ndarray,
    ):
        return update_impl(
            state,
            counts_full,
            pert_full,
            size_factors=size_full,
            covariates=covariate_full,
            guide_matrix=guide_matrix_full,
            cell_idx=idx,
            num_cells=num_cells,
            forward_mode_differentiation=forward_mode_differentiation,
            **static_kwargs,
        )

    update_step = jax.jit(update_step)

    if init_state is None:
        init_idx = sample_indices()
        svi_state = svi.init(
            rng_key,
            counts,
            pert_id,
            init_params=init_params,
            size_factors=size_factors,
            covariates=covariates,
            guide_matrix=guide_matrix,
            cell_idx=init_idx,
            num_cells=num_cells,
            skip_obs_sampling=True,
            **static_kwargs,
        )
        svi_state = _pin_svi_params_float32(svi, svi_state)
    else:
        svi_state = init_state

    if progress_chunk_size < 1:
        raise ValueError("progress_chunk_size must be >= 1.")

    losses = []
    remaining = num_steps
    if progress:
        with tqdm(total=num_steps, desc="[perturbo] SVI", unit="step") as pbar:
            while remaining > 0:
                step_count = min(progress_chunk_size, remaining)
                for _ in range(step_count):
                    idx = sample_indices()
                    svi_state, loss = update_step(
                        svi_state,
                        counts,
                        pert_id,
                        size_factors,
                        covariates,
                        guide_matrix,
                        idx,
                    )
                    loss = jax.block_until_ready(loss)
                    losses.append(loss)
                    pbar.set_postfix(loss=float(np.asarray(loss)))
                    pbar.update(1)
                remaining -= step_count
    else:
        for _ in range(num_steps):
            idx = sample_indices()
            svi_state, loss = update_step(
                svi_state,
                counts,
                pert_id,
                size_factors,
                covariates,
                guide_matrix,
                idx,
            )
            losses.append(loss)

    losses_arr = jnp.stack(losses) if losses else jnp.array([])
    return _SVIRunResult(params=svi.get_params(svi_state), state=svi_state, losses=losses_arr)


def _resolve_model(model_name: str):
    name = model_name.lower()
    if name in ("negbin", "nb"):
        return NegBinModel
    if name in ("censored_nb", "censored_negbin"):
        return CensoredNegativeBinomialModel
    if name in ("lognormal_nb", "lnnb"):
        return LogNormalNegativeBinomialModel
    if name == "mixture_nb":
        return MixtureNegativeBinomialModel
    raise ValueError(
        "model_name must be one of: 'negbin'/'nb', 'censored_nb'/'censored_negbin', "
        "'lognormal_nb'/'lnnb', or 'mixture_nb'."
    )


def _resolve_guide_shared_model(model_name: str):
    name = model_name.lower()
    if name in ("negbin", "nb"):
        return GuideSharedNegativeBinomialModel
    if name in ("censored_nb", "censored_negbin"):
        return GuideSharedCensoredNegativeBinomialModel
    if name in ("lognormal_nb", "lnnb"):
        return GuideSharedLogNormalNegativeBinomialModel
    if name == "mixture_nb":
        return GuideSharedMixtureNegativeBinomialModel
    raise ValueError(
        "model_name must be one of: 'negbin'/'nb', 'censored_nb'/'censored_negbin', "
        "'lognormal_nb'/'lnnb', or 'mixture_nb'."
    )


def _is_censored_model_name(model_name: str) -> bool:
    return model_name.lower() in {"censored_nb", "censored_negbin"}


def _count_censoring_static_kwargs(
    counts: jnp.ndarray,
    model_name: str,
    *,
    count_censoring_percentile: float | None = None,
) -> dict[str, Any]:
    if not _is_censored_model_name(model_name):
        return {}
    percentile = require_count_censoring_percentile(count_censoring_percentile)
    return {
        "count_censoring_threshold": compute_gene_count_censoring_thresholds(counts, percentile=percentile),
    }


def _count_censoring_stage2_kwargs(
    *,
    control_fit: ControlFit,
    counts: jnp.ndarray,
    model_name: str,
    count_censoring_percentile: float | None,
) -> dict[str, Any]:
    del counts, count_censoring_percentile
    if not _is_censored_model_name(model_name):
        return {}
    if control_fit.count_censoring_threshold is None:
        raise ValueError(
            "control_fit.count_censoring_threshold is required for censored_nb stage-2 fits. "
            "Run fit_control with model_name='censored_nb' and count_censoring_percentile to precompute control thresholds."
        )
    return {
        "count_censoring_threshold": jnp.asarray(control_fit.count_censoring_threshold),
    }


def _validate_guide_strategy(guide_effect_strategy: str, guide_activity_mode: str) -> tuple[str, str]:
    strategy = str(guide_effect_strategy).lower()
    activity = str(guide_activity_mode).lower()
    if strategy not in {"shared", "relative"}:
        raise ValueError(
            "guide_effect_strategy must be one of: 'shared' or 'relative'."
        )
    if activity not in {"always_on", "absolute"}:
        raise ValueError(
            "guide_activity_mode must be one of: 'always_on' or 'absolute'."
        )
    return strategy, activity


def _validate_guide_model_request(
    data: PerTurboData,
    *,
    guide_effect_strategy: str,
    guide_activity_mode: str,
    model_name: str,
) -> tuple[str, str]:
    strategy, activity = _validate_guide_strategy(guide_effect_strategy, guide_activity_mode)
    if activity == "absolute":
        model_name_lower = model_name.lower()
        if model_name_lower not in {"negbin", "nb", "censored_nb", "censored_negbin"}:
            raise ValueError(
                "guide_activity_mode='absolute' is currently only compatible with negative-binomial likelihoods."
            )
        raise ValueError(
            "guide_activity_mode='absolute' is not implemented yet; the current stage-2 SVI path "
            "does not have a clean exact inference implementation for cell-level guide activity."
        )
    if strategy != "shared":
        if data.guide_matrix is None or data.guide_to_element is None or data.guide_names is None:
            raise ValueError(
                "guide_effect_strategy other than 'shared' requires guide-level perturbation inputs "
                "with perturbation_modality_key plus perturbation_element_varm_key."
            )
        _validate_one_parent_guide_mapping(data.guide_to_element, guide_names=data.guide_names)
    return strategy, activity


def _get_device(device: str | Any | None) -> Any | None:
    if device is None:
        return None
    if isinstance(device, jax.Device):
        return device
    if isinstance(device, str):
        spec = device.strip().lower()
        if spec == "":
            return None
        platform = spec
        device_idx = 0
        if ":" in spec:
            platform, raw_idx = spec.split(":", 1)
            if raw_idx == "":
                raise ValueError("Device spec must be '<platform>' or '<platform>:<index>' (e.g., 'gpu:1').")
            try:
                device_idx = int(raw_idx)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid device index in '{device}'. Use '<platform>:<index>' (e.g., 'gpu:1')."
                ) from exc
            if device_idx < 0:
                raise ValueError("Device index must be >= 0.")
        if platform == "cuda":
            platform = "gpu"
        matches = [d for d in jax.devices() if d.platform == platform]
        if not matches:
            raise ValueError(f"No JAX device found for platform '{platform}'.")
        if device_idx >= len(matches):
            raise ValueError(
                f"Requested device index {device_idx} for platform '{platform}', "
                f"but only {len(matches)} device(s) are available."
            )
        return matches[device_idx]
    raise TypeError("device must be None, a platform string like 'gpu'/'gpu:1', or a jax.Device.")


_to_dense = to_dense_array
_compute_gene_outlier_thresholds = compute_gene_clip_thresholds
_count_gene_outliers_per_cell = count_gene_outliers_per_cell
_winsorize_counts_to_gene_thresholds = winsorize_counts_to_gene_thresholds


def _pin_svi_params_float32(svi: SVI, svi_state):
    """Rebuild the optimizer state with every floating parameter in float32.

    ``AutoNormal`` initialises its scale parameters with ``jnp.full``, which under
    the package's float64 setting produces float64 whatever the initial locations
    are. A float64 scale promotes the reparameterised sample, and with it every
    cells-by-genes intermediate of the likelihood, to float64: twice the memory
    of the float32 fit the earlier releases ran. Pinning the parameters once,
    right after initialisation, keeps the whole fit in float32; Adam preserves
    the dtype of what it updates.
    """
    # The optimizer holds the *unconstrained* parameters. ``svi.get_params`` would
    # return them pushed through their constraints (a positive scale comes back as
    # exp of what the optimizer holds), and feeding those back through
    # ``optim.init`` would silently move every constrained parameter: an
    # AutoNormal scale initialised at 0.1 would restart at exp(0.1), ten times too
    # wide, and the fit would spend its first thousand steps shrinking it back.
    params = svi.optim.get_params(svi_state.optim_state)
    pinned = {
        name: (jnp.asarray(value, dtype=jnp.float32) if jnp.issubdtype(jnp.asarray(value).dtype, jnp.floating) else value)
        for name, value in params.items()
    }
    # numpyro's SVIState is (optim_state, mutable_state, rng_key); only the
    # optimizer state carries the parameters.
    return svi_state._replace(optim_state=svi.optim.init(pinned))


def _float32_init_values(values: dict[str, Any]) -> dict[str, Any]:
    """Pin every floating initial value to float32.

    The package enables float64 at import for the CRT's tails, and under that
    setting ``jnp.zeros`` and ``jnp.log`` produce float64. The SVI guides take their
    parameter dtypes from these initial values and the model casts its design
    matrices to the parameters' dtype, so a float64 initial value doubles every
    cells-by-genes intermediate of the likelihood. On the Replogle pipeline input
    (20,000-cell chunks, 21,629 genes) that was the difference between a 40 GB
    card fitting and running out of memory.
    """
    out: dict[str, Any] = {}
    for name, value in values.items():
        arr = jnp.asarray(value)
        out[name] = arr.astype(jnp.float32) if jnp.issubdtype(arr.dtype, jnp.floating) else value
    return out


def _to_jax(x: Any, device: Any | None, dtype: Any | None = None) -> jnp.ndarray:
    if dtype is None:
        arr = jnp.asarray(x)
    else:
        arr = jnp.asarray(x, dtype=dtype)
    if device is not None:
        return jax.device_put(arr, device=device)
    return arr


def _indexed_design_to_device(design: IndexedDesignMatrix, device: Any | None) -> IndexedDesignMatrix:
    return IndexedDesignMatrix(
        indices=_to_jax(design.indices, device, dtype=jnp.int32),
        values=_to_jax(design.values, device, dtype=jnp.float32),
        num_columns=design.num_columns,
    )


def _design_row_has_activity(design: Any) -> np.ndarray:
    if isinstance(design, IndexedDesignMatrix):
        return np.any(np.asarray(design.values) != 0, axis=1)
    if sp.issparse(design):
        return np.asarray((design != 0).sum(axis=1)).reshape(-1) > 0
    return np.asarray(design).sum(axis=1) > 0


def _take_design_rows(design: Any, rows) -> Any:
    if isinstance(design, IndexedDesignMatrix):
        return design.take_rows(rows)
    return design[rows]


def _dedupe_preserve_order(values: list[str] | None) -> list[str]:
    if values is None:
        return []
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw)
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _load_observed_size_factors(
    obs: pd.DataFrame,
    counts: Any,
    *,
    size_factor_key: str | None = None,
    library_size_key: str | None = None,
    library_size_center_log_mean: float | None = None,
    counts_library_sizes: np.ndarray | None = None,
) -> tuple[jnp.ndarray | None, float | None]:
    if size_factor_key is not None and size_factor_key in obs.columns:
        values = pd.to_numeric(obs[size_factor_key], errors="coerce").to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"size_factor_key '{size_factor_key}' contains non-finite values.")
        if _is_count_like(values):
            raise ValueError(
                f"size_factor_key '{size_factor_key}' appears to contain integer counts. "
                "Use library_size_key for count data, or provide real-valued size factors centered around zero."
            )
        if values.size > 0 and bool(np.all(values > 0)):
            raise ValueError(
                f"size_factor_key '{size_factor_key}' contains only positive values. "
                "Provide explicitly transformed and centered size factors (for example centered log factors)."
            )
        return jnp.asarray(values[:, None], dtype=jnp.float32), None
    if library_size_key is not None and library_size_key in obs.columns:
        library_size = pd.to_numeric(obs[library_size_key], errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(library_size)):
            raise ValueError(f"library_size_key '{library_size_key}' contains non-finite values.")
        if library_size.size > 0:
            if not bool(np.all(library_size > 0)):
                raise ValueError(
                    f"library_size_key '{library_size_key}' must contain only positive integers."
                )
            if not bool(np.all(np.isclose(library_size, np.round(library_size), atol=1e-6))):
                raise ValueError(
                    f"library_size_key '{library_size_key}' must contain only positive integers."
                )
        log_lib = np.log1p(library_size)
        if library_size_center_log_mean is None:
            log_mean = float(np.mean(log_lib)) if log_lib.size > 0 else 0.0
        else:
            log_mean = float(library_size_center_log_mean)
        size_factors = (log_lib - log_mean).astype(np.float32, copy=False)[:, None]
        return jnp.asarray(size_factors, dtype=jnp.float32), log_mean
    if library_size_key is None and size_factor_key is None:
        totals = counts_library_sizes
        if totals is None and counts is not None and hasattr(counts, "sum"):
            totals = np.asarray(counts.sum(axis=1), dtype=np.float64).reshape(-1)
        if totals is not None:
            totals = np.asarray(totals, dtype=np.float64).reshape(-1)
            if np.any(~np.isfinite(totals)) or np.any(totals < 0):
                raise ValueError("Counts-derived library sizes must be finite and non-negative.")
            log_lib = np.log1p(totals)
            log_mean = (
                float(np.mean(log_lib))
                if library_size_center_log_mean is None
                else float(library_size_center_log_mean)
            )
            print(
                "[perturbo] No --library-size-key given: using each cell's total count over the full "
                "analysis panel as its library size."
            )
            return jnp.asarray((log_lib - log_mean).astype(np.float32)[:, None], dtype=jnp.float32), log_mean
    return None, None


def _is_count_like(values: np.ndarray) -> bool:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    if float(np.min(finite)) < 0.0:
        return False
    return bool(np.mean(np.isclose(finite, np.round(finite), atol=1e-6)) >= 0.98)


def _zscore(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std <= 0.0 or not np.isfinite(std):
        return np.zeros_like(arr, dtype=float), mean, std
    return (arr - mean) / std, mean, std


def fit_covariate_transform(
    obs: pd.DataFrame,
    continuous_covariates: list[str] | None,
    batch_covariate: str | None,
) -> CovariateTransformState:
    continuous = _dedupe_preserve_order(continuous_covariates)
    batch_col = None if batch_covariate in (None, "", "None") else str(batch_covariate)

    for col in continuous:
        if col not in obs.columns:
            raise KeyError(f"Continuous covariate column '{col}' not found in obs.")
    if batch_col is not None and batch_col not in obs.columns:
        raise KeyError(f"Batch covariate column '{batch_col}' not found in obs.")

    continuous_medians: dict[str, float] = {}
    continuous_transforms: dict[str, str] = {}
    continuous_means: dict[str, float] = {}
    continuous_stds: dict[str, float] = {}
    all_feature_names: list[str] = []
    normalized_parts: list[np.ndarray] = []

    for col in continuous:
        raw = pd.to_numeric(obs[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(raw).any():
            raise ValueError(f"Continuous covariate column '{col}' has no numeric values.")
        median = float(np.nanmedian(raw))
        imputed = np.where(np.isfinite(raw), raw, median)
        if _is_count_like(imputed):
            transformed = np.log1p(np.clip(imputed, a_min=0.0, a_max=None))
            transform_name = "log1p+zscore"
        else:
            transformed = imputed
            transform_name = "zscore"
        normalized, mean, std = _zscore(transformed)
        continuous_medians[col] = median
        continuous_transforms[col] = transform_name
        continuous_means[col] = mean
        continuous_stds[col] = std
        all_feature_names.append(col)
        normalized_parts.append(normalized.reshape(-1, 1))

    batch_reference: str | None = None
    batch_levels: list[str] = []
    if batch_col is not None:
        batch_series = obs[batch_col].astype("string").fillna("__missing__").astype(str)
        counts = batch_series.value_counts(dropna=False)
        if counts.shape[0] > 0:
            batch_reference = str(counts.index[0])
        batch_levels = [str(level) for level in sorted(batch_series.unique().tolist()) if str(level) != batch_reference]
        batch_arr = batch_series.to_numpy()
        for level in batch_levels:
            all_feature_names.append(f"batch:{batch_col}={level}")
            normalized_parts.append((batch_arr == level).astype(float).reshape(-1, 1))

    if not normalized_parts:
        raise ValueError("No covariate features were selected. Provide continuous and/or batch covariates.")

    matrix = np.concatenate(normalized_parts, axis=1)
    variances = np.var(matrix, axis=0)
    keep_mask = np.asarray(variances > 0.0)
    feature_names = [name for name, keep in zip(all_feature_names, keep_mask, strict=False) if keep]
    dropped = [name for name, keep in zip(all_feature_names, keep_mask, strict=False) if not keep]
    if not feature_names:
        raise ValueError("All selected covariate features have zero variance after preprocessing.")

    return CovariateTransformState(
        continuous_covariates=continuous,
        batch_covariate=batch_col,
        continuous_medians=continuous_medians,
        continuous_transforms=continuous_transforms,
        continuous_means=continuous_means,
        continuous_stds=continuous_stds,
        batch_reference=batch_reference,
        batch_levels=batch_levels,
        all_feature_names=all_feature_names,
        feature_names=feature_names,
        dropped_features=dropped,
    )


def apply_covariate_transform(
    obs: pd.DataFrame,
    transform_state: CovariateTransformState,
) -> tuple[np.ndarray, list[str]]:
    if transform_state is None:
        raise ValueError("transform_state must be provided.")

    parts: list[np.ndarray] = []
    names: list[str] = []

    for col in transform_state.continuous_covariates:
        if col not in obs.columns:
            raise KeyError(f"Continuous covariate column '{col}' not found in obs.")
        raw = pd.to_numeric(obs[col], errors="coerce").to_numpy(dtype=float)
        median = float(transform_state.continuous_medians[col])
        imputed = np.where(np.isfinite(raw), raw, median)
        transform_name = transform_state.continuous_transforms[col]
        if transform_name == "log1p+zscore":
            transformed = np.log1p(np.clip(imputed, a_min=0.0, a_max=None))
        elif transform_name == "zscore":
            transformed = imputed
        else:
            raise ValueError(f"Unsupported transform '{transform_name}' for covariate '{col}'.")
        mean = float(transform_state.continuous_means[col])
        std = float(transform_state.continuous_stds[col])
        if std <= 0.0 or not np.isfinite(std):
            normalized = np.zeros_like(transformed, dtype=float)
        else:
            normalized = (transformed - mean) / std
        parts.append(normalized.reshape(-1, 1))
        names.append(col)

    batch_col = transform_state.batch_covariate
    if batch_col is not None:
        if batch_col not in obs.columns:
            raise KeyError(f"Batch covariate column '{batch_col}' not found in obs.")
        batch_series = obs[batch_col].astype("string").fillna("__missing__").astype(str)
        batch_arr = batch_series.to_numpy()
        for level in transform_state.batch_levels:
            parts.append((batch_arr == level).astype(float).reshape(-1, 1))
            names.append(f"batch:{batch_col}={level}")

    if not parts:
        return np.empty((obs.shape[0], 0), dtype=np.float32), []
    matrix = np.concatenate(parts, axis=1).astype(np.float32, copy=False)

    index_map = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in transform_state.feature_names if name not in index_map]
    if missing:
        raise RuntimeError(f"Transformed covariate features missing required columns: {missing}")
    selected_idx = [index_map[name] for name in transform_state.feature_names]
    return matrix[:, selected_idx], list(transform_state.feature_names)


def _prepare_pca_inputs(counts: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    arr = jnp.asarray(counts)
    if arr.ndim != 2:
        raise ValueError("counts must be a 2D array for PCA initialization.")
    log_counts = jnp.log1p(arr.astype(jnp.float32))
    center = jnp.mean(log_counts, axis=0)
    return log_counts - center, center


def _pca_init(
    counts: jnp.ndarray,
    num_factors: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if num_factors < 1:
        raise ValueError("num_factors must be >= 1 for PCA initialization.")
    centered, center = _prepare_pca_inputs(counts)
    n_cells, n_genes = centered.shape
    max_rank = min(int(n_cells), int(n_genes))
    k = min(num_factors, max_rank)
    if k == 0:
        raise ValueError("PCA initialization requires at least one cell and one gene.")
    if num_factors > max_rank:
        print(
            "[perturbo] num_factors exceeds PCA rank; padding remaining factors with zeros "
            f"(num_factors={num_factors}, rank={max_rank})."
        )
    u, s, vt = jnp.linalg.svd(centered, full_matrices=False)
    assert s[0] >= s[1:].max(), "SVD did not return singular values in descending order."
    components = vt[:k]
    scores = (u[:, :k] * s[:k]).T

    loadings = jnp.zeros((num_factors, n_genes), dtype=jnp.float32)
    factor_scores = jnp.zeros((num_factors, n_cells), dtype=jnp.float32)
    loadings = loadings.at[:k].set(components)
    factor_scores = factor_scores.at[:k].set(scores)

    loadings = loadings[:, jnp.newaxis, :]
    factor_scores = factor_scores[:, :, jnp.newaxis]
    return factor_scores, loadings, center


def _project_factor_scores(
    counts: jnp.ndarray,
    factor_loadings: jnp.ndarray,
    center: jnp.ndarray | None,
) -> jnp.ndarray:
    arr = jnp.asarray(counts)
    if arr.ndim != 2:
        raise ValueError("counts must be a 2D array for PCA projection.")
    log_counts = jnp.log1p(arr.astype(jnp.float32))
    if center is None:
        center = jnp.mean(log_counts, axis=0)
    centered = log_counts - jnp.asarray(center)
    loadings = jnp.asarray(factor_loadings)
    loadings = jnp.squeeze(loadings)
    if loadings.ndim == 1:
        loadings = loadings[jnp.newaxis, :]
    if loadings.ndim != 2:
        raise ValueError("factor_loadings must be 2D after squeezing singleton dimensions.")
    scores = centered @ loadings.T
    factor_scores = scores.T
    factor_scores = factor_scores[:, :, jnp.newaxis]
    return factor_scores


def _select_count_dtype(counts: np.ndarray) -> np.dtype:
    max_count = np.max(counts)
    min_count = np.min(counts)
    assert min_count >= 0 and float(max_count).is_integer(), "Counts must be non-negative integers."
    if min_count >= 0 and max_count <= np.iinfo(np.uint16).max:
        print("[perturbo] Using np.uint16 for count data.")
        return np.uint16
    if max_count <= np.iinfo(np.int16).max and min_count >= np.iinfo(np.int16).min:
        print("[perturbo] Using np.int16 for count data.")
        return np.int16
    if max_count <= np.iinfo(np.int32).max and min_count >= np.iinfo(np.int32).min:
        print("[perturbo] Using np.int32 for count data.")
        return np.int32
    if min_count >= 0 and max_count <= np.iinfo(np.uint32).max:
        print("[perturbo] Using np.uint32 for count data.")
        return np.uint32
    print("[perturbo] Using np.int64 for count data.")
    return np.int64


def _infer_num_perts(pert_id: jnp.ndarray | IndexedDesignMatrix) -> int:
    if isinstance(pert_id, IndexedDesignMatrix):
        return int(pert_id.num_columns)
    arr = jnp.asarray(pert_id)
    if arr.ndim == 1:
        return int(jnp.max(arr)) + 1
    if arr.ndim == 2:
        return int(arr.shape[1])
    raise ValueError("pert_id must be 1D (indices) or 2D (binary matrix).")


def _derive_guide_matrix_from_pert_id(pert_id: jnp.ndarray) -> jnp.ndarray | None:
    pert_arr = np.asarray(pert_id)
    if pert_arr.ndim == 2:
        guide_matrix = np.asarray(pert_arr > 0, dtype=np.float32)
    elif pert_arr.ndim == 1:
        if pert_arr.size == 0:
            return None
        n_guides = int(np.max(pert_arr)) + 1
        if n_guides <= 1:
            return None
        guide_matrix = np.zeros((pert_arr.shape[0], n_guides), dtype=np.float32)
        guide_matrix[np.arange(pert_arr.shape[0], dtype=np.int64), pert_arr.astype(np.int64)] = 1.0
    else:
        raise ValueError("pert_id must be 1D or 2D to derive guide random-effect design.")
    if guide_matrix.shape[1] <= 1:
        return None
    if not np.any(np.var(guide_matrix, axis=0) > 0.0):
        return None
    return jnp.asarray(guide_matrix, dtype=jnp.float32)


def _resolve_adata(data, modality_key: str | None):
    if hasattr(data, "mod"):
        available = list(data.mod.keys())
        if modality_key is None:
            raise ValueError(f"modality_key must be provided for MuData inputs. Available modalities: {available}")
        if modality_key not in data.mod:
            raise KeyError(f"modality_key '{modality_key}' not found in MuData.mod. Available modalities: {available}")
        return data.mod[modality_key]
    return data


def _positive_counts_per_row_bounded(
    matrix: Any,
    *,
    column_indices: np.ndarray | None = None,
    row_chunk_size: int = BACKED_ROW_CHUNK_SIZE,
) -> np.ndarray:
    """Count positive entries per row for dense, sparse, or backed matrices."""
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be positive.")
    num_rows = int(matrix.shape[0])
    counts = np.empty(num_rows, dtype=np.int32)
    columns = None if column_indices is None else np.asarray(column_indices, dtype=np.int64)
    for start in range(0, num_rows, row_chunk_size):
        stop = min(start + row_chunk_size, num_rows)
        block = matrix[start:stop]
        if columns is not None:
            block = block[:, columns]
        if sp.issparse(block) or hasattr(block, "tocsr"):
            row_counts = np.asarray((block > 0).sum(axis=1)).reshape(-1)
        else:
            row_counts = np.asarray(block) > 0
            row_counts = row_counts.sum(axis=1)
        counts[start:stop] = np.asarray(row_counts, dtype=np.int32)
    return counts


def measure_realized_moi(
    data,
    *,
    perturbation_modality_key: str | None,
    perturbation_layer: str | None,
    perturbation_key: str | None = None,
    control_substring: str | None = None,
    perturbation_element_varm_key: str | None = None,
    perturbation_element_names_uns_key: str | None = None,
    modality_key: str | None = None,
) -> dict[str, float]:
    """Guides per cell and the size of the unperturbed pool, as the data has them.

    The two CRT pools suit different designs, and the design is a property of the
    experiment rather than of how the file was written. This measures it: the median
    and mean number of perturbations a cell carries, and how many cells carry none or
    a control label.

    An AnnData input carries one perturbation label per cell, so its multiplicity is
    one by construction and only the control count has to be counted.
    """
    if perturbation_modality_key is None or not hasattr(data, "mod"):
        # One label per cell, read from the same obs frame the fit itself uses: the
        # analysed modality's when the input is a MuData, the object's own otherwise.
        adata = _resolve_adata(data, modality_key) if hasattr(data, "mod") else data
        n_cells = int(adata.n_obs)
        controls = 0
        if perturbation_key is not None and perturbation_key in adata.obs:
            labels = adata.obs[perturbation_key].astype(str).to_numpy()
            if control_substring is not None:
                controls = int(np.sum(np.char.find(labels.astype(str), str(control_substring)) >= 0))
        return {
            "median_guides_per_cell": 1.0,
            "mean_guides_per_cell": 1.0,
            "n_cells": float(n_cells),
            "n_control_cells": float(controls),
            "source": "anndata: one label per cell",
        }

    pert_adata = _resolve_perturbation_modality(data, perturbation_modality_key)
    matrix = _get_layer_matrix(pert_adata, perturbation_layer)
    per_cell = _positive_counts_per_row_bounded(matrix)
    per_cell = per_cell.astype(np.float64)
    n_control = int(np.sum(per_cell == 0))
    if control_substring is not None:
        # A guide is a control when its own name carries the substring, or when the
        # element it maps to does: the CLI matches the substring against the per-cell
        # label, which is the element name whenever an element map is in use.
        names = np.asarray(_extract_pert_names(pert_adata), dtype=str)
        is_control_guide = np.char.find(names, str(control_substring)) >= 0
        if perturbation_element_varm_key is not None and perturbation_element_varm_key in pert_adata.varm:
            mapping = pert_adata.varm[perturbation_element_varm_key]
            if hasattr(mapping, "to_numpy"):
                mapping = mapping.to_numpy()
            elif not sp.issparse(mapping):
                mapping = np.asarray(mapping)
            element_names = None
            if perturbation_element_names_uns_key is not None and perturbation_element_names_uns_key in pert_adata.uns:
                element_names = np.asarray(pert_adata.uns[perturbation_element_names_uns_key], dtype=str)
            elif hasattr(pert_adata.varm[perturbation_element_varm_key], "columns"):
                element_names = np.asarray(pert_adata.varm[perturbation_element_varm_key].columns, dtype=str)
            if element_names is not None and element_names.size == mapping.shape[1]:
                control_elements = np.char.find(element_names, str(control_substring)) >= 0
                is_control_guide |= np.asarray((mapping[:, control_elements] > 0).sum(axis=1)).reshape(-1) > 0
        if is_control_guide.any():
            carried = _positive_counts_per_row_bounded(
                matrix,
                column_indices=np.flatnonzero(is_control_guide),
            )
            # A cell counts as a control when everything it carries is a control guide.
            n_control = int(np.sum((carried > 0) & (carried == per_cell)))
    return {
        "median_guides_per_cell": float(np.median(per_cell)),
        "mean_guides_per_cell": float(np.mean(per_cell)),
        "n_cells": float(per_cell.size),
        "n_control_cells": float(n_control),
        "source": f"mudata: guides per cell from '{perturbation_modality_key}'",
    }


def _resolve_perturbation_modality(data, perturbation_modality_key: str):
    if not hasattr(data, "mod"):
        raise ValueError("perturbation_modality_key requires a MuData input.")
    available = list(data.mod.keys())
    if perturbation_modality_key not in data.mod:
        raise KeyError(
            f"perturbation_modality_key '{perturbation_modality_key}' not found in MuData.mod. "
            f"Available modalities: {available}"
        )
    return data.mod[perturbation_modality_key]


def _get_layer_matrix(adata, layer: str | None):
    if layer is None or str(layer).lower() == "x":
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in AnnData.layers.")
    return adata.layers[layer]


def _extract_pert_names(adata) -> list[str]:
    return adata.var.index.astype(str).tolist()


def _load_perturbation_matrix(
    data,
    *,
    perturbation_modality_key: str,
    perturbation_layer: str | None,
    obs_names,
    pert_subset: list[str] | None = None,
    indexed_design: bool = False,
) -> tuple[Any, list[str]]:
    pert_adata = _resolve_perturbation_modality(data, perturbation_modality_key)
    if obs_names is not None:
        pert_adata = pert_adata[obs_names]
    matrix = _get_layer_matrix(pert_adata, perturbation_layer)
    pert_names = _extract_pert_names(pert_adata)
    if pert_subset is not None:
        name_to_idx = {name: idx for idx, name in enumerate(pert_names)}
        missing = [name for name in pert_subset if name not in name_to_idx]
        if missing:
            raise KeyError(f"Perturbations not found in perturbation modality: {missing}")
        col_idx = [name_to_idx[name] for name in pert_subset]
        _validate_exact_perturbation_subset(matrix, col_idx, label="perturbations")
        matrix = matrix[:, col_idx]
        pert_names = list(pert_subset)
    return (indexed_design_from_matrix(matrix) if indexed_design else matrix), pert_names


def _resolve_element_names(
    pert_adata,
    element_mapping_raw: Any,
    *,
    perturbation_element_varm_key: str,
    perturbation_element_names_uns_key: str | None,
    n_elements: int,
) -> list[str]:
    if hasattr(element_mapping_raw, "columns"):
        names = [str(x) for x in list(element_mapping_raw.columns)]
    else:
        if perturbation_element_names_uns_key is None:
            available_uns = sorted(pert_adata.uns)
            raise ValueError(
                f"Perturbation element map '{perturbation_element_varm_key}' does not include column labels. "
                "Provide --perturbation-element-names-uns-key with labels stored in "
                f"{available_uns}."
            )
        if perturbation_element_names_uns_key not in pert_adata.uns:
            available_uns = sorted(pert_adata.uns)
            raise KeyError(
                f"Element names key '{perturbation_element_names_uns_key}' not found in perturbation modality uns. "
                f"Available keys: {available_uns}"
            )
        names_arr = np.asarray(pert_adata.uns[perturbation_element_names_uns_key]).reshape(-1)
        names = [str(x) for x in names_arr.tolist()]

    if len(names) != n_elements:
        raise ValueError(
            f"Element names length mismatch: expected {n_elements}, got {len(names)}. "
            "Ensure element labels match the number of columns in the element mapping."
        )
    if len(set(names)) != len(names):
        raise ValueError("Element names must be unique.")
    return names


def _load_perturbation_element_mapping(
    data,
    *,
    perturbation_modality_key: str,
    perturbation_element_varm_key: str,
    perturbation_element_names_uns_key: str | None,
    preserve_sparse: bool = False,
) -> tuple[Any, list[str]]:
    pert_adata = _resolve_perturbation_modality(data, perturbation_modality_key)
    if perturbation_element_varm_key not in pert_adata.varm:
        available_varm = list(pert_adata.varm.keys())
        raise KeyError(
            f"Perturbation element map '{perturbation_element_varm_key}' not found in "
            f"mdata['{perturbation_modality_key}'].varm. Available keys: {available_varm}"
        )
    element_mapping_raw = pert_adata.varm[perturbation_element_varm_key]
    if sp.issparse(element_mapping_raw):
        element_mapping = element_mapping_raw.tocsr() if preserve_sparse else element_mapping_raw.toarray()
    elif hasattr(element_mapping_raw, "to_numpy"):
        element_mapping = element_mapping_raw.to_numpy()
    else:
        element_mapping = np.asarray(element_mapping_raw)
    if element_mapping.ndim != 2:
        raise ValueError(
            f"Perturbation element map '{perturbation_element_varm_key}' must be 2D. "
            f"Got shape={element_mapping.shape}."
        )
    if element_mapping.shape[0] != pert_adata.n_vars:
        raise ValueError(
            "Perturbation element map row count must equal number of perturbations "
            f"in modality '{perturbation_modality_key}' (expected {pert_adata.n_vars}, got {element_mapping.shape[0]})."
        )
    element_names = _resolve_element_names(
        pert_adata,
        element_mapping_raw,
        perturbation_element_varm_key=perturbation_element_varm_key,
        perturbation_element_names_uns_key=perturbation_element_names_uns_key,
        n_elements=element_mapping.shape[1],
    )
    if sp.issparse(element_mapping):
        return (element_mapping > 0).astype(np.int8).tocsr(), element_names
    return np.asarray(element_mapping > 0, dtype=np.int8), element_names


def _group_perturbation_matrix_by_element(
    perturbation_matrix: Any,
    element_mapping: np.ndarray,
    *,
    preserve_sparse: bool = False,
) -> np.ndarray | sp.csr_matrix:
    # Sparse multiplication is essential even when the input is a dense array:
    # guide assignments have few positives per cell, and dense integer matmul
    # has no BLAS kernel. Keep a compact result until callers select their rows
    # and columns; int32 accumulation also avoids duplicate-guide overflow.
    mapping = sp.csr_matrix(element_mapping > 0, dtype=np.int32)
    indicator = sp.csr_matrix(perturbation_matrix > 0, dtype=np.int32)
    grouped = (indicator @ mapping).tocsr()
    grouped.data = np.asarray(grouped.data > 0, dtype=np.int8)
    grouped.eliminate_zeros()
    return grouped if preserve_sparse else grouped.toarray()


def _validate_exact_perturbation_subset(matrix: Any, selected_columns: list[int], *, label: str) -> None:
    """Reject target-axis chunks with omitted active predictors on retained rows."""
    num_columns = int(matrix.shape[1])
    selected = np.zeros(num_columns, dtype=bool)
    selected[np.asarray(selected_columns, dtype=np.int64)] = True
    if selected.all() or not selected.any():
        return
    selected_rows = np.asarray((matrix[:, selected] != 0).sum(axis=1)).reshape(-1) > 0
    if not selected_rows.any():
        return
    excluded_on_selected_rows = np.asarray((matrix[selected_rows][:, ~selected] != 0).sum(axis=1)).reshape(-1) > 0
    if np.any(excluded_on_selected_rows):
        count = int(np.count_nonzero(excluded_on_selected_rows))
        raise ValueError(
            f"Cannot fit selected {label} exactly: {count} retained cell(s) also carry excluded effects. "
            "Use gene-axis chunking so every co-occurring predictor remains in the fitted model."
        )


def _load_grouped_perturbation_matrix(
    data,
    *,
    perturbation_modality_key: str,
    perturbation_layer: str | None,
    perturbation_element_varm_key: str,
    perturbation_element_names_uns_key: str | None,
    obs_names,
    pert_subset: list[str] | None = None,
    indexed_design: bool = False,
) -> tuple[Any, list[str]]:
    pert_adata = _resolve_perturbation_modality(data, perturbation_modality_key)
    if obs_names is not None:
        pert_adata = pert_adata[obs_names]
    matrix = _get_layer_matrix(pert_adata, perturbation_layer)
    element_mapping, element_names = _load_perturbation_element_mapping(
        data,
        perturbation_modality_key=perturbation_modality_key,
        perturbation_element_varm_key=perturbation_element_varm_key,
        perturbation_element_names_uns_key=perturbation_element_names_uns_key,
        preserve_sparse=True,
    )
    grouped = _group_perturbation_matrix_by_element(matrix, element_mapping, preserve_sparse=True)
    if pert_subset is not None:
        name_to_idx = {name: idx for idx, name in enumerate(element_names)}
        missing = [name for name in pert_subset if name not in name_to_idx]
        if missing:
            raise KeyError(f"Grouped perturbation elements not found: {missing}")
        col_idx = [name_to_idx[name] for name in pert_subset]
        _validate_exact_perturbation_subset(grouped, col_idx, label="perturbation elements")
        grouped = grouped[:, col_idx]
        element_names = list(pert_subset)
    return (indexed_design_from_matrix(grouped) if indexed_design else grouped), element_names


def _load_guide_shared_perturbation_data(
    data,
    *,
    perturbation_modality_key: str,
    perturbation_layer: str | None,
    perturbation_element_varm_key: str,
    perturbation_element_names_uns_key: str | None,
    obs_names,
    element_subset: list[str] | None = None,
    indexed_design: bool = False,
) -> tuple[Any, list[str], Any, np.ndarray | sp.csr_matrix, list[str]]:
    pert_adata = _resolve_perturbation_modality(data, perturbation_modality_key)
    if obs_names is not None:
        pert_adata = pert_adata[obs_names]
    raw_matrix = _get_layer_matrix(pert_adata, perturbation_layer)
    guide_names = _extract_pert_names(pert_adata)
    element_mapping, all_element_names = _load_perturbation_element_mapping(
        data,
        perturbation_modality_key=perturbation_modality_key,
        perturbation_element_varm_key=perturbation_element_varm_key,
        perturbation_element_names_uns_key=perturbation_element_names_uns_key,
        preserve_sparse=True,
    )
    grouped = _group_perturbation_matrix_by_element(raw_matrix, element_mapping, preserve_sparse=True)
    if element_subset is not None:
        name_to_idx = {name: idx for idx, name in enumerate(all_element_names)}
        missing = [name for name in element_subset if name not in name_to_idx]
        if missing:
            raise KeyError(f"Grouped perturbation elements not found: {missing}")
        col_idx = [name_to_idx[name] for name in element_subset]
        _validate_exact_perturbation_subset(grouped, col_idx, label="perturbation elements")
        guide_mask = np.asarray(element_mapping[:, col_idx].sum(axis=1) > 0, dtype=bool).reshape(-1)
        raw_matrix = raw_matrix[:, guide_mask]
        element_mapping = element_mapping[guide_mask][:, col_idx]
        grouped = grouped[:, col_idx]
        guide_names = [name for name, keep in zip(guide_names, guide_mask, strict=True) if keep]
        element_names = list(element_subset)
    else:
        element_names = list(all_element_names)
    binary_mapping = (element_mapping > 0).astype(np.float32)
    if sp.issparse(binary_mapping):
        binary_mapping = binary_mapping.tocsr()
    if not indexed_design and sp.issparse(binary_mapping):
        binary_mapping = binary_mapping.toarray()
    return (
        indexed_design_from_matrix(raw_matrix) if indexed_design else raw_matrix,
        guide_names,
        indexed_design_from_matrix(grouped) if indexed_design else grouped,
        np.asarray(binary_mapping, dtype=np.float32) if not sp.issparse(binary_mapping) else binary_mapping,
        element_names,
    )


_PADDING_GUIDE_PREFIX = "__padding_guide_"


def _validate_one_parent_guide_mapping(
    guide_to_element: np.ndarray | sp.spmatrix,
    *,
    guide_names: list[str] | None = None,
) -> None:
    if sp.issparse(guide_to_element):
        mapping = (guide_to_element > 0).astype(np.int32).tocsr()
        row_sums = np.asarray(mapping.sum(axis=1)).reshape(-1)
    else:
        mapping = np.asarray(guide_to_element, dtype=np.int32)
        row_sums = mapping.sum(axis=1)
    invalid = np.flatnonzero(row_sums != 1)
    if guide_names is not None and len(guide_names) == mapping.shape[0] and invalid.size:
        # Chunk padding appends all-zero guide rows to reach a shared guide
        # capacity. No cell carries them and they map to no element, so they
        # contribute nothing to any guide-sharing model; only a real guide
        # without exactly one parent is a user error.
        is_padding = np.fromiter(
            (str(guide_names[int(idx)]).startswith(_PADDING_GUIDE_PREFIX) for idx in invalid),
            dtype=bool,
            count=invalid.size,
        )
        invalid = invalid[~(is_padding & (row_sums[invalid] == 0))]
    if invalid.size == 0:
        return
    if guide_names is None or len(guide_names) == 0:
        preview = ", ".join(str(int(idx)) for idx in invalid[:5])
        raise ValueError(
            "Guide-sharing modes require each guide to map to exactly one element. "
            f"Invalid guide rows: {preview}."
        )
    preview = ", ".join(str(guide_names[int(idx)]) for idx in invalid[:5])
    raise ValueError(
        "Guide-sharing modes require each guide to map to exactly one element. "
        f"Invalid guides: {preview}."
    )


def _get_control_mask(obs: pd.DataFrame, perturbation_key: str | None, control_selector):
    if control_selector is None:
        return np.ones(len(obs), dtype=bool)
    if isinstance(control_selector, str):
        if perturbation_key is None:
            raise ValueError("perturbation_key is required when control_selector is a string.")
        return obs[perturbation_key].astype(str).str.contains(control_selector).values
    if callable(control_selector):
        mask = control_selector(obs)
        return np.asarray(mask, dtype=bool)
    mask = np.asarray(control_selector, dtype=bool)
    if mask.shape[0] != len(obs):
        raise ValueError("control_selector mask length does not match obs length.")
    return mask


CONTROL_GUIDE_NAME_PATTERN = r"random|scrambled|non[-_ ]?target(?:ing)?"


def _guides_for_matching_elements(
    guide_to_element: np.ndarray | sp.spmatrix | None,
    element_names: list[str] | None,
    pattern: str,
    *,
    case: bool,
) -> np.ndarray | None:
    if guide_to_element is None or element_names is None:
        return None
    mapping = (guide_to_element > 0).astype(bool)
    if mapping.ndim != 2 or mapping.shape[1] != len(element_names):
        raise ValueError("Guide-to-element mapping shape does not match perturbation element names.")
    element_mask = (
        pd.Series(element_names, dtype="string")
        .astype(str)
        .str.contains(pattern, case=case, regex=True, na=False)
        .to_numpy()
    )
    if not np.any(element_mask):
        return None
    if sp.issparse(mapping):
        return np.asarray(mapping[:, element_mask].getnnz(axis=1) > 0, dtype=bool)
    return np.asarray(mapping[:, element_mask].any(axis=1), dtype=bool)


def _infer_control_guide_columns(
    pert_names: list[str],
    *,
    guide_to_element: np.ndarray | sp.spmatrix | None,
    element_names: list[str] | None,
) -> np.ndarray | None:
    control_cols = _guides_for_matching_elements(
        guide_to_element,
        element_names,
        CONTROL_GUIDE_NAME_PATTERN,
        case=False,
    )
    if control_cols is not None and np.any(control_cols):
        return control_cols

    mask = (
        pd.Series(pert_names, dtype="string")
        .astype(str)
        .str.contains(CONTROL_GUIDE_NAME_PATTERN, case=False, regex=True, na=False)
        .to_numpy()
    )
    if np.any(mask):
        return np.asarray(mask, dtype=bool)
    return None


def _resolve_high_moi_control_guide_columns(
    pert_names: list[str],
    control_selector: str | Iterable[bool] | Callable[[pd.DataFrame], Iterable[bool]] | None,
    *,
    guide_var: pd.DataFrame,
    guide_to_element: np.ndarray | None,
    element_names: list[str] | None,
    infer_control_guides: bool,
    perturbation_modality_key: str,
) -> np.ndarray | None:
    if control_selector is None:
        if not infer_control_guides:
            return None
        control_cols = _infer_control_guide_columns(
            pert_names,
            guide_to_element=guide_to_element,
            element_names=element_names,
        )
        if control_cols is None or not np.any(control_cols):
            raise ValueError(
                "--guide-random-effects with high-MOI perturbation matrices requires identifiable control guides. "
                "Provide --control-substring, or provide perturbation element mapping plus element names "
                "containing random, scrambled, or non-targeting controls."
            )
        return np.asarray(control_cols, dtype=bool)

    if isinstance(control_selector, str):
        control_cols = pd.Series(pert_names, dtype="string").astype(str).str.contains(control_selector).values
        if not np.any(control_cols):
            element_control_cols = _guides_for_matching_elements(
                guide_to_element,
                element_names,
                control_selector,
                case=True,
            )
            if element_control_cols is not None:
                control_cols = element_control_cols
        if not np.any(control_cols):
            preview = ", ".join(pert_names[:10])
            raise ValueError(
                f"control_selector '{control_selector}' did not match any perturbation names "
                f"or perturbation element names in modality '{perturbation_modality_key}'. "
                f"Example names: {preview}"
            )
        return np.asarray(control_cols, dtype=bool)

    if callable(control_selector):
        control_cols = np.asarray(control_selector(guide_var), dtype=bool)
    else:
        control_cols = np.asarray(control_selector, dtype=bool)
    if control_cols.shape != (len(pert_names),):
        raise ValueError("control_selector guide mask length must match the number of perturbation columns.")
    return control_cols


def _resolve_control_element_mask(
    element_names: list[str],
    control_selector: str | Iterable[bool] | None,
    *,
    infer_control_elements: bool,
) -> np.ndarray:
    if isinstance(control_selector, str):
        return (
            pd.Series(element_names, dtype="string")
            .astype(str)
            .str.contains(control_selector, case=True, regex=True, na=False)
            .to_numpy(dtype=bool)
        )
    if control_selector is not None:
        mask = np.asarray(control_selector, dtype=bool)
        if mask.shape != (len(element_names),):
            raise ValueError("control_selector element mask length must match the number of perturbation elements.")
        return mask
    if infer_control_elements:
        return (
            pd.Series(element_names, dtype="string")
            .astype(str)
            .str.contains(CONTROL_GUIDE_NAME_PATTERN, case=False, regex=True, na=False)
            .to_numpy(dtype=bool)
        )
    return np.zeros(len(element_names), dtype=bool)


def _extract_gene_names(adata, gene_name_key: str | None) -> list[str]:
    if gene_name_key is not None and gene_name_key in adata.var.columns:
        gene_series = adata.var[gene_name_key].astype(str)
        assert gene_series.is_unique, (
            f"Gene names in var['{gene_name_key}'] must be unique."
        )
        return gene_series.tolist()
    return adata.var.index.astype(str).tolist()


def _normalize_gene_indices(selected_gene_indices, num_genes: int) -> np.ndarray | slice:
    if selected_gene_indices is None:
        return slice(None)
    if isinstance(selected_gene_indices, slice):
        start, stop, step = selected_gene_indices.indices(num_genes)
        if step != 1:
            return np.arange(start, stop, step, dtype=np.int64)
        return slice(start, stop)
    indices = np.asarray(selected_gene_indices)
    if indices.ndim != 1:
        raise ValueError("selected_gene_indices must be a one-dimensional sequence or slice.")
    if indices.dtype == bool:
        if indices.shape != (num_genes,):
            raise ValueError("A boolean selected_gene_indices mask must match the full gene count.")
        indices = np.flatnonzero(indices)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size and (np.any(indices < 0) or np.any(indices >= num_genes)):
        raise IndexError("selected_gene_indices contains an index outside the full gene panel.")
    if np.unique(indices).size != indices.size:
        raise ValueError("selected_gene_indices must not contain duplicate indices.")
    return indices


def _analysis_design_configuration(
    *,
    perturbation_key: str | None,
    modality_key: str | None,
    perturbation_modality_key: str | None,
    perturbation_layer: str | None,
    perturbation_element_varm_key: str | None,
    perturbation_element_names_uns_key: str | None,
    gene_name_key: str | None,
    device: Any | None,
    continuous_covariates: list[str] | None,
    batch_covariate: str | None,
    covariate_transform_state: CovariateTransformState | None,
    retain_guide_structure: bool,
    indexed_perturbation_design: bool,
) -> tuple[Any, ...]:
    """Options whose outputs may be reused unchanged across expression-gene blocks."""
    return (
        perturbation_key,
        modality_key,
        perturbation_modality_key,
        perturbation_layer,
        perturbation_element_varm_key,
        perturbation_element_names_uns_key,
        gene_name_key,
        str(device),
        tuple(_dedupe_preserve_order(continuous_covariates)),
        None if batch_covariate in (None, "", "None") else str(batch_covariate),
        id(covariate_transform_state) if covariate_transform_state is not None else None,
        bool(retain_guide_structure),
        bool(indexed_perturbation_design),
    )


def _sum_matrix_rows_bounded(matrix, row_indices: np.ndarray, *, row_chunk_size: int = BACKED_ROW_CHUNK_SIZE) -> np.ndarray:
    """Sum selected matrix rows without materializing their full gene panel."""
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be positive.")
    rows = np.asarray(row_indices, dtype=np.int64)
    totals = np.empty(rows.size, dtype=np.float64)
    for start in range(0, rows.size, row_chunk_size):
        stop = min(start + row_chunk_size, rows.size)
        block = matrix[rows[start:stop], :]
        totals[start:stop] = np.asarray(block.sum(axis=1), dtype=np.float64).reshape(-1)
    return totals


def _load_dense_matrix_slice_bounded(
    matrix: Any,
    row_indices: np.ndarray,
    column_indices: slice | np.ndarray,
    *,
    row_chunk_size: int = 2_048,
) -> np.ndarray:
    """Read a backed matrix slice without loading every selected row at once."""
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be positive.")
    rows = np.asarray(row_indices, dtype=np.int64)
    columns = np.arange(int(matrix.shape[1]), dtype=np.int64)[column_indices]
    result = np.empty((rows.size, columns.size), dtype=matrix.dtype)
    for start in range(0, rows.size, row_chunk_size):
        stop = min(start + row_chunk_size, rows.size)
        block_rows = rows[start:stop]
        if block_rows.size and np.array_equal(
            block_rows,
            np.arange(int(block_rows[0]), int(block_rows[-1]) + 1, dtype=np.int64),
        ):
            row_selector: slice | np.ndarray = slice(int(block_rows[0]), int(block_rows[-1]) + 1)
        else:
            row_selector = block_rows
        # Backed CSR indexing reads the selected rows' compressed vectors before
        # applying a column slice. Bound that temporary by rows, then select the
        # requested genes while the block is in memory.
        block = matrix[row_selector, :]
        block = block[:, column_indices]
        if hasattr(block, "toarray"):
            block = block.toarray()
        result[start:stop] = np.asarray(block)
    return result


def _select_library_sizes_for_rows(
    full_panel_library_sizes: np.ndarray,
    *,
    source_num_cells: int,
    selected_rows: np.ndarray,
) -> np.ndarray:
    totals = np.asarray(full_panel_library_sizes, dtype=np.float64).reshape(-1)
    if totals.shape[0] == source_num_cells:
        return totals[selected_rows]
    if totals.shape[0] == selected_rows.size:
        return totals
    raise ValueError(
        "full_panel_library_sizes must have one value per source cell or one value per selected analysis cell."
    )


def _validate_cli_input_keys(
    data,
    *,
    perturbation_key: str | None,
    modality_key: str | None,
    perturbation_modality_key: str | None,
    perturbation_layer: str | None,
    perturbation_element_varm_key: str | None,
    perturbation_element_names_uns_key: str | None,
    size_factor_key: str | None,
    library_size_key: str | None,
    gene_name_key: str | None,
    continuous_covariates: list[str] | None,
    batch_covariate: str | None,
) -> None:
    adata = _resolve_adata(data, modality_key)

    if gene_name_key is not None and gene_name_key not in adata.var.columns:
        raise KeyError(
            f"gene_name_key '{gene_name_key}' not found in adata.var. "
            f"Available columns: {sorted(adata.var.columns.astype(str).tolist())}"
        )
    if size_factor_key is not None and size_factor_key not in adata.obs.columns:
        raise KeyError(
            f"size_factor_key '{size_factor_key}' not found in adata.obs. "
            f"Available columns: {sorted(adata.obs.columns.astype(str).tolist())}"
        )
    if library_size_key is not None and library_size_key not in adata.obs.columns:
        raise KeyError(
            f"library_size_key '{library_size_key}' not found in adata.obs. "
            f"Available columns: {sorted(adata.obs.columns.astype(str).tolist())}"
        )
    if perturbation_modality_key is None:
        if perturbation_key is not None and perturbation_key not in adata.obs.columns:
            raise KeyError(
                f"perturbation_key '{perturbation_key}' not found in adata.obs. "
                f"Available columns: {sorted(adata.obs.columns.astype(str).tolist())}"
            )
    missing_continuous = [col for col in _dedupe_preserve_order(continuous_covariates) if col not in adata.obs.columns]
    if missing_continuous:
        raise KeyError(
            "continuous_covariates not found in adata.obs: "
            f"{missing_continuous}. Available columns: {sorted(adata.obs.columns.astype(str).tolist())}"
        )
    batch_col = None if batch_covariate in (None, "", "None") else str(batch_covariate)
    if batch_col is not None and batch_col not in adata.obs.columns:
        raise KeyError(
            f"batch_covariate '{batch_col}' not found in adata.obs. "
            f"Available columns: {sorted(adata.obs.columns.astype(str).tolist())}"
        )

    if perturbation_modality_key is not None:
        pert_adata = _resolve_perturbation_modality(data, perturbation_modality_key)
        _get_layer_matrix(pert_adata, perturbation_layer)
        if perturbation_element_varm_key is not None and perturbation_element_varm_key not in pert_adata.varm:
            available_varm = sorted(pert_adata.varm.keys())
            raise KeyError(
                f"perturbation_element_varm_key '{perturbation_element_varm_key}' not found in "
                f"mdata['{perturbation_modality_key}'].varm. Available keys: {available_varm}"
            )
        if perturbation_element_names_uns_key is not None and perturbation_element_names_uns_key not in pert_adata.uns:
            available_uns = sorted(pert_adata.uns.keys())
            raise KeyError(
                f"perturbation_element_names_uns_key '{perturbation_element_names_uns_key}' not found in "
                f"mdata['{perturbation_modality_key}'].uns. Available keys: {available_uns}"
            )


def _extract_names(
    adata,
    gene_name_key: str | None,
    perturbation_key: str,
    pert_names_override: list[str] | None = None,
):
    gene_names = _extract_gene_names(adata, gene_name_key)
    pert_series = adata.obs[perturbation_key].astype(str)
    if pert_names_override is not None:
        pert_cat = pd.Categorical(pert_series, categories=pert_names_override)
        pert_names = [str(x) for x in pert_names_override]
    else:
        pert_cat = pd.Categorical(pert_series)
        pert_names = [str(x) for x in pert_cat.categories.tolist()]
    # pandas narrows categorical codes to int8 below 128 categories. A chunk that holds
    # one perturbation would then carry int8 codes into stage two, where a Python
    # integer as large as the first chunk's perturbation count is combined with them
    # and JAX refuses ("Python integer 742 out of bounds for int8"). Codes are int32.
    pert_id = np.asarray(pert_cat.codes, dtype=np.int32)
    return gene_names, pert_names, pert_id


def load_controls(
    data,
    *,
    perturbation_key: str | None = None,
    control_selector: str | Iterable[bool] | Callable[[pd.DataFrame], Iterable[bool]] | None,
    modality_key: str | None = None,
    perturbation_modality_key: str | None = None,
    perturbation_layer: str | None = None,
    perturbation_element_varm_key: str | None = None,
    perturbation_element_names_uns_key: str | None = None,
    max_control_cells: int = 10000,
    size_factor_key: str | None = None,
    library_size_key: str | None = None,
    gene_name_key: str | None = None,
    device: str | Any | None = None,
    cell_keep_mask: np.ndarray | None = None,
    clip_gene_expression_percentile: float | None = None,
    winsorize_gene_expression: bool = False,
    gene_outlier_threshold_floor: int = 2,
    gene_clip_thresholds: np.ndarray | None = None,
    continuous_covariates: list[str] | None = None,
    batch_covariate: str | None = None,
    return_covariate_transform_state: bool = False,
    infer_control_guides: bool = False,
    only_control_guides: bool = False,
) -> PerTurboData | tuple[PerTurboData, CovariateTransformState | None]:
    print("[perturbo] Loading controls...")
    adata = _resolve_adata(data, modality_key)

    # Compose all obs-level filters into a single integer index array before
    # touching adata.X. Backed AnnData forbids "view of a view", so compose the
    # row selection before creating the single metadata/count view below.
    if cell_keep_mask is not None:
        keep_arr = np.asarray(cell_keep_mask, dtype=bool)
        if keep_arr.shape != (adata.n_obs,):
            raise ValueError("cell_keep_mask length must match the number of cells in the analysis modality.")
        obs_idx = np.flatnonzero(keep_arr)
    else:
        obs_idx = np.arange(adata.n_obs)

    working_obs = adata.obs.iloc[obs_idx]
    working_obs_names = adata.obs_names[obs_idx]

    use_matrix = perturbation_modality_key is not None
    if use_matrix:
        pert_adata = _resolve_perturbation_modality(data, perturbation_modality_key)
        if working_obs_names is not None:
            pert_adata = pert_adata[working_obs_names]
        # Keep a sparse perturbation matrix sparse while selecting and
        # subsampling controls. It is densified only after the control cap.
        pert_id = _get_layer_matrix(pert_adata, perturbation_layer)
        pert_names = _extract_pert_names(pert_adata)
        guide_to_element = None
        element_names = None
        if perturbation_element_varm_key is not None and (
            isinstance(control_selector, str) or infer_control_guides
        ):
            guide_to_element, element_names = _load_perturbation_element_mapping(
                data,
                perturbation_modality_key=perturbation_modality_key,
                perturbation_element_varm_key=perturbation_element_varm_key,
                perturbation_element_names_uns_key=perturbation_element_names_uns_key,
                preserve_sparse=True,
            )
        control_cols = _resolve_high_moi_control_guide_columns(
            pert_names,
            control_selector,
            guide_var=pert_adata.var,
            guide_to_element=guide_to_element,
            element_names=element_names,
            infer_control_guides=infer_control_guides,
            perturbation_modality_key=perturbation_modality_key,
        )
        if control_cols is not None:
            control_cols = np.asarray(control_cols, dtype=bool)
            has_control = np.asarray(pert_id[:, control_cols].sum(axis=1) > 0).reshape(-1)
            mask = has_control
            if only_control_guides:
                # A control pool for a one-perturbation-per-cell design is the cells
                # that carry nothing else. A cell with a control guide beside a
                # targeting one is perturbed, and belongs on the analysed side.
                carries_other = np.asarray(pert_id[:, ~control_cols].sum(axis=1) > 0).reshape(-1)
                mask = has_control & ~carries_other
                print(
                    f"[perturbo] Control pool: {int(mask.sum())} cells carry only control guides; "
                    f"{int(np.count_nonzero(has_control & carries_other))} cells carrying a control guide "
                    "beside a targeting one are treated as perturbed, not as controls."
                )
            obs_idx = obs_idx[mask]
            pert_id = pert_id[mask][:, control_cols]
            pert_names = [n for n, c in zip(pert_names, control_cols) if c]
    else:
        if perturbation_key is None:
            raise ValueError("perturbation_key must be provided for obs-based perturbations.")
        mask = _get_control_mask(working_obs, perturbation_key, control_selector)
        obs_idx = obs_idx[mask]

    if max_control_cells is not None and len(obs_idx) > max_control_cells:
        rng = np.random.default_rng(0)
        subsample = np.sort(rng.choice(len(obs_idx), size=max_control_cells, replace=False))
        obs_idx = obs_idx[subsample]
        if use_matrix:
            pert_id = pert_id[subsample]
        print(f"[perturbo] Subsampled controls to {max_control_cells} cells.")

    # Single slice: for backed data this is one targeted disk read.
    adata = adata[obs_idx]

    if not use_matrix:
        gene_names, pert_names, pert_id = _extract_names(adata, gene_name_key, perturbation_key)
    else:
        gene_names = _extract_gene_names(adata, gene_name_key)

    print(f"[perturbo] Controls loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    counts = _to_dense(adata.X)
    if winsorize_gene_expression:
        counts = _winsorize_counts_to_gene_thresholds(counts, gene_clip_thresholds)
    count_dtype = _select_count_dtype(counts)

    device_obj = _get_device(device)
    counts_jax = _to_jax(counts, device_obj, dtype=count_dtype)
    size_factors, library_size_center_log_mean = _load_observed_size_factors(
        adata.obs,
        counts_jax,
        size_factor_key=size_factor_key,
        library_size_key=library_size_key,
    )
    covariate_transform_state: CovariateTransformState | None = None
    covariates = None
    covariate_names = None
    continuous = _dedupe_preserve_order(continuous_covariates)
    batch_col = None if batch_covariate in (None, "", "None") else str(batch_covariate)
    if continuous or batch_col is not None:
        covariate_transform_state = fit_covariate_transform(
            adata.obs,
            continuous_covariates=continuous,
            batch_covariate=batch_col,
        )
        cov_matrix, covariate_names = apply_covariate_transform(adata.obs, covariate_transform_state)
        if cov_matrix.shape[0] != counts.shape[0]:
            raise RuntimeError(
                "Covariate matrix row count does not match controls. "
                f"Got {cov_matrix.shape[0]} covariate rows for {counts.shape[0]} control cells."
            )
        covariates = _to_jax(cov_matrix, device_obj, dtype=jnp.float32)
    print("[perturbo] Controls prepared for JAX.")
    pert_id_arr = np.asarray(_to_dense(pert_id))
    pert_dtype = jnp.bool_ if pert_id_arr.ndim == 2 else None
    out = PerTurboData(
        counts=counts_jax,
        pert_id=_to_jax(pert_id_arr, device_obj, dtype=pert_dtype),
        pert_names=pert_names,
        gene_names=gene_names,
        size_factors=_to_jax(size_factors, device_obj, dtype=jnp.float32) if size_factors is not None else None,
        covariates=covariates,
        covariate_names=covariate_names,
        library_size_center_log_mean=library_size_center_log_mean,
        _analysis_design_token=object(),
    )
    if return_covariate_transform_state:
        return out, covariate_transform_state
    return out


def load_analysis_cells(
    data,
    *,
    perturbation_key: str | None = None,
    modality_key: str | None = None,
    perturbation_modality_key: str | None = None,
    perturbation_layer: str | None = None,
    perturbation_element_varm_key: str | None = None,
    perturbation_element_names_uns_key: str | None = None,
    size_factor_key: str | None = None,
    library_size_key: str | None = None,
    gene_name_key: str | None = None,
    device: str | Any | None = None,
    selected_perturbations: list[str] | None = None,
    cell_keep_mask: np.ndarray | None = None,
    clip_gene_expression_percentile: float | None = None,
    winsorize_gene_expression: bool = False,
    gene_outlier_threshold_floor: int = 2,
    gene_clip_thresholds: np.ndarray | None = None,
    continuous_covariates: list[str] | None = None,
    batch_covariate: str | None = None,
    covariate_transform_state: CovariateTransformState | None = None,
    retain_guide_structure: bool = False,
    library_size_center_log_mean: float | None = None,
    selected_gene_indices: slice | Iterable[int] | np.ndarray | None = None,
    full_panel_library_sizes: np.ndarray | None = None,
    indexed_perturbation_design: bool = False,
    _design_cache: _AnalysisDesignCache | None = None,
    _return_design_cache: bool = False,
) -> PerTurboData | tuple[PerTurboData, _AnalysisDesignCache]:
    subset_suffix = " for selected perturbations" if selected_perturbations is not None else ""
    print(f"[perturbo] Loading analysis cells{subset_suffix}...")
    adata = _resolve_adata(data, modality_key)
    source_adata = adata
    is_backed = bool(getattr(adata, "isbacked", False))

    configuration = _analysis_design_configuration(
        perturbation_key=perturbation_key,
        modality_key=modality_key,
        perturbation_modality_key=perturbation_modality_key,
        perturbation_layer=perturbation_layer,
        perturbation_element_varm_key=perturbation_element_varm_key,
        perturbation_element_names_uns_key=perturbation_element_names_uns_key,
        gene_name_key=gene_name_key,
        device=device,
        continuous_covariates=continuous_covariates,
        batch_covariate=batch_covariate,
        covariate_transform_state=covariate_transform_state,
        retain_guide_structure=retain_guide_structure,
        indexed_perturbation_design=indexed_perturbation_design,
    )

    # Compose all obs-level filters into a single integer index array before
    # touching adata.X. Backed AnnData forbids "view of a view", so compose the
    # row selection before creating the single metadata/count view below.
    if cell_keep_mask is not None:
        keep_arr = np.asarray(cell_keep_mask, dtype=bool)
        if keep_arr.shape != (adata.n_obs,):
            raise ValueError("cell_keep_mask length must match the number of cells in the analysis modality.")
        obs_idx = np.flatnonzero(keep_arr)
    else:
        obs_idx = np.arange(adata.n_obs)

    if _design_cache is not None:
        if selected_perturbations is not None:
            raise ValueError("A gene-block design cache cannot be combined with selected perturbations.")
        if id(adata) != _design_cache.source_adata_id:
            raise ValueError("The gene-block design cache belongs to a different analysis object.")
        if id(data) != _design_cache.source_data_id:
            raise ValueError("The gene-block design cache belongs to a different input object.")
        if adata.obs_names is not _design_cache.source_obs_names or adata.var_names is not _design_cache.source_var_names:
            raise ValueError("The analysis cell or gene identity/order changed after the gene-block cache was built.")
        if configuration != _design_cache.configuration:
            raise ValueError("Gene-independent analysis loading options changed after the gene-block cache was built.")
        if not np.array_equal(obs_idx, _design_cache.obs_indices):
            raise ValueError("The analysis cell selection/order changed after the gene-block cache was built.")
        if perturbation_modality_key is not None:
            source_perturbations = _resolve_perturbation_modality(data, perturbation_modality_key)
            if (
                id(source_perturbations) != _design_cache.source_perturbation_adata_id
                or source_perturbations.obs_names is not _design_cache.source_perturbation_obs_names
                or source_perturbations.var_names is not _design_cache.source_perturbation_var_names
            ):
                raise ValueError(
                    "The perturbation cells or guide identity/order changed after the gene-block cache was built."
                )

    working_obs = adata.obs.iloc[obs_idx]
    working_obs_names = adata.obs_names[obs_idx]

    use_matrix = perturbation_modality_key is not None
    guide_matrix = None
    guide_names = None
    guide_to_element = None
    if _design_cache is not None:
        pert_id = _design_cache.pert_id
        pert_names = list(_design_cache.pert_names)
        guide_matrix = _design_cache.guide_matrix
        guide_names = None if _design_cache.guide_names is None else list(_design_cache.guide_names)
        guide_to_element = _design_cache.guide_to_element
    elif use_matrix:
        if retain_guide_structure:
            if perturbation_element_varm_key is None:
                raise ValueError("retain_guide_structure requires perturbation_element_varm_key.")
            guide_matrix, guide_names, pert_id, guide_to_element, pert_names = _load_guide_shared_perturbation_data(
                data,
                perturbation_modality_key=perturbation_modality_key,
                perturbation_layer=perturbation_layer,
                perturbation_element_varm_key=perturbation_element_varm_key,
                perturbation_element_names_uns_key=perturbation_element_names_uns_key,
                obs_names=working_obs_names,
                element_subset=selected_perturbations,
                indexed_design=indexed_perturbation_design,
            )
            if selected_perturbations is not None:
                mask = _design_row_has_activity(pert_id)
                obs_idx = obs_idx[mask]
                pert_id = _take_design_rows(pert_id, mask)
                guide_matrix = _take_design_rows(guide_matrix, mask)
        elif perturbation_element_varm_key is not None:
            pert_id, pert_names = _load_grouped_perturbation_matrix(
                data,
                perturbation_modality_key=perturbation_modality_key,
                perturbation_layer=perturbation_layer,
                perturbation_element_varm_key=perturbation_element_varm_key,
                perturbation_element_names_uns_key=perturbation_element_names_uns_key,
                obs_names=working_obs_names,
                pert_subset=selected_perturbations,
                indexed_design=indexed_perturbation_design,
            )
            if selected_perturbations is not None:
                mask = _design_row_has_activity(pert_id)
                obs_idx = obs_idx[mask]
                pert_id = _take_design_rows(pert_id, mask)
        else:
            pert_id, pert_names = _load_perturbation_matrix(
                data,
                perturbation_modality_key=perturbation_modality_key,
                perturbation_layer=perturbation_layer,
                obs_names=working_obs_names,
                pert_subset=selected_perturbations,
                indexed_design=indexed_perturbation_design,
            )
            if selected_perturbations is not None:
                mask = _design_row_has_activity(pert_id)
                obs_idx = obs_idx[mask]
                pert_id = _take_design_rows(pert_id, mask)
    else:
        if perturbation_key is None:
            raise ValueError("perturbation_key must be provided for obs-based perturbations.")
        if selected_perturbations is not None:
            mask = working_obs[perturbation_key].astype(str).isin(selected_perturbations).values
            obs_idx = obs_idx[mask]

    gene_indices = _normalize_gene_indices(selected_gene_indices, adata.n_vars)
    counts_library_sizes = None
    if size_factor_key is None and library_size_key is None:
        if full_panel_library_sizes is not None:
            counts_library_sizes = _select_library_sizes_for_rows(
                full_panel_library_sizes,
                source_num_cells=adata.n_obs,
                selected_rows=obs_idx,
            )
        elif selected_gene_indices is not None:
            counts_library_sizes = _sum_matrix_rows_bounded(_get_layer_matrix(adata, None), obs_idx)

    # One compound slice ensures backed AnnData reads only the requested rows
    # and gene chunk. Accessing ``adata.X`` below materializes that matrix slice
    # alone; converting the whole AnnData view to memory would also load its
    # unsliced ``.raw`` matrix and unrelated layers.
    counts = None
    if is_backed and selected_gene_indices is not None:
        counts = _load_dense_matrix_slice_bounded(
            _get_layer_matrix(adata, None),
            obs_idx,
            gene_indices,
        )
    adata = adata[obs_idx, gene_indices]

    if use_matrix or _design_cache is not None:
        gene_names = _extract_gene_names(adata, gene_name_key)
    else:
        gene_names, pert_names, pert_id = _extract_names(
            adata,
            gene_name_key,
            perturbation_key,
            pert_names_override=selected_perturbations,
        )
    if counts is None:
        counts = _to_dense(adata.X)
    if winsorize_gene_expression:
        selected_thresholds = gene_clip_thresholds
        if gene_clip_thresholds is not None and selected_gene_indices is not None:
            selected_thresholds = np.asarray(gene_clip_thresholds)[gene_indices]
        counts = _winsorize_counts_to_gene_thresholds(counts, selected_thresholds)
    count_dtype = _select_count_dtype(counts)

    print(f"[perturbo] Analysis cells loaded: {adata.n_obs} cells, {adata.n_vars} genes")

    device_obj = _get_device(device)
    counts_jax = _to_jax(counts, device_obj, dtype=count_dtype)
    size_factors, loaded_library_size_center_log_mean = _load_observed_size_factors(
        adata.obs,
        counts_jax,
        size_factor_key=size_factor_key,
        library_size_key=library_size_key,
        library_size_center_log_mean=library_size_center_log_mean,
        counts_library_sizes=counts_library_sizes,
    )
    covariates = None if _design_cache is None else _design_cache.covariates
    covariate_names = (
        None if _design_cache is None or _design_cache.covariate_names is None
        else list(_design_cache.covariate_names)
    )
    if _design_cache is not None:
        pass
    elif covariate_transform_state is not None:
        cov_matrix, covariate_names = apply_covariate_transform(adata.obs, covariate_transform_state)
        if cov_matrix.shape[0] != counts.shape[0]:
            raise RuntimeError(
                "Covariate matrix row count does not match analysis cells. "
                f"Got {cov_matrix.shape[0]} covariate rows for {counts.shape[0]} cells."
            )
        covariates = _to_jax(cov_matrix, device_obj, dtype=jnp.float32)
    else:
        continuous = _dedupe_preserve_order(continuous_covariates)
        batch_col = None if batch_covariate in (None, "", "None") else str(batch_covariate)
        if continuous or batch_col is not None:
            transform_state = fit_covariate_transform(
                adata.obs,
                continuous_covariates=continuous,
                batch_covariate=batch_col,
            )
            cov_matrix, covariate_names = apply_covariate_transform(adata.obs, transform_state)
            if cov_matrix.shape[0] != counts.shape[0]:
                raise RuntimeError(
                    "Covariate matrix row count does not match analysis cells. "
                    f"Got {cov_matrix.shape[0]} covariate rows for {counts.shape[0]} cells."
                )
            covariates = _to_jax(cov_matrix, device_obj, dtype=jnp.float32)
    print("[perturbo] Analysis cells prepared for JAX.")
    if _design_cache is not None:
        pert_id_jax = pert_id
    elif isinstance(pert_id, IndexedDesignMatrix):
        pert_id_jax = _indexed_design_to_device(pert_id, device_obj)
    else:
        pert_id_arr = np.asarray(_to_dense(pert_id))
        pert_dtype = jnp.bool_ if pert_id_arr.ndim == 2 else None
        pert_id_jax = _to_jax(pert_id_arr, device_obj, dtype=pert_dtype)
    if _design_cache is not None:
        guide_matrix_jax = guide_matrix
    elif isinstance(guide_matrix, IndexedDesignMatrix):
        guide_matrix_jax = _indexed_design_to_device(guide_matrix, device_obj)
    else:
        guide_matrix_jax = (
            _to_jax(_to_dense(guide_matrix), device_obj, dtype=jnp.float32)
            if guide_matrix is not None
            else None
        )
    if sp.issparse(guide_to_element):
        guide_to_element_out = guide_to_element
    else:
        guide_to_element_out = (
            _to_jax(np.asarray(guide_to_element), device_obj, dtype=jnp.float32)
            if guide_to_element is not None
            else None
        )
    token = _design_cache.token if _design_cache is not None else object()
    out = PerTurboData(
        counts=counts_jax,
        pert_id=pert_id_jax,
        pert_names=pert_names,
        gene_names=gene_names,
        size_factors=_to_jax(size_factors, device_obj, dtype=jnp.float32) if size_factors is not None else None,
        covariates=covariates,
        covariate_names=covariate_names,
        guide_matrix=guide_matrix_jax,
        guide_names=guide_names,
        guide_to_element=guide_to_element_out,
        library_size_center_log_mean=loaded_library_size_center_log_mean,
        _analysis_design_token=token,
    )
    if not _return_design_cache:
        return out
    if selected_perturbations is not None:
        raise ValueError("A gene-block design cache requires every perturbation predictor.")
    cached_rows = np.asarray(obs_idx, dtype=np.int64).copy()
    cached_rows.flags.writeable = False
    source_perturbations = (
        None
        if perturbation_modality_key is None
        else _resolve_perturbation_modality(data, perturbation_modality_key)
    )
    cache = _AnalysisDesignCache(
        source_adata_id=id(source_adata),
        source_data_id=id(data),
        source_obs_names=source_adata.obs_names,
        source_var_names=source_adata.var_names,
        source_perturbation_adata_id=None if source_perturbations is None else id(source_perturbations),
        source_perturbation_obs_names=(
            None if source_perturbations is None else source_perturbations.obs_names
        ),
        source_perturbation_var_names=(
            None if source_perturbations is None else source_perturbations.var_names
        ),
        obs_indices=cached_rows,
        configuration=configuration,
        pert_id=pert_id_jax,
        pert_names=tuple(str(name) for name in pert_names),
        covariates=covariates,
        covariate_names=None if covariate_names is None else tuple(covariate_names),
        guide_matrix=guide_matrix_jax,
        guide_names=None if guide_names is None else tuple(guide_names),
        guide_to_element=guide_to_element_out,
        categorical_batch_codes=out.categorical_batch_codes,
        categorical_batch_names=(
            None if out.categorical_batch_names is None else tuple(out.categorical_batch_names)
        ),
        token=token,
    )
    return out, cache


def fit_control(
    data: PerTurboData,
    *,
    num_steps: int = 1000,
    prior: str = "normal",
    svi_config: SVIConfig | None = None,
    model_name: str = "negbin",
    num_factors: int | None = None,
    use_observed_size_factors: bool = False,
    count_censoring_percentile: float | None = None,
    minibatch_size: int | None = None,
    progress: bool = False,
    progress_chunk_size: int = 100,
    guide_random_effects: bool = False,
) -> ControlFit:
    print("[perturbo] Fitting control model...")
    counts = data.counts
    pert_id = data.pert_id
    num_perts = _infer_num_perts(pert_id)
    num_genes = counts.shape[1]
    if minibatch_size is not None and minibatch_size > counts.shape[0]:
        print(
            "[perturbo] minibatch_size exceeds dataset size; using full batch "
            f"(minibatch_size={minibatch_size}, n_cells={counts.shape[0]})."
        )
        minibatch_size = None

    model_name_lower = model_name.lower()
    is_lognormal_model = model_name_lower in {"lognormal_nb", "lnnb"}
    is_mixture_model = model_name_lower == "mixture_nb"
    count_censoring_kwargs = _count_censoring_static_kwargs(
        counts,
        model_name,
        count_censoring_percentile=count_censoring_percentile,
    )

    mean_counts = jnp.mean(counts, axis=0)
    beta_0_init = jnp.log(mean_counts + 1e-3)
    var_counts = jnp.var(counts, axis=0)
    overdispersion = jnp.maximum((var_counts - mean_counts) / (mean_counts**2 + 1e-3), 1e-3)
    theta_init = 1.0 / overdispersion
    noise_scale_init = jnp.full((num_genes,), 0.01)
    pi_outlier_init = jnp.full((num_genes,), 0.01, dtype=jnp.float32)
    outlier_mean_shift_init = jnp.full((num_genes,), 0.25, dtype=jnp.float32)

    size_factors = data.size_factors
    covariates = data.covariates
    if size_factors is None:
        size_factors = compute_size_factors(counts)
    guide_matrix_for_random_effects = _derive_guide_matrix_from_pert_id(pert_id) if guide_random_effects else None
    random_effects_active = bool(guide_random_effects and guide_matrix_for_random_effects is not None)
    if guide_random_effects and not random_effects_active:
        raise ValueError(
            "guide_random_effects=True requires at least two guide/control-guide labels with variation "
            "across stage-1 control cells. For high-MOI inputs, provide --control-substring that matches "
            "multiple control guide columns."
        )
    num_guides = (
        int(guide_matrix_for_random_effects.shape[1])
        if guide_matrix_for_random_effects is not None
        else None
    )

    init_values = {
        "beta_0": beta_0_init,
        "beta": jnp.zeros((num_perts, num_genes)),
        "theta": theta_init,
    }
    if is_lognormal_model:
        init_values["noise_scale"] = noise_scale_init
    if is_mixture_model:
        init_values["pi_outlier"] = pi_outlier_init
        init_values["theta_outlier"] = theta_init
        init_values["outlier_mean_shift"] = outlier_mean_shift_init
    if covariates is not None:
        init_values["covariate_coef"] = jnp.zeros((covariates.shape[1], num_genes))
    if random_effects_active and num_guides is not None:
        init_values["guide_random_effect_log_tau_loc"] = jnp.asarray(-2.0, dtype=jnp.float32)
        init_values["guide_random_effect_log_tau_scale"] = jnp.asarray(0.5, dtype=jnp.float32)
        init_values["guide_random_effect_tau"] = jnp.full((num_genes,), 0.1, dtype=jnp.float32)
        init_values["guide_random_effect"] = jnp.zeros((num_guides, num_genes), dtype=jnp.float32)
    factor_scores_init = None
    factor_loadings_init = None
    factor_center = None
    if num_factors is not None:
        factor_scores_init, factor_loadings_init, factor_center = _pca_init(counts, num_factors)

    if minibatch_size is None:
        init_values["size_factor"] = size_factors
    if factor_scores_init is not None and minibatch_size is None:
        init_values["factor_scores"] = factor_scores_init
    if factor_loadings_init is not None:
        init_values["factor_loadings"] = factor_loadings_init
    init_value_fn = numpyro.infer.init_to_value(values=_float32_init_values(init_values))
    model_cls = _resolve_model(model_name)
    model = numpyro.handlers.condition(
        model_cls,
        data={"beta": jnp.zeros((num_perts, num_genes), dtype=jnp.float32)},
    )
    guide = AutoNormal(model, init_loc_fn=init_value_fn, create_plates=create_plates)

    if svi_config is None:
        svi_config = SVIConfig()
    optimizer = _build_optimizer(svi_config)
    svi = SVI(model, guide, optimizer, loss=_build_elbo(svi_config))

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    if minibatch_size is None:
        result = _run_svi(
            svi,
            key,
            num_steps,
            counts,
            pert_id,
            size_factors=size_factors if use_observed_size_factors else None,
            covariates=covariates,
            guide_matrix=guide_matrix_for_random_effects if random_effects_active else None,
            progress=progress,
            progress_chunk_size=progress_chunk_size,
            num_cells=counts.shape[0],
            num_perts=num_perts,
            num_guides=num_guides,
            num_genes=num_genes,
            num_factors=num_factors,
            subsample_size=None,
            prior=prior,
            guide_random_effects=random_effects_active,
            **count_censoring_kwargs,
        )
    else:
        result = _run_svi_minibatch(
            svi,
            key,
            num_steps,
            counts,
            pert_id,
            size_factors=size_factors if use_observed_size_factors else None,
            covariates=covariates,
            guide_matrix=guide_matrix_for_random_effects if random_effects_active else None,
            progress=progress,
            progress_chunk_size=progress_chunk_size,
            batch_size=minibatch_size,
            num_cells=counts.shape[0],
            num_perts=num_perts,
            num_guides=num_guides,
            num_genes=num_genes,
            num_factors=num_factors,
            subsample_size=None,
            prior=prior,
            guide_random_effects=random_effects_active,
            **count_censoring_kwargs,
        )
    baseline_posterior = BaselinePosteriorSummary(
        beta_0_loc=result.params["beta_0_auto_loc"],
        beta_0_scale=jnp.clip(result.params["beta_0_auto_scale"], 1e-6, None),
        theta_log_loc=result.params["theta_auto_loc"],
        theta_log_scale=jnp.clip(result.params["theta_auto_scale"], 1e-6, None),
    )
    med = guide.median(result.params)
    noise_scale = med.get("noise_scale", noise_scale_init)
    if noise_scale is not None:
        # Keep the saved fit in the dtype the likelihood runs in.
        noise_scale = jnp.asarray(noise_scale, dtype=jnp.float32)
    pi_outlier = med.get("pi_outlier")
    theta_outlier = med.get("theta_outlier")
    outlier_mean_shift = med.get("outlier_mean_shift")
    covariate_coef = med.get("covariate_coef")
    guide_random_effect_tau = med.get("guide_random_effect_tau")
    guide_random_effect_log_tau_loc = med.get("guide_random_effect_log_tau_loc")
    guide_random_effect_log_tau_scale = med.get("guide_random_effect_log_tau_scale")
    factor_loadings = med.get("factor_loadings")
    factor_scores = med.get("factor_scores")
    print("[perturbo] Control fit complete.")
    return ControlFit(
        beta_0=med["beta_0"],
        theta=med["theta"],
        noise_scale=noise_scale,
        covariate_coef=covariate_coef,
        factor_loadings=factor_loadings,
        factor_scores=factor_scores,
        factor_center=factor_center,
        pca_loadings=factor_loadings_init,
        size_factors=med["size_factor"] if "size_factor" in med else size_factors,
        baseline_posterior=baseline_posterior,
        losses=result.losses,
        svi_result=result.state,
        pi_outlier=pi_outlier,
        theta_outlier=theta_outlier,
        outlier_mean_shift=outlier_mean_shift,
        guide_random_effect_tau=guide_random_effect_tau,
        guide_random_effect_log_tau_loc=guide_random_effect_log_tau_loc,
        guide_random_effect_log_tau_scale=guide_random_effect_log_tau_scale,
        count_censoring_threshold=count_censoring_kwargs.get("count_censoring_threshold"),
    )


def _posterior_summary_from_draws(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(draws, dtype=np.float32)
    mean = np.asarray(arr.mean(axis=0), dtype=np.float32)
    scale = np.clip(np.asarray(arr.std(axis=0), dtype=np.float32), 1e-6, None)
    z_values = mean / scale
    return mean, scale, z_values


@partial(jax.jit, static_argnames=("num_samples",))
def _summarize_relative_guide_block(
    beta_loc: jnp.ndarray,
    beta_scale: jnp.ndarray,
    relative_loc: jnp.ndarray,
    relative_scale: jnp.ndarray,
    guide_to_element: jnp.ndarray,
    beta_key: jax.Array,
    relative_key: jax.Array,
    *,
    num_samples: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Draw and summarize one bounded element-by-guide block of the posterior."""
    num_elements, num_genes = beta_loc.shape
    num_guides = relative_loc.shape[0]
    beta_draws = beta_loc + beta_scale * jax.random.normal(
        beta_key, shape=(num_samples, num_elements, num_genes), dtype=beta_loc.dtype
    )
    relative_draws = jax.nn.sigmoid(
        relative_loc
        + relative_scale
        * jax.random.normal(relative_key, shape=(num_samples, num_guides, num_genes), dtype=relative_loc.dtype)
    )
    guide_effect_draws = jnp.einsum("qe,seg->sqg", guide_to_element, beta_draws) * relative_draws
    return (
        jnp.mean(guide_effect_draws, axis=0),
        jnp.clip(jnp.std(guide_effect_draws, axis=0), 1e-6),
        jnp.mean(relative_draws, axis=0),
        jnp.clip(jnp.std(relative_draws, axis=0), 1e-6),
    )


def _summarize_stage2_guide_posteriors(
    params: dict[str, Any],
    *,
    data: PerTurboData,
    guide_effect_strategy: str,
    num_samples: int = 64,
    guide_block_size: int = 16,
    element_block_size: int = 100,
) -> dict[str, np.ndarray]:
    """Guide-level effect summaries without materializing every posterior draw.

    A summary is returned exactly when a guide's effect differs from its element's.

    Under ``shared`` it does not: the model sets ``guide_effect = guide_to_element
    @ beta``, so the guide posterior is the element posterior copied, and nothing
    is returned. Under ``offset`` the guide adds an independent Normal, so the
    moments are closed-form. Only ``relative`` is nonlinear, the efficiency
    passing through a sigmoid, and it is sampled in bounded element and guide
    blocks.

    The previous implementation drew ``num_samples`` samples of the whole
    ``(elements, genes)`` effect matrix and contracted them against the guide map
    whatever the strategy. On a 13,000-guide by 13,000-gene screen that is 44 GB
    of intermediates on an element fit and about 89 GB on a guide-level one,
    which put guide output out of reach at the scale that asks for it - and under
    ``shared`` it spent all of that to recompute a copy.
    """
    if data.guide_to_element is None:
        return {}
    guide_to_element = np.asarray(data.guide_to_element, dtype=np.float32)
    beta_loc = np.asarray(params["beta_auto_loc"], dtype=np.float32)
    beta_scale = np.clip(np.asarray(params["beta_auto_scale"], dtype=np.float32), 1e-6, None)

    def _closed_form(extra_mean: np.ndarray | None, extra_variance: np.ndarray | None) -> dict[str, np.ndarray]:
        effect_mean = guide_to_element @ beta_loc
        effect_variance = np.square(guide_to_element) @ np.square(beta_scale)
        if extra_mean is not None:
            effect_mean = effect_mean + extra_mean
        if extra_variance is not None:
            effect_variance = effect_variance + extra_variance
        effect_scale = np.clip(np.sqrt(effect_variance), 1e-6, None)
        return {
            "guide_effect_mean": effect_mean.astype(np.float32, copy=False),
            "guide_effect_scale": effect_scale.astype(np.float32, copy=False),
            "guide_effect_z_values": (effect_mean / effect_scale).astype(np.float32, copy=False),
        }

    if guide_effect_strategy == "shared":
        # Under this strategy the model sets guide_effect = guide_to_element @ beta,
        # so a guide's posterior *is* its element's, copied. Returning it here would
        # duplicate the element table as a (guides, genes) array for no information;
        # consumers that want guide rows join the element table through the map, and
        # :meth:`PerTurboModel._guide_effect_payload` does exactly that.
        return {}

    if guide_effect_strategy == "offset":
        offset_loc = np.asarray(params["guide_offset_auto_loc"], dtype=np.float32)
        offset_scale = np.clip(np.asarray(params["guide_offset_auto_scale"], dtype=np.float32), 1e-6, None)
        summary = _closed_form(offset_loc, np.square(offset_scale))
        summary["guide_offset_mean"] = offset_loc
        summary["guide_offset_scale"] = offset_scale
        return summary

    if guide_effect_strategy != "relative":
        raise ValueError(f"Unknown guide_effect_strategy: {guide_effect_strategy}")
    if guide_block_size < 1 or element_block_size < 1:
        raise ValueError("guide_block_size and element_block_size must be >= 1.")

    beta_loc_jax = jnp.asarray(params["beta_auto_loc"])
    beta_scale_jax = jnp.asarray(params["beta_auto_scale"])
    relative_loc = jnp.asarray(params["guide_relative_efficiency_auto_loc"])
    relative_scale = jnp.asarray(params["guide_relative_efficiency_auto_scale"])
    num_guides, num_genes = guide_to_element.shape[0], beta_loc.shape[1]
    effect_mean = np.zeros((num_guides, num_genes), dtype=np.float32)
    effect_scale = np.zeros((num_guides, num_genes), dtype=np.float32)
    relative_mean = np.zeros((num_guides, num_genes), dtype=np.float32)
    relative_scale_summary = np.zeros((num_guides, num_genes), dtype=np.float32)
    rng_key = jax.random.PRNGKey(2024)

    for element_start in range(0, beta_loc.shape[0], element_block_size):
        element_stop = min(element_start + element_block_size, beta_loc.shape[0])
        guide_indices = np.flatnonzero(np.any(guide_to_element[:, element_start:element_stop] != 0, axis=1))
        if guide_indices.size == 0:
            continue
        for start in range(0, guide_indices.size, guide_block_size):
            indices = guide_indices[start : start + guide_block_size]
            block = _summarize_relative_guide_block(
                beta_loc_jax[element_start:element_stop],
                beta_scale_jax[element_start:element_stop],
                relative_loc[indices],
                relative_scale[indices],
                jnp.asarray(guide_to_element[indices, element_start:element_stop]),
                jax.random.fold_in(rng_key, element_start),
                jax.random.fold_in(rng_key, int(indices[0]) + 1),
                num_samples=num_samples,
            )
            (
                effect_mean[indices],
                effect_scale[indices],
                relative_mean[indices],
                relative_scale_summary[indices],
            ) = (np.asarray(value, dtype=np.float32) for value in block)

    return {
        "guide_effect_mean": effect_mean,
        "guide_effect_scale": effect_scale,
        "guide_effect_z_values": effect_mean / np.clip(effect_scale, 1e-6, None),
        "guide_relative_efficiency_mean": relative_mean,
        "guide_relative_efficiency_scale": relative_scale_summary,
    }


def fit_perturbation_effects(
    data: PerTurboData,
    control_fit: ControlFit,
    *,
    num_steps: int = 1000,
    prior: str = "normal",
    svi_config: SVIConfig | None = None,
    model_name: str = "negbin",
    num_factors: int | None = None,
    propagate_baseline_uncertainty: bool = False,
    use_observed_size_factors: bool = False,
    count_censoring_percentile: float | None = None,
    minibatch_size: int | None = None,
    progress: bool = False,
    progress_chunk_size: int = 100,
    guide_effect_strategy: str = "shared",
    guide_activity_mode: str = "always_on",
    guide_random_effects: bool = False,
    fit_perturbation_dispersion: bool = False,
    perturbation_dispersion_prior_rate: float = 10.0,
    _runner_cache: dict[str, _ReusableSVIRunner] | None = None,
) -> BetaFit:
    print("[perturbo] Fitting perturbation effects...")
    counts = data.counts
    pert_id = data.pert_id
    num_perts = len(data.pert_names)
    inferred_num_perts = _infer_num_perts(pert_id)
    if inferred_num_perts > num_perts:
        raise ValueError(
            "pert_id references more perturbations than provided in data.pert_names. "
            f"Got {inferred_num_perts} perturbations from pert_id and {num_perts} names."
        )
    num_genes = counts.shape[1]
    guide_effect_strategy, guide_activity_mode = _validate_guide_model_request(
        data,
        guide_effect_strategy=guide_effect_strategy,
        guide_activity_mode=guide_activity_mode,
        model_name=model_name,
    )
    has_guide_mapping = data.guide_matrix is not None and data.guide_to_element is not None
    random_effects_in_model = bool(guide_random_effects and has_guide_mapping)
    if guide_random_effects and has_guide_mapping and control_fit.guide_random_effect_tau is None:
        raise ValueError(
            "guide_random_effects=True in stage-2 requires control_fit.guide_random_effect_tau. "
            "Run fit_control(..., guide_random_effects=True) first."
        )
    apply_scale_inflation_fallback = bool(guide_random_effects and not has_guide_mapping)
    if apply_scale_inflation_fallback and control_fit.guide_random_effect_tau is None:
        raise ValueError(
            "guide_random_effects=True without guide mapping requires control_fit.guide_random_effect_tau "
            "for conservative scale inflation."
        )
    if apply_scale_inflation_fallback:
        print(
            "[perturbo] guide_random_effects enabled without guide mapping; using conservative stage-2 "
            "posterior scale inflation from stage-1 guide_random_effect_tau."
        )
    use_guide_shared_model = (
        guide_effect_strategy != "shared"
        or random_effects_in_model
        or (fit_perturbation_dispersion and has_guide_mapping)
    )
    num_guides = int(data.guide_matrix.shape[1]) if data.guide_matrix is not None else None
    if minibatch_size is not None and minibatch_size > counts.shape[0]:
        print(
            "[perturbo] minibatch_size exceeds dataset size; using full batch "
            f"(minibatch_size={minibatch_size}, n_cells={counts.shape[0]})."
        )
        minibatch_size = None

    size_factors = data.size_factors
    covariates = data.covariates
    if size_factors is None:
        size_factors = compute_size_factors(counts)

    model_name_lower = model_name.lower()
    if fit_perturbation_dispersion:
        if model_name_lower not in {"negbin", "nb", "censored_nb", "censored_negbin"}:
            raise ValueError(
                "fit_perturbation_dispersion=True is currently supported only for negative-binomial likelihoods."
            )
        pert_ndim = pert_id.ndim
        if pert_ndim not in {1, 2}:
            raise ValueError("fit_perturbation_dispersion=True requires 1D or 2D perturbation labels.")
        if has_guide_mapping:
            if data.guide_matrix.ndim != 2 or not design_values_are_finite_nonnegative(data.guide_matrix):
                raise ValueError("High-MOI guide_matrix must be a finite, non-negative 2D matrix.")
        elif pert_ndim == 2:
            if isinstance(pert_id, IndexedDesignMatrix):
                row_totals = np.asarray(pert_id.values).sum(axis=1)
                invalid = not design_values_are_finite_nonnegative(pert_id) or np.any(row_totals > 1)
            else:
                pert_array = np.asarray(pert_id)
                invalid = np.any(pert_array < 0) or np.any(np.asarray(pert_array.sum(axis=1)).reshape(-1) > 1)
            if invalid:
                raise ValueError(
                    "Without guide-level inputs, fit_perturbation_dispersion=True requires at most one active "
                    "perturbation per cell."
                )
        if not np.isfinite(perturbation_dispersion_prior_rate) or perturbation_dispersion_prior_rate <= 0:
            raise ValueError("perturbation_dispersion_prior_rate must be finite and > 0.")
    model_cls = _resolve_guide_shared_model(model_name) if use_guide_shared_model else _resolve_model(model_name)
    conditioned_data: dict[str, jnp.ndarray | None] = {}
    is_lognormal_model = model_name_lower in {"lognormal_nb", "lnnb"}
    is_mixture_model = model_name_lower == "mixture_nb"
    count_censoring_kwargs = _count_censoring_stage2_kwargs(
        control_fit=control_fit,
        counts=counts,
        model_name=model_name,
        count_censoring_percentile=count_censoring_percentile,
    )
    if is_mixture_model and propagate_baseline_uncertainty:
        raise ValueError(
            "propagate_baseline_uncertainty=True is not supported for model_name='mixture_nb'."
        )
    if num_factors is not None:
        conditioned_data["factor_loadings"] = control_fit.factor_loadings
    if is_lognormal_model:
        conditioned_data["noise_scale"] = control_fit.noise_scale
    if is_mixture_model:
        if control_fit.pi_outlier is None or control_fit.theta_outlier is None:
            raise ValueError(
                "model_name='mixture_nb' requires control_fit.pi_outlier and control_fit.theta_outlier from stage-1."
            )
        conditioned_data["pi_outlier"] = control_fit.pi_outlier
        conditioned_data["theta_outlier"] = control_fit.theta_outlier
        conditioned_data["outlier_mean_shift"] = (
            control_fit.outlier_mean_shift
            if control_fit.outlier_mean_shift is not None
            else jnp.zeros_like(control_fit.theta_outlier)
        )
    if covariates is not None:
        if control_fit.covariate_coef is None:
            raise ValueError(
                "Covariates were provided in data, but control_fit.covariate_coef is missing. "
                "Run fit_control with matching covariates first."
            )
        conditioned_data["covariate_coef"] = control_fit.covariate_coef
    if random_effects_in_model:
        conditioned_data["guide_random_effect_tau"] = control_fit.guide_random_effect_tau
        if control_fit.guide_random_effect_log_tau_loc is not None:
            conditioned_data["guide_random_effect_log_tau_loc"] = control_fit.guide_random_effect_log_tau_loc
        if control_fit.guide_random_effect_log_tau_scale is not None:
            conditioned_data["guide_random_effect_log_tau_scale"] = control_fit.guide_random_effect_log_tau_scale
    if not propagate_baseline_uncertainty:
        conditioned_data["beta_0"] = control_fit.beta_0
        conditioned_data["theta"] = control_fit.theta
    model = numpyro.handlers.condition(
        model_cls,
        data=conditioned_data,
    )
    dispersion_kwargs = (
        {
            "fit_perturbation_dispersion": True,
            "perturbation_dispersion_prior_rate": perturbation_dispersion_prior_rate,
        }
        if fit_perturbation_dispersion
        else {}
    )
    cacheable_runner = bool(
        _runner_cache is not None
        and minibatch_size is None
        and num_factors is None
        and data.guide_matrix is None
        and not use_guide_shared_model
    )
    reusable_runner = _runner_cache.get("runner") if cacheable_runner and _runner_cache is not None else None

    if reusable_runner is None:
        init_values = {
            "beta": jnp.zeros((num_perts, num_genes)),
        }
        if guide_effect_strategy == "relative" and num_guides is not None:
            init_values["guide_relative_efficiency"] = jnp.full((num_guides, num_genes), 0.8, dtype=jnp.float32)
        if guide_effect_strategy == "offset" and num_guides is not None:
            init_values["guide_offset"] = jnp.zeros((num_guides, num_genes), dtype=jnp.float32)
        if random_effects_in_model and num_guides is not None:
            init_values["guide_random_effect"] = jnp.zeros((num_guides, num_genes), dtype=jnp.float32)
        if minibatch_size is None:
            init_values["size_factor"] = size_factors
        if (
            num_factors is not None
            and minibatch_size is None
            and (control_fit.pca_loadings is not None or control_fit.factor_loadings is not None)
        ):
            loadings = control_fit.factor_loadings
            init_values["factor_scores"] = _project_factor_scores(
                counts,
                loadings,
                control_fit.factor_center,
            )
        if is_lognormal_model:
            init_values["noise_scale"] = control_fit.noise_scale
        init_value_fn = numpyro.infer.init_to_value(values=_float32_init_values(init_values))

        if propagate_baseline_uncertainty:
            if control_fit.baseline_posterior is None:
                raise ValueError(
                    "propagate_baseline_uncertainty=True requires stage-1 baseline posterior summary in control_fit."
                )
            baseline_posterior = control_fit.baseline_posterior
            if svi_config is not None and svi_config.num_particles == 1:
                print(
                    "[perturbo] propagate_baseline_uncertainty is enabled with num_particles=1; "
                    "this is valid but can increase gradient noise."
                )
            guide_model = numpyro.handlers.block(model, hide=["beta_0", "theta"])
        else:
            guide_model = model

        auto_guide = AutoNormal(guide_model, init_loc_fn=init_value_fn, create_plates=create_plates)
        if propagate_baseline_uncertainty:
            def guide(
                counts: jnp.ndarray | None,
                pert_id: jnp.ndarray | None,
                *,
                size_factors: jnp.ndarray | None = None,
                **kwargs: Any,
            ):
                numpyro.sample(
                    "beta_0",
                    dist.Normal(baseline_posterior.beta_0_loc, baseline_posterior.beta_0_scale),
                )
                numpyro.sample(
                    "theta",
                    dist.LogNormal(baseline_posterior.theta_log_loc, baseline_posterior.theta_log_scale),
                )
                return auto_guide(
                    counts,
                    pert_id,
                    size_factors=size_factors,
                    **kwargs,
                )

        else:
            guide = auto_guide

        if svi_config is None:
            svi_config = SVIConfig()
        optimizer = _build_optimizer(svi_config)
        svi = SVI(model, guide, optimizer, loss=_build_elbo(svi_config))
        if cacheable_runner and _runner_cache is not None:
            reusable_runner = _ReusableSVIRunner(svi=svi, auto_guide=auto_guide)
            _runner_cache["runner"] = reusable_runner
    else:
        svi = reusable_runner.svi
        auto_guide = reusable_runner.auto_guide

    rng = jax.random.PRNGKey(1)
    rng, key = jax.random.split(rng)
    if minibatch_size is None:
        result = _run_svi(
            svi,
            key,
            num_steps,
            counts,
            pert_id,
            size_factors=size_factors if use_observed_size_factors else None,
            covariates=covariates,
            cell_mask=data.cell_mask,
            progress=progress,
            progress_chunk_size=progress_chunk_size,
            reusable_runner=reusable_runner,
            num_cells=counts.shape[0],
            num_perts=num_perts,
            num_guides=num_guides,
            num_genes=num_genes,
            num_factors=num_factors,
            subsample_size=None,
            prior=prior,
            guide_matrix=data.guide_matrix if use_guide_shared_model else None,
            guide_to_element=data.guide_to_element if use_guide_shared_model else None,
            guide_effect_strategy=guide_effect_strategy,
            guide_random_effects=random_effects_in_model,
            **dispersion_kwargs,
            **count_censoring_kwargs,
        )
    else:
        result = _run_svi_minibatch(
            svi,
            key,
            num_steps,
            counts,
            pert_id,
            size_factors=size_factors if use_observed_size_factors else None,
            covariates=covariates,
            cell_mask=data.cell_mask,
            progress=progress,
            progress_chunk_size=progress_chunk_size,
            batch_size=minibatch_size,
            num_cells=counts.shape[0],
            num_perts=num_perts,
            num_guides=num_guides,
            num_genes=num_genes,
            num_factors=num_factors,
            subsample_size=None,
            prior=prior,
            guide_matrix=data.guide_matrix if use_guide_shared_model else None,
            guide_to_element=data.guide_to_element if use_guide_shared_model else None,
            guide_effect_strategy=guide_effect_strategy,
            guide_random_effects=random_effects_in_model,
            **dispersion_kwargs,
            **count_censoring_kwargs,
        )

    loc = result.params["beta_auto_loc"]
    scale = jnp.clip(result.params["beta_auto_scale"], 1e-6, None)
    if apply_scale_inflation_fallback:
        tau = jnp.asarray(control_fit.guide_random_effect_tau, dtype=scale.dtype)
        if tau.ndim != 1 or tau.shape[0] != num_genes:
            raise ValueError(
                "control_fit.guide_random_effect_tau must be a 1D array with length equal to num_genes "
                "for stage-2 scale inflation fallback."
            )
        scale = jnp.sqrt(jnp.square(scale) + jnp.square(tau)[jnp.newaxis, :])
    z_values = loc / scale
    dispersion_excess_inverse = None
    guide_dispersion_excess_inverse = None
    if fit_perturbation_dispersion:
        dispersion_posterior = auto_guide.median(result.params)
        if has_guide_mapping:
            guide_dispersion_excess_inverse = dispersion_posterior["guide_dispersion_excess_inverse"]
        else:
            dispersion_excess_inverse = dispersion_posterior["dispersion_excess_inverse"]
    guide_summary = _summarize_stage2_guide_posteriors(
        result.params,
        data=data,
        guide_effect_strategy=guide_effect_strategy,
    )
    print("[perturbo] Beta fit complete.")
    return BetaFit(
        posterior_mean=loc,
        posterior_scale=scale,
        z_values=z_values,
        losses=result.losses,
        svi_result=result.state,
        guide_effect_mean=guide_summary.get("guide_effect_mean"),
        guide_effect_scale=guide_summary.get("guide_effect_scale"),
        guide_effect_z_values=guide_summary.get("guide_effect_z_values"),
        guide_relative_efficiency_mean=guide_summary.get("guide_relative_efficiency_mean"),
        guide_relative_efficiency_scale=guide_summary.get("guide_relative_efficiency_scale"),
        guide_offset_mean=guide_summary.get("guide_offset_mean"),
        guide_offset_scale=guide_summary.get("guide_offset_scale"),
        dispersion_excess_inverse=dispersion_excess_inverse,
        guide_dispersion_excess_inverse=guide_dispersion_excess_inverse,
    )


def summarize_betas(
    beta_fit: BetaFit,
    *,
    pert_names: list[str],
    gene_names: list[str],
    single_frame: bool = False,
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    print("[perturbo] Summarizing posterior estimates...")
    mean = beta_fit.posterior_mean
    scale = beta_fit.posterior_scale
    z = beta_fit.z_values
    posterior_prob = 2 * jsp_stats.norm.sf(jnp.abs(z))

    if not single_frame:
        print("[perturbo] Returning wide DataFrames.")
        return {
            "posterior_mean": pd.DataFrame(mean, index=pert_names, columns=gene_names),
            "posterior_scale": pd.DataFrame(scale, index=pert_names, columns=gene_names),
            "posterior_prob": pd.DataFrame(posterior_prob, index=pert_names, columns=gene_names),
        }

    pert_idx = np.repeat(pert_names, len(gene_names))
    gene_idx = np.tile(gene_names, len(pert_names))
    print("[perturbo] Returning long DataFrame.")
    return pd.DataFrame(
        {
            "pert": pert_idx,
            "gene": gene_idx,
            "posterior_mean": mean.reshape(-1),
            "posterior_scale": scale.reshape(-1),
            "posterior_prob": posterior_prob.reshape(-1),
        }
    )


def _guide_effect_payload_from_fit(
    data: PerTurboData,
    beta_fit: BetaFit,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]] | None:
    if data.guide_to_element is None or data.guide_names is None:
        return None
    mapping = np.asarray(data.guide_to_element, dtype=np.float32)
    guide_names = list(data.guide_names)
    parent_elements: list[str] = []
    for row in np.asarray(mapping, dtype=bool):
        idx = np.flatnonzero(row)
        if idx.size == 0:
            parent_elements.append("")
        elif idx.size == 1:
            parent_elements.append(str(data.pert_names[int(idx[0])]))
        else:
            parent_elements.append("|".join(str(data.pert_names[int(i)]) for i in idx.tolist()))

    if beta_fit.guide_effect_mean is not None and beta_fit.guide_effect_scale is not None:
        return (
            np.asarray(beta_fit.guide_effect_mean),
            np.asarray(beta_fit.guide_effect_scale),
            guide_names,
            parent_elements,
        )
    effect_loc = mapping @ np.asarray(beta_fit.posterior_mean)
    effect_scale = np.clip(mapping @ np.asarray(beta_fit.posterior_scale), 1e-6, None)
    return effect_loc, effect_scale, guide_names, parent_elements


def _save_loss_plot(losses: jnp.ndarray | np.ndarray, path: str | Path, title: str) -> None:
    arr = np.asarray(losses).flatten()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(arr)), arr, linewidth=1.2)
    ax.set_xlabel("SVI iteration")
    ax.set_ylabel("ELBO loss")
    ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_multi_loss_plot(
    losses_list: list[jnp.ndarray | np.ndarray],
    path: str | Path,
    title: str,
    *,
    labels: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, losses in enumerate(losses_list):
        arr = np.asarray(losses).flatten()
        label = labels[idx] if labels is not None and idx < len(labels) else f"chunk {idx + 1}"
        ax.plot(np.arange(len(arr)), arr, linewidth=1.2, alpha=0.85, label=label)
    ax.set_xlabel("SVI iteration")
    ax.set_ylabel("ELBO loss")
    ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _load_from_path(path: str):
    if path.endswith(".h5ad"):
        if ad is None:
            raise ImportError("anndata is required to read .h5ad files.")
        return ad.read_h5ad(path)
    if path.endswith(".h5mu"):
        if md is None:
            raise ImportError("mudata is required to read .h5mu files.")
        return md.read_h5mu(path)
    raise ValueError("Unsupported file extension. Use .h5ad or .h5mu")


def _load_from_path_with_backing(path: str, *, backed: bool = False):
    if backed and path.endswith(".h5ad"):
        if ad is None:
            raise ImportError("anndata is required to read .h5ad files.")
        return ad.read_h5ad(path, backed="r")
    if backed and path.endswith(".h5mu"):
        if md is None:
            raise ImportError("mudata is required to read .h5mu files.")
        try:
            return md.read_h5mu(path, backed="r")
        except TypeError:
            return md.read_h5mu(path)
    return _load_from_path(path)


def _build_obs_membership(
    pert_series: pd.Series,
    pert_names: list[str],
) -> list[np.ndarray]:
    pert_cat = pd.Categorical(pert_series.astype(str), categories=pert_names)
    codes = np.asarray(pert_cat.codes, dtype=np.int64)
    valid = codes >= 0
    if not np.any(valid):
        return [np.array([], dtype=np.int64) for _ in pert_names]
    cell_idx = np.flatnonzero(valid)
    codes_valid = codes[valid]
    order = np.argsort(codes_valid, kind="stable")
    sorted_codes = codes_valid[order]
    sorted_cells = cell_idx[order]
    counts = np.bincount(sorted_codes, minlength=len(pert_names))
    split = np.cumsum(counts)[:-1]
    return [arr.astype(np.int64, copy=False) for arr in np.split(sorted_cells, split)]


def _build_matrix_membership(
    pert_matrix: Any,
    *,
    num_perts: int,
) -> list[np.ndarray]:
    if hasattr(pert_matrix, "tocsc"):
        matrix_csc = pert_matrix.tocsc()
        indptr = np.asarray(matrix_csc.indptr)
        indices = np.asarray(matrix_csc.indices)
        membership = []
        for col in range(num_perts):
            start = int(indptr[col])
            end = int(indptr[col + 1])
            membership.append(indices[start:end].astype(np.int64, copy=False))
        return membership

    membership = []
    for col in range(num_perts):
        col_view = pert_matrix[:, col]
        if hasattr(col_view, "toarray"):
            col_arr = np.asarray(col_view.toarray()).reshape(-1)
        else:
            col_arr = np.asarray(col_view).reshape(-1)
        membership.append(np.flatnonzero(col_arr).astype(np.int64, copy=False))
    return membership


def _construct_perturbation_chunks(
    pert_names: list[str],
    membership: list[np.ndarray],
    *,
    max_chunk_size: int,
    max_perturbations_per_chunk: int | None,
) -> list[_PerturbationChunk]:
    if max_chunk_size < 1:
        raise ValueError("--max-chunk-size must be >= 1.")
    if max_perturbations_per_chunk is not None and max_perturbations_per_chunk < 1:
        raise ValueError("--perturbation-chunk-size must be >= 1 when provided.")
    if len(pert_names) != len(membership):
        raise ValueError("pert_names and membership length mismatch.")

    n_cells = 0
    for rows in membership:
        rows_arr = np.asarray(rows)
        if rows_arr.size > 0:
            n_cells = max(n_cells, int(rows_arr.max()) + 1)

    chunks: list[_PerturbationChunk] = []
    oversized: list[tuple[str, int]] = []
    marks = np.zeros(n_cells if n_cells > 0 else 1, dtype=np.int32)
    token = 1
    current_names: list[str] = []
    current_indices: list[int] = []
    current_parts: list[np.ndarray] = []
    current_cells = 0

    def _finalize_current() -> None:
        if not current_indices:
            return
        if current_parts:
            cell_indices = np.concatenate(current_parts)
        else:
            cell_indices = np.array([], dtype=np.int64)
        chunks.append(
            _PerturbationChunk(
                pert_names=list(current_names),
                pert_indices=np.asarray(current_indices, dtype=np.int32),
                cell_indices=cell_indices,
            )
        )

    def _new_cells_for(rows: np.ndarray) -> int:
        if rows.size == 0:
            return 0
        unseen = marks[rows] != token
        return int(np.count_nonzero(unseen))

    def _add_rows(rows: np.ndarray) -> int:
        if rows.size == 0:
            return 0
        unseen = marks[rows] != token
        if not np.any(unseen):
            return 0
        added = rows[unseen]
        marks[added] = token
        current_parts.append(added)
        return int(added.size)

    for pert_idx, (name, rows_raw) in enumerate(zip(pert_names, membership, strict=True)):
        rows = np.asarray(rows_raw, dtype=np.int64)

        hits_pert_cap = max_perturbations_per_chunk is not None and len(current_indices) >= max_perturbations_per_chunk
        candidate_cells = current_cells + _new_cells_for(rows)
        hits_cell_cap = bool(current_indices) and candidate_cells > max_chunk_size
        if hits_pert_cap or hits_cell_cap:
            _finalize_current()
            current_names = []
            current_indices = []
            current_parts = []
            current_cells = 0
            token += 1
            if token == np.iinfo(np.int32).max:
                marks.fill(0)
                token = 1
            candidate_cells = _new_cells_for(rows)

        if candidate_cells > max_chunk_size:
            # A single perturbation larger than the cap cannot be split: every one
            # of its cells is needed to estimate its own effect. Refusing the run
            # would make the cap a hard limit on the screens the tool accepts, and
            # screens do exist with tens of thousands of cells behind one
            # perturbation. It gets a chunk to itself instead, and the caller is
            # told, because that chunk's peak memory is the run's peak memory.
            oversized.append((name, candidate_cells))

        current_names.append(name)
        current_indices.append(pert_idx)
        current_cells += _add_rows(rows)

    _finalize_current()
    if oversized:
        worst = max(count for _, count in oversized)
        listed = ", ".join(f"{name} ({count:,} cells)" for name, count in oversized[:3])
        more = f" and {len(oversized) - 3} more" if len(oversized) > 3 else ""
        print(
            f"[perturbo] {len(oversized)} perturbation(s) exceed --max-chunk-size={max_chunk_size:,} "
            f"and each takes a chunk of its own: {listed}{more}. Peak memory follows the largest "
            f"({worst:,} cells); lower --crt-gene-chunk-size or --minibatch-size if that does not fit."
        )
    return chunks


def _subset_cortado_data_for_chunk(
    data: PerTurboData,
    chunk: _PerturbationChunk,
) -> PerTurboData:
    row_idx = np.asarray(chunk.cell_indices, dtype=np.int32)
    chunk_indices = np.asarray(chunk.pert_indices, dtype=np.int32)

    counts_chunk = data.counts[row_idx]
    size_factors_chunk = data.size_factors[row_idx] if data.size_factors is not None else None
    covariates_chunk = data.covariates[row_idx] if data.covariates is not None else None
    pert_names_chunk = [data.pert_names[int(i)] for i in chunk_indices.tolist()]

    pert_arr = np.asarray(data.pert_id)
    if pert_arr.ndim == 1:
        selected_codes = pert_arr[row_idx]
        code_map = np.full((len(data.pert_names),), -1, dtype=np.int32)
        code_map[chunk_indices] = np.arange(chunk_indices.size, dtype=np.int32)
        local_codes = code_map[selected_codes]
        if np.any(local_codes < 0):
            raise RuntimeError("Chunk remapping encountered perturbation codes outside chunk selection.")
        pert_id_chunk = jnp.asarray(local_codes, dtype=jnp.int32)
    elif pert_arr.ndim == 2:
        pert_id_chunk = jnp.asarray(pert_arr[row_idx][:, chunk_indices], dtype=jnp.bool_)
    else:
        raise ValueError("pert_id must be 1D or 2D.")

    guide_matrix_chunk = None
    guide_to_element_chunk = None
    guide_names_chunk = None
    if data.guide_matrix is not None and data.guide_to_element is not None and data.guide_names is not None:
        guide_to_element_global = np.asarray(data.guide_to_element, dtype=np.float32)
        if guide_to_element_global.ndim != 2:
            raise ValueError("guide_to_element must be 2D when guide structure is provided.")
        guide_to_element_chunk_full = guide_to_element_global[:, chunk_indices]

        guide_matrix_global = np.asarray(data.guide_matrix)
        if guide_matrix_global.ndim == 1:
            guide_assign = guide_matrix_global[row_idx].astype(np.int64, copy=False)
            n_guides_global = int(guide_to_element_global.shape[0])
            if np.any(guide_assign < 0) or np.any(guide_assign >= n_guides_global):
                raise ValueError("guide assignment vector contains out-of-range guide indices.")
            guide_matrix_rows = np.zeros((guide_assign.size, n_guides_global), dtype=np.float32)
            guide_matrix_rows[np.arange(guide_assign.size, dtype=np.int64), guide_assign] = 1.0
        elif guide_matrix_global.ndim == 2:
            guide_matrix_rows = np.asarray(guide_matrix_global[row_idx], dtype=np.float32)
        else:
            raise ValueError("guide_matrix must be 1D assignment vector or 2D matrix.")

        parent_mask = guide_to_element_chunk_full.sum(axis=1) > 0
        present_mask = guide_matrix_rows.sum(axis=0) > 0
        guide_keep = parent_mask & present_mask
        if not np.any(guide_keep):
            guide_keep = parent_mask

        guide_matrix_chunk = jnp.asarray(guide_matrix_rows[:, guide_keep], dtype=jnp.float32)
        guide_to_element_chunk = jnp.asarray(guide_to_element_chunk_full[guide_keep], dtype=jnp.float32)
        guide_names_chunk = [str(name) for keep, name in zip(guide_keep.tolist(), data.guide_names, strict=True) if keep]

    return PerTurboData(
        counts=counts_chunk,
        pert_id=pert_id_chunk,
        pert_names=pert_names_chunk,
        gene_names=list(data.gene_names),
        cell_mask=data.cell_mask[row_idx] if data.cell_mask is not None else None,
        size_factors=size_factors_chunk,
        covariates=covariates_chunk,
        covariate_names=None if data.covariate_names is None else list(data.covariate_names),
        guide_matrix=guide_matrix_chunk,
        guide_names=guide_names_chunk,
        guide_to_element=guide_to_element_chunk,
        library_size_center_log_mean=data.library_size_center_log_mean,
    )


def _pad_cortado_data_for_chunk(
    data: PerTurboData,
    *,
    cell_capacity: int,
    pert_capacity: int,
    guide_capacity: int | None = None,
) -> PerTurboData:
    """Pad one perturbation chunk into a fixed-shape inference buffer.

    Padded cells are marked through ``cell_mask`` and are excluded by the
    model's mask handler. Perturbation and guide padding is all-zero, so no
    padded effect is referenced by a real observation.
    """

    n_cells = int(data.counts.shape[0])
    n_perts = len(data.pert_names)
    if n_cells > cell_capacity:
        raise ValueError(f"Chunk has {n_cells} cells but cell_capacity is {cell_capacity}.")
    if n_perts > pert_capacity:
        raise ValueError(f"Chunk has {n_perts} perturbations but pert_capacity is {pert_capacity}.")

    cell_pad = cell_capacity - n_cells
    pert_pad = pert_capacity - n_perts
    counts = jnp.pad(data.counts, ((0, cell_pad), (0, 0)))

    pert_arr = jnp.asarray(data.pert_id)
    if pert_arr.ndim == 1:
        # Padded rows are masked, so assigning them to the first real
        # perturbation keeps indexing valid without affecting the likelihood.
        pert_id = jnp.pad(pert_arr, ((0, cell_pad),), constant_values=0)
    elif pert_arr.ndim == 2:
        pert_id = jnp.pad(pert_arr, ((0, cell_pad), (0, pert_pad)))
    else:
        raise ValueError("pert_id must be 1D or 2D.")

    if data.cell_mask is None:
        real_mask = jnp.ones((n_cells,), dtype=bool)
    else:
        real_mask = jnp.asarray(data.cell_mask, dtype=bool).reshape((-1,))
    cell_mask = jnp.pad(real_mask, ((0, cell_pad),), constant_values=False)

    def _pad_cell_axis(values: jnp.ndarray | None) -> jnp.ndarray | None:
        if values is None:
            return None
        return jnp.pad(values, ((0, cell_pad),) + ((0, 0),) * (values.ndim - 1))

    # Preserve the unpadded initialization statistic. Recomputing it after
    # appending all-zero rows would shift every real cell's centered factor.
    real_size_factors = data.size_factors
    if real_size_factors is None:
        real_size_factors = compute_size_factors(data.counts)
    size_factors = _pad_cell_axis(real_size_factors)
    covariates = _pad_cell_axis(data.covariates)

    guide_matrix = None
    guide_to_element = None
    guide_names = None
    if data.guide_matrix is not None:
        n_guides = int(data.guide_matrix.shape[1])
        if guide_capacity is None:
            guide_capacity = n_guides
        if n_guides > guide_capacity:
            raise ValueError(f"Chunk has {n_guides} guides but guide_capacity is {guide_capacity}.")
        guide_pad = guide_capacity - n_guides
        guide_matrix = jnp.pad(data.guide_matrix, ((0, cell_pad), (0, guide_pad)))
        if data.guide_to_element is None:
            raise ValueError("guide_to_element is required when guide_matrix is provided.")
        guide_to_element = jnp.pad(data.guide_to_element, ((0, guide_pad), (0, pert_pad)))
        source_names = data.guide_names or [f"guide_{idx}" for idx in range(n_guides)]
        guide_names = list(source_names) + [f"{_PADDING_GUIDE_PREFIX}{idx}" for idx in range(guide_pad)]

    pert_names = list(data.pert_names) + [f"__padding_pert_{idx}" for idx in range(pert_pad)]
    return PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=pert_names,
        gene_names=list(data.gene_names),
        cell_mask=cell_mask,
        size_factors=size_factors,
        covariates=covariates,
        covariate_names=None if data.covariate_names is None else list(data.covariate_names),
        guide_matrix=guide_matrix,
        guide_names=guide_names,
        guide_to_element=guide_to_element,
        library_size_center_log_mean=data.library_size_center_log_mean,
    )


def _fit_perturbation_effects_chunked(
    data: PerTurboData,
    control_fit: ControlFit,
    *,
    num_steps: int = 1000,
    prior: str = "normal",
    svi_config: SVIConfig | None = None,
    model_name: str = "negbin",
    num_factors: int | None = None,
    propagate_baseline_uncertainty: bool = False,
    use_observed_size_factors: bool = False,
    count_censoring_percentile: float | None = None,
    minibatch_size: int | None = None,
    progress: bool = False,
    progress_chunk_size: int = 100,
    guide_effect_strategy: str = "shared",
    guide_activity_mode: str = "always_on",
    guide_random_effects: bool = False,
    fit_perturbation_dispersion: bool = False,
    perturbation_dispersion_prior_rate: float = 10.0,
    max_chunk_size: int = 50_000,
    max_perturbations_per_chunk: int | None = None,
) -> BetaFit:
    if len(data.pert_names) == 0:
        raise ValueError("Chunked beta fitting requires at least one perturbation.")

    pert_arr = np.asarray(data.pert_id)
    if pert_arr.ndim == 1:
        membership = [np.flatnonzero(pert_arr == idx).astype(np.int64, copy=False) for idx in range(len(data.pert_names))]
    elif pert_arr.ndim == 2:
        membership = _build_matrix_membership(pert_arr, num_perts=len(data.pert_names))
    else:
        raise ValueError("pert_id must be 1D or 2D.")

    chunks = _construct_perturbation_chunks(
        list(data.pert_names),
        membership,
        max_chunk_size=max_chunk_size,
        max_perturbations_per_chunk=max_perturbations_per_chunk,
    )
    if not chunks:
        raise RuntimeError("No perturbation chunks were constructed.")

    n_perts = len(data.pert_names)
    n_genes = int(np.asarray(data.counts).shape[1])
    posterior_mean = np.zeros((n_perts, n_genes), dtype=np.float32)
    posterior_scale = np.zeros((n_perts, n_genes), dtype=np.float32)
    z_values = np.zeros((n_perts, n_genes), dtype=np.float32)
    dispersion_excess_inverse = (
        np.zeros((n_perts, n_genes), dtype=np.float32) if fit_perturbation_dispersion else None
    )
    losses_parts: list[jnp.ndarray] = []
    last_state = None

    guide_eff_global: np.ndarray | None = None
    guide_name_to_global: dict[str, int] | None = None
    if guide_effect_strategy == "relative" and data.guide_names is not None:
        guide_name_to_global = {str(name): idx for idx, name in enumerate(data.guide_names)}
        guide_eff_global = np.full((len(data.guide_names), n_genes), np.nan, dtype=np.float32)

    prepared_chunks = [(chunk, _subset_cortado_data_for_chunk(data, chunk)) for chunk in chunks]
    cell_capacity = max(int(chunk_data.counts.shape[0]) for _, chunk_data in prepared_chunks)
    pert_capacity = max(len(chunk_data.pert_names) for _, chunk_data in prepared_chunks)
    guide_capacity = None
    if data.guide_matrix is not None:
        guide_capacity = max(
            int(chunk_data.guide_matrix.shape[1])
            for _, chunk_data in prepared_chunks
            if chunk_data.guide_matrix is not None
        )

    runner_cache: dict[str, _ReusableSVIRunner] = {}
    for chunk, chunk_data in prepared_chunks:
        padded_chunk_data = _pad_cortado_data_for_chunk(
            chunk_data,
            cell_capacity=cell_capacity,
            pert_capacity=pert_capacity,
            guide_capacity=guide_capacity,
        )
        chunk_fit = fit_perturbation_effects(
            padded_chunk_data,
            control_fit,
            num_steps=num_steps,
            prior=prior,
            svi_config=svi_config,
            model_name=model_name,
            num_factors=num_factors,
            propagate_baseline_uncertainty=propagate_baseline_uncertainty,
            use_observed_size_factors=use_observed_size_factors,
            count_censoring_percentile=count_censoring_percentile,
            minibatch_size=minibatch_size,
            progress=progress,
            progress_chunk_size=progress_chunk_size,
            guide_effect_strategy=guide_effect_strategy,
            guide_activity_mode=guide_activity_mode,
            guide_random_effects=guide_random_effects,
            fit_perturbation_dispersion=fit_perturbation_dispersion,
            perturbation_dispersion_prior_rate=perturbation_dispersion_prior_rate,
            _runner_cache=runner_cache,
        )
        n_chunk_perts = len(chunk_data.pert_names)
        posterior_mean[np.asarray(chunk.pert_indices)] = np.asarray(chunk_fit.posterior_mean)[:n_chunk_perts]
        posterior_scale[np.asarray(chunk.pert_indices)] = np.asarray(chunk_fit.posterior_scale)[:n_chunk_perts]
        z_values[np.asarray(chunk.pert_indices)] = np.asarray(chunk_fit.z_values)[:n_chunk_perts]
        if dispersion_excess_inverse is not None and chunk_fit.dispersion_excess_inverse is not None:
            dispersion_excess_inverse[np.asarray(chunk.pert_indices)] = np.asarray(
                chunk_fit.dispersion_excess_inverse
            )[:n_chunk_perts]
        last_state = chunk_fit.svi_result
        if chunk_fit.losses.size > 0:
            losses_parts.append(chunk_fit.losses)

        if (
            guide_eff_global is not None
            and guide_name_to_global is not None
            and chunk_fit.guide_relative_efficiency_mean is not None
            and chunk_data.guide_names is not None
        ):
            rel = np.asarray(chunk_fit.guide_relative_efficiency_mean, dtype=np.float32)
            for local_idx, guide_name in enumerate(chunk_data.guide_names):
                global_idx = guide_name_to_global.get(str(guide_name))
                if global_idx is not None:
                    guide_eff_global[global_idx] = rel[local_idx]

    losses = jnp.concatenate(losses_parts) if losses_parts else jnp.array([])
    guide_efficiency = None
    if guide_eff_global is not None:
        guide_efficiency = jnp.asarray(np.nan_to_num(guide_eff_global, nan=0.0), dtype=jnp.float32)

    return BetaFit(
        posterior_mean=jnp.asarray(posterior_mean),
        posterior_scale=jnp.asarray(posterior_scale),
        z_values=jnp.asarray(z_values),
        losses=losses,
        svi_result=last_state,
        guide_relative_efficiency_mean=guide_efficiency,
        dispersion_excess_inverse=(
            None if dispersion_excess_inverse is None else jnp.asarray(dispersion_excess_inverse)
        ),
    )
