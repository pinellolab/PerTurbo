"""Notebook-friendly fitted-model workflow for perturbo."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import core
from .io import MuDataSetup, control_fit_arrays, get_mudata_setup, load_fit_bundle, save_fit_bundle, setup_mudata
from .preprocessing import to_dense_array
from .results import PosteriorMedians, build_element_effects_df, build_guide_effects_df, extract_parameter_table
from .training_schedule import resolve_training_schedule


def _infer_device(accelerator: str | None, device: str | None) -> str | None:
    if device is not None:
        return device
    if accelerator is None:
        return None
    if accelerator.lower() in {"gpu", "cuda"}:
        return "gpu"
    if accelerator.lower() == "cpu":
        return "cpu"
    return None


def _to_numpy_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value)


def _num_guides(mdata, setup: MuDataSetup) -> int:
    return int(mdata[setup.perturbation_modality].n_vars)


def _guide_efficacy_from_model(
    *,
    beta_fit: core.BetaFit,
    guide_effect_strategy: str,
    n_guides: int,
) -> np.ndarray:
    strategy = str(guide_effect_strategy).lower()
    if strategy in {"shared", "singleton"}:
        return np.ones((n_guides,), dtype=np.float32)
    if strategy != "relative":
        raise ValueError(
            "guide_effect_strategy must be one of: 'shared', 'singleton', or 'relative'."
        )

    relative = beta_fit.guide_relative_efficiency_mean
    if relative is None:
        raise RuntimeError(
            "guide_relative_efficiency_mean is missing for guide_effect_strategy='relative'."
        )

    rel = np.asarray(relative, dtype=np.float32)
    if rel.ndim == 0:
        return np.full((n_guides,), float(rel), dtype=np.float32)
    if rel.shape[0] != n_guides:
        raise ValueError(
            f"guide_relative_efficiency_mean first dimension ({rel.shape[0]}) "
            f"must match number of guides ({n_guides})."
        )
    if rel.ndim == 1:
        eff = rel
    else:
        eff = rel.reshape(n_guides, -1).mean(axis=1)
    return np.clip(np.asarray(eff, dtype=np.float32), a_min=0.0, a_max=None)


def _control_cell_keep_mask(mdata, setup: MuDataSetup) -> np.ndarray | None:
    if setup.control_substring:
        return None
    pert = mdata[setup.perturbation_modality]
    matrix = core._get_layer_matrix(pert, setup.perturbation_layer)
    total = np.asarray(matrix.sum(axis=1)).reshape(-1)
    mask = total == 0
    if np.any(mask):
        return mask
    return None


def _beta_arrays(beta_fit: core.BetaFit) -> dict[str, Any]:
    return {
        "posterior_mean": beta_fit.posterior_mean,
        "posterior_scale": beta_fit.posterior_scale,
        "z_values": beta_fit.z_values,
        "losses": beta_fit.losses,
        "dispersion_excess_inverse": beta_fit.dispersion_excess_inverse,
    }


def _guide_posterior_arrays(beta_fit: core.BetaFit) -> dict[str, Any]:
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


def _restore_control_fit(arrays: dict[str, np.ndarray]) -> core.ControlFit:
    baseline = None
    if "baseline_beta_0_loc" in arrays:
        baseline = core.BaselinePosteriorSummary(
            beta_0_loc=arrays["baseline_beta_0_loc"],
            beta_0_scale=arrays["baseline_beta_0_scale"],
            theta_log_loc=arrays["baseline_theta_log_loc"],
            theta_log_scale=arrays["baseline_theta_log_scale"],
        )
    return core.ControlFit(
        beta_0=arrays["beta_0"],
        theta=arrays["theta"],
        noise_scale=arrays["noise_scale"],
        factor_loadings=arrays.get("factor_loadings"),
        factor_scores=arrays.get("factor_scores"),
        factor_center=arrays.get("factor_center"),
        pca_loadings=arrays.get("pca_loadings"),
        size_factors=arrays["size_factors"],
        losses=arrays.get("losses", np.array([])),
        svi_result=None,
        baseline_posterior=baseline,
        pi_outlier=arrays.get("pi_outlier"),
        theta_outlier=arrays.get("theta_outlier"),
        outlier_mean_shift=arrays.get("outlier_mean_shift"),
        covariate_coef=arrays.get("covariate_coef"),
        guide_random_effect_tau=arrays.get("guide_random_effect_tau"),
        guide_random_effect_log_tau_loc=arrays.get("guide_random_effect_log_tau_loc"),
        guide_random_effect_log_tau_scale=arrays.get("guide_random_effect_log_tau_scale"),
        count_censoring_threshold=arrays.get("count_censoring_threshold"),
    )


def _restore_beta_fit(
    arrays: dict[str, np.ndarray],
    guide_arrays: dict[str, np.ndarray] | None = None,
) -> core.BetaFit:
    guide_arrays = guide_arrays or {}
    return core.BetaFit(
        posterior_mean=arrays["posterior_mean"],
        posterior_scale=arrays["posterior_scale"],
        z_values=arrays["z_values"],
        losses=arrays.get("losses", np.array([])),
        svi_result=None,
        guide_effect_mean=guide_arrays.get("guide_effect_mean"),
        guide_effect_scale=guide_arrays.get("guide_effect_scale"),
        guide_effect_z_values=guide_arrays.get("guide_effect_z_values"),
        guide_relative_efficiency_mean=guide_arrays.get("guide_relative_efficiency_mean"),
        guide_relative_efficiency_scale=guide_arrays.get("guide_relative_efficiency_scale"),
        guide_offset_mean=guide_arrays.get("guide_offset_mean"),
        guide_offset_scale=guide_arrays.get("guide_offset_scale"),
        dispersion_excess_inverse=arrays.get("dispersion_excess_inverse"),
        guide_dispersion_excess_inverse=guide_arrays.get("guide_dispersion_excess_inverse"),
    )


def _normalize_guide_effect_strategy(
    guide_effect_strategy: str | None,
    efficiency_mode: str | None,
) -> str:
    if guide_effect_strategy is not None:
        value = str(guide_effect_strategy).lower()
    elif efficiency_mode is not None:
        value = str(efficiency_mode).lower()
    else:
        value = "shared"
    if value in {"shared", "relative"}:
        return value
    if value in {"scaled", "default", "same"}:
        return "shared"
    raise ValueError(
        "guide_effect_strategy must be one of: 'shared' or 'relative'."
    )


def _guide_mapping_payload(model: "PerTurboModel") -> tuple[np.ndarray, list[str], list[str]] | None:
    if model.setup.guide_by_element_key is None:
        return None
    pert = model.adata[model.setup.perturbation_modality]
    if model.setup.guide_by_element_key not in pert.varm:
        return None
    mapping = to_dense_array(pert.varm[model.setup.guide_by_element_key]).astype(bool, copy=False)
    guide_names = list(pert.var_names.astype(str))
    element_names = model.element_names
    return mapping, guide_names, element_names


def _guide_parent_elements(mapping: np.ndarray, element_names: list[str]) -> list[str]:
    parents: list[str] = []
    for row in np.asarray(mapping, dtype=bool):
        idx = np.flatnonzero(row)
        if idx.size == 0:
            parents.append("")
        elif idx.size == 1:
            parents.append(str(element_names[int(idx[0])]))
        else:
            parents.append("|".join(str(element_names[int(i)]) for i in idx.tolist()))
    return parents


class _GuideCompat:
    def __init__(self, model: "PerTurboModel") -> None:
        self._model = model

    def median(self) -> dict[str, "_CompatTensor"]:
        medians = self._model.posterior_medians().to_dict()
        return {key: _CompatTensor(value) for key, value in medians.items()}


class _CompatTensor:
    def __init__(self, value: Any) -> None:
        self._value = np.asarray(value)

    def detach(self) -> "_CompatTensor":
        return self

    def cpu(self) -> "_CompatTensor":
        return self

    def numpy(self) -> np.ndarray:
        return np.asarray(self._value)

    def __array__(self, dtype=None) -> np.ndarray:
        arr = np.asarray(self._value)
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr


class _ModuleCompat:
    def __init__(self, model: "PerTurboModel") -> None:
        self._model = model
        self.guide = _GuideCompat(model)

    def eval(self) -> "_ModuleCompat":
        return self

    def state_dict(self) -> dict[str, np.ndarray]:
        raise NotImplementedError(
            "PyTorch-style state_dict() is not supported in perturbo bundles. "
            "Use the saved fit bundle plus stateless simulation helpers instead."
        )

    def load_state_dict(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError(
            "PyTorch-style load_state_dict() is not supported in perturbo bundles."
        )


class PerTurboModel:
    """Notebook-friendly fitted model with a PerTurbo-like public workflow surface."""

    def __init__(
        self,
        mdata,
        *,
        setup: MuDataSetup | None = None,
        likelihood: str = "lnnb",
        effect_prior_dist: str = "normal",
        efficiency_mode: str | None = "scaled",
        guide_effect_strategy: str | None = None,
        guide_activity_mode: str = "always_on",
        n_factors: int | None = None,
        clip_gene_expression_percentile: float | None = None,
        winsorize_gene_expression: bool = False,
        gene_outlier_threshold_floor: int = 2,
        count_censoring_percentile: float | None = None,
        guide_random_effects: bool = False,
        fit_perturbation_dispersion: bool = False,
        perturbation_dispersion_prior_rate: float = 10.0,
        fit_guide_efficacy: bool | None = None,
        library_size_center_log_mean: float | None = None,
        svi_config: core.SVIConfig | None = None,
    ) -> None:
        self.adata = mdata
        self.setup = setup or get_mudata_setup(mdata)
        self.likelihood = core._normalize_likelihood_name(likelihood)
        self.effect_prior_dist = effect_prior_dist
        normalized_strategy = _normalize_guide_effect_strategy(guide_effect_strategy, efficiency_mode)
        normalized_strategy, normalized_activity = core._validate_guide_strategy(
            normalized_strategy,
            guide_activity_mode,
        )
        self.guide_effect_strategy = normalized_strategy
        self.guide_activity_mode = normalized_activity
        self.efficiency_mode = self.guide_effect_strategy
        self.n_factors = n_factors
        self.clip_gene_expression_percentile = clip_gene_expression_percentile
        self.winsorize_gene_expression = winsorize_gene_expression
        self.gene_outlier_threshold_floor = gene_outlier_threshold_floor
        self.count_censoring_percentile = count_censoring_percentile
        self.guide_random_effects = bool(guide_random_effects)
        self.fit_perturbation_dispersion = bool(fit_perturbation_dispersion)
        self.perturbation_dispersion_prior_rate = float(perturbation_dispersion_prior_rate)
        self.fit_guide_efficacy = fit_guide_efficacy
        self.library_size_center_log_mean = library_size_center_log_mean
        self.svi_config = svi_config or core.SVIConfig(step_size=0.01)
        self.control_fit: core.ControlFit | None = None
        self.beta_fit: core.BetaFit | None = None
        self.covariate_transform_state: core.CovariateTransformState | None = None
        self.history: dict[str, pd.DataFrame] = {}
        self._guide_efficacy: np.ndarray | None = None
        self.module = _ModuleCompat(self)

    @classmethod
    def setup_mudata(cls, mdata, **kwargs: Any) -> MuDataSetup:
        return setup_mudata(mdata, **kwargs)

    @classmethod
    def load(cls, bundle_dir: str | Path, adata=None) -> "PerTurboModel":
        bundle_path = Path(bundle_dir)
        is_light_bundle = not (bundle_path / "mdata.h5mu").exists()
        loaded = load_fit_bundle(bundle_path, source_data=adata if is_light_bundle else None)
        mdata = loaded["mdata"] if (adata is None or is_light_bundle) else adata
        setup = get_mudata_setup(mdata)
        metadata = loaded["metadata"]
        model = cls(
            mdata,
            setup=setup,
            likelihood=metadata["likelihood"],
            effect_prior_dist=metadata["effect_prior_dist"],
            efficiency_mode=metadata.get("efficiency_mode"),
            guide_effect_strategy=metadata.get("guide_effect_strategy"),
            guide_activity_mode=metadata.get("guide_activity_mode", "always_on"),
            n_factors=metadata.get("n_factors"),
            clip_gene_expression_percentile=metadata.get("clip_gene_expression_percentile"),
            winsorize_gene_expression=metadata.get("winsorize_gene_expression", False),
            gene_outlier_threshold_floor=metadata.get("gene_outlier_threshold_floor", 2),
            count_censoring_percentile=metadata.get("count_censoring_percentile"),
            guide_random_effects=metadata.get("guide_random_effects", False),
            fit_perturbation_dispersion=metadata.get("fit_perturbation_dispersion", False),
            perturbation_dispersion_prior_rate=metadata.get("perturbation_dispersion_prior_rate", 10.0),
            library_size_center_log_mean=metadata.get("library_size_center_log_mean"),
            svi_config=core.SVIConfig(**metadata.get("svi_config", {})),
        )
        model.control_fit = _restore_control_fit(loaded["control_arrays"])
        model.beta_fit = _restore_beta_fit(loaded["beta_arrays"], loaded.get("guide_posteriors"))
        covariate_state_payload = metadata.get("covariate_transform_state")
        if covariate_state_payload is not None:
            model.covariate_transform_state = core.CovariateTransformState(**covariate_state_payload)
        loaded_guide_efficacy = loaded.get("guide_efficacy")
        if loaded_guide_efficacy is not None:
            model._guide_efficacy = np.asarray(loaded_guide_efficacy, dtype=np.float32).reshape(-1)
        else:
            model._guide_efficacy = _guide_efficacy_from_model(
                beta_fit=model.beta_fit,
                guide_effect_strategy=model.guide_effect_strategy,
                n_guides=_num_guides(model.adata, model.setup),
            )
        control_loss = np.asarray(model.control_fit.losses).reshape(-1)
        beta_loss = np.asarray(model.beta_fit.losses).reshape(-1)
        history_len = max(len(control_loss), len(beta_loss))
        history = {}
        if history_len > 0:
            history["control_loss"] = np.full(history_len, np.nan, dtype=float)
            history["beta_loss"] = np.full(history_len, np.nan, dtype=float)
            history["control_loss"][: len(control_loss)] = control_loss
            history["beta_loss"][: len(beta_loss)] = beta_loss
        model.history = {"elbo_train": pd.DataFrame(history)}
        return model

    @property
    def gene_names(self) -> list[str]:
        if self.beta_fit is not None:
            return list(self.adata[self.setup.rna_modality].var_names.astype(str))
        return list(self.adata[self.setup.rna_modality].var_names.astype(str))

    @property
    def element_names(self) -> list[str]:
        if self.setup.guide_by_element_key is None:
            return list(self.adata[self.setup.perturbation_modality].var_names.astype(str))
        return core._load_perturbation_element_mapping(
            self.adata,
            perturbation_modality_key=self.setup.perturbation_modality,
            perturbation_element_varm_key=self.setup.guide_by_element_key,
            perturbation_element_names_uns_key=self.setup.guide_element_uns_key,
        )[1]

    @property
    def guide_names(self) -> list[str]:
        return list(self.adata[self.setup.perturbation_modality].var_names.astype(str))

    def _guide_effect_payload(self) -> tuple[np.ndarray, np.ndarray, list[str], list[str]] | None:
        if self.beta_fit is None:
            return None
        mapping_payload = _guide_mapping_payload(self)
        if mapping_payload is None:
            return None
        mapping, guide_names, element_names = mapping_payload
        parent_elements = _guide_parent_elements(mapping, element_names)
        if self.beta_fit.guide_effect_mean is not None and self.beta_fit.guide_effect_scale is not None:
            return (
                np.asarray(self.beta_fit.guide_effect_mean),
                np.asarray(self.beta_fit.guide_effect_scale),
                guide_names,
                parent_elements,
            )
        effect_loc = np.asarray(mapping, dtype=np.float32) @ np.asarray(self.beta_fit.posterior_mean)
        effect_scale = np.clip(
            np.asarray(mapping, dtype=np.float32) @ np.asarray(self.beta_fit.posterior_scale),
            1e-6,
            None,
        )
        return effect_loc, effect_scale, guide_names, parent_elements

    def view_anndata_setup(self) -> dict[str, Any]:
        payload = self.setup.to_json_dict()
        print(payload)
        return payload

    def train(
        self,
        max_epochs: int | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
        accelerator: str = "cpu",
        device: str | None = None,
        *,
        steps: int | None = None,
        control_steps: int | None = None,
        beta_steps: int | None = None,
        control_epochs: int | None = None,
        beta_epochs: int | None = None,
        **_trainer_kwargs: Any,
    ) -> "PerTurboModel":
        """Fit the control and perturbation-effect stages.

        `max_epochs`, `control_epochs`, and `beta_epochs` are true dataset passes.
        Use `steps`, `control_steps`, or `beta_steps` to request raw SVI step counts.
        """
        device_spec = _infer_device(accelerator, device)
        if lr is not None:
            self.svi_config = core.SVIConfig(
                elbo=self.svi_config.elbo,
                num_particles=self.svi_config.num_particles,
                vectorize_particles=self.svi_config.vectorize_particles,
                step_size=lr,
            )

        control_keep_mask = _control_cell_keep_mask(self.adata, self.setup)
        controls = core.load_controls(
            self.adata,
            perturbation_key=None,
            control_selector=self.setup.control_substring,
            modality_key=self.setup.rna_modality,
            perturbation_modality_key=self.setup.perturbation_modality,
            perturbation_layer=self.setup.perturbation_layer,
            size_factor_key=self.setup.size_factor_key,
            library_size_key=self.setup.library_size_key,
            gene_name_key=self.setup.gene_name_key,
            device=device_spec,
            cell_keep_mask=control_keep_mask,
            continuous_covariates=self.setup.continuous_covariates_keys,
            batch_covariate=self.setup.batch_key,
            clip_gene_expression_percentile=self.clip_gene_expression_percentile,
            winsorize_gene_expression=self.winsorize_gene_expression,
            gene_outlier_threshold_floor=self.gene_outlier_threshold_floor,
            return_covariate_transform_state=True,
            retain_perturbation_design=self.guide_random_effects,
        )
        if isinstance(controls, tuple):
            control_data, covariate_transform_state = controls
        else:
            control_data = controls
            covariate_transform_state = None
        self.covariate_transform_state = covariate_transform_state
        self.library_size_center_log_mean = control_data.library_size_center_log_mean

        analysis_data = core.load_analysis_cells(
            self.adata,
            modality_key=self.setup.rna_modality,
            perturbation_modality_key=self.setup.perturbation_modality,
            perturbation_layer=self.setup.perturbation_layer,
            perturbation_element_varm_key=self.setup.guide_by_element_key,
            perturbation_element_names_uns_key=self.setup.guide_element_uns_key,
            size_factor_key=self.setup.size_factor_key,
            library_size_key=self.setup.library_size_key,
            gene_name_key=self.setup.gene_name_key,
            device=device_spec,
            continuous_covariates=self.setup.continuous_covariates_keys,
            batch_covariate=self.setup.batch_key,
            clip_gene_expression_percentile=self.clip_gene_expression_percentile,
            winsorize_gene_expression=self.winsorize_gene_expression,
            gene_outlier_threshold_floor=self.gene_outlier_threshold_floor,
            covariate_transform_state=covariate_transform_state,
            retain_guide_structure=bool(
                self.setup.guide_by_element_key is not None
                and (
                    self.guide_effect_strategy != "shared"
                    or self.guide_random_effects
                    or self.fit_perturbation_dispersion
                )
            ),
            library_size_center_log_mean=control_data.library_size_center_log_mean,
        )
        minibatch_size = batch_size if batch_size not in (None, 0) else None
        schedule = resolve_training_schedule(
            shared_steps=steps,
            shared_epochs=max_epochs,
            control_steps=control_steps,
            beta_steps=beta_steps,
            control_epochs=control_epochs,
            beta_epochs=beta_epochs,
            default_shared_epochs=100,
        )
        self.control_fit = core.fit_control(
            control_data,
            num_steps=schedule.resolve_stage_steps(
                stage="control",
                num_cells=int(control_data.counts.shape[0]),
                minibatch_size=minibatch_size,
            ),
            prior=self.effect_prior_dist,
            svi_config=self.svi_config,
            model_name=self.likelihood,
            num_factors=self.n_factors,
            use_observed_size_factors=control_data.size_factors is not None,
            count_censoring_percentile=self.count_censoring_percentile,
            minibatch_size=minibatch_size,
            guide_random_effects=self.guide_random_effects,
            fit_perturbation_dispersion=self.fit_perturbation_dispersion,
            perturbation_dispersion_prior_rate=self.perturbation_dispersion_prior_rate,
        )
        self.beta_fit = core.fit_perturbation_effects(
            analysis_data,
            self.control_fit,
            num_steps=schedule.resolve_stage_steps(
                stage="beta",
                num_cells=int(analysis_data.counts.shape[0]),
                minibatch_size=minibatch_size,
            ),
            prior=self.effect_prior_dist,
            svi_config=self.svi_config,
            model_name=self.likelihood,
            num_factors=self.n_factors,
            use_observed_size_factors=analysis_data.size_factors is not None,
            count_censoring_percentile=self.count_censoring_percentile,
            minibatch_size=minibatch_size,
            guide_effect_strategy=self.guide_effect_strategy,
            guide_activity_mode=self.guide_activity_mode,
            guide_random_effects=self.guide_random_effects,
        )
        self._guide_efficacy = _guide_efficacy_from_model(
            beta_fit=self.beta_fit,
            guide_effect_strategy=self.guide_effect_strategy,
            n_guides=_num_guides(self.adata, self.setup),
        )
        self.history = {
            "elbo_train": pd.DataFrame(
                {
                    "control_loss": np.asarray(self.control_fit.losses).reshape(-1),
                    "beta_loss": np.asarray(self.beta_fit.losses).reshape(-1),
                }
            )
        }
        return self

    def _require_fit(self) -> tuple[core.ControlFit, core.BetaFit]:
        if self.control_fit is None or self.beta_fit is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.control_fit, self.beta_fit

    def posterior_medians(self) -> PosteriorMedians:
        control_fit, beta_fit = self._require_fit()
        values = {
            "beta_0": np.asarray(control_fit.beta_0),
            "theta": np.asarray(control_fit.theta),
            "noise_scale": np.asarray(control_fit.noise_scale),
            "beta": np.asarray(beta_fit.posterior_mean),
            "element_effects": np.asarray(beta_fit.posterior_mean),
            "size_factor": np.asarray(control_fit.size_factors),
            "guide_efficacy": np.asarray(self.guide_efficacy),
        }
        guide_payload = self._guide_effect_payload()
        if guide_payload is not None:
            guide_loc, _guide_scale, _guide_names, _guide_parent_elements = guide_payload
            values["guide_effects"] = np.asarray(guide_loc)
        if beta_fit.guide_relative_efficiency_mean is not None:
            values["guide_relative_efficiency"] = np.asarray(beta_fit.guide_relative_efficiency_mean)
        if beta_fit.guide_offset_mean is not None:
            values["guide_offset"] = np.asarray(beta_fit.guide_offset_mean)
        if beta_fit.dispersion_excess_inverse is not None:
            values["perturbation_dispersion_excess_inverse"] = np.asarray(beta_fit.dispersion_excess_inverse)
        if beta_fit.guide_dispersion_excess_inverse is not None:
            values["guide_dispersion_excess_inverse"] = np.asarray(beta_fit.guide_dispersion_excess_inverse)
        if control_fit.covariate_coef is not None:
            values["covariate_coef"] = np.asarray(control_fit.covariate_coef)
        if control_fit.factor_loadings is not None:
            values["factor_loadings"] = np.asarray(control_fit.factor_loadings)
        if control_fit.factor_scores is not None:
            values["factor_scores"] = np.asarray(control_fit.factor_scores)
        if control_fit.pi_outlier is not None:
            values["pi_outlier"] = np.asarray(control_fit.pi_outlier)
        if control_fit.theta_outlier is not None:
            values["theta_outlier"] = np.asarray(control_fit.theta_outlier)
        if control_fit.outlier_mean_shift is not None:
            values["outlier_mean_shift"] = np.asarray(control_fit.outlier_mean_shift)
        return PosteriorMedians(values)

    @property
    def guide_efficacy(self) -> np.ndarray:
        if self._guide_efficacy is None:
            _control_fit, beta_fit = self._require_fit()
            self._guide_efficacy = _guide_efficacy_from_model(
                beta_fit=beta_fit,
                guide_effect_strategy=self.guide_effect_strategy,
                n_guides=_num_guides(self.adata, self.setup),
            )
        return np.asarray(self._guide_efficacy)

    def posterior_parameter_table(self):
        control_fit, beta_fit = self._require_fit()
        params = {
            "beta_0": control_fit.beta_0,
            "theta": control_fit.theta,
            "noise_scale": control_fit.noise_scale,
            "beta_loc": beta_fit.posterior_mean,
            "beta_scale": beta_fit.posterior_scale,
            "element_effects_loc": beta_fit.posterior_mean,
            "element_effects_scale": beta_fit.posterior_scale,
        }
        guide_payload = self._guide_effect_payload()
        if guide_payload is not None:
            guide_loc, guide_scale, _guide_names, _guide_parent_elements = guide_payload
            params["guide_effects_loc"] = guide_loc
            params["guide_effects_scale"] = guide_scale
        if beta_fit.guide_relative_efficiency_mean is not None:
            params["guide_relative_efficiency_loc"] = beta_fit.guide_relative_efficiency_mean
        if beta_fit.guide_relative_efficiency_scale is not None:
            params["guide_relative_efficiency_scale"] = beta_fit.guide_relative_efficiency_scale
        if beta_fit.guide_offset_mean is not None:
            params["guide_offset_loc"] = beta_fit.guide_offset_mean
        if beta_fit.guide_offset_scale is not None:
            params["guide_offset_scale"] = beta_fit.guide_offset_scale
        if beta_fit.dispersion_excess_inverse is not None:
            params["perturbation_dispersion_excess_inverse"] = beta_fit.dispersion_excess_inverse
        if beta_fit.guide_dispersion_excess_inverse is not None:
            params["guide_dispersion_excess_inverse"] = beta_fit.guide_dispersion_excess_inverse
        if control_fit.baseline_posterior is not None:
            params["baseline_beta_0_loc"] = control_fit.baseline_posterior.beta_0_loc
            params["baseline_beta_0_scale"] = control_fit.baseline_posterior.beta_0_scale
            params["baseline_theta_loc"] = control_fit.baseline_posterior.theta_log_loc
            params["baseline_theta_scale"] = control_fit.baseline_posterior.theta_log_scale
        return extract_parameter_table(params)

    def get_element_effects(self) -> pd.DataFrame:
        _control_fit, beta_fit = self._require_fit()
        return build_element_effects_df(
            effect_loc=np.asarray(beta_fit.posterior_mean),
            effect_scale=np.asarray(beta_fit.posterior_scale),
            element_names=self.element_names,
            gene_names=self.gene_names,
        )

    def get_guide_effects(self) -> pd.DataFrame:
        payload = self._guide_effect_payload()
        if payload is None:
            raise RuntimeError("Guide effects are unavailable because no guide-to-element mapping is registered.")
        guide_loc, guide_scale, guide_names, parent_elements = payload
        return build_guide_effects_df(
            effect_loc=guide_loc,
            effect_scale=guide_scale,
            guide_names=guide_names,
            guide_parent_elements=parent_elements,
            gene_names=self.gene_names,
        )

    def save(
        self,
        out_dir: str | Path,
        *,
        save_anndata: bool = True,
        overwrite: bool = False,
    ) -> Path:
        self._require_fit()
        out_path = Path(out_dir)
        if out_path.exists() and any(out_path.iterdir()) and not overwrite:
            raise FileExistsError(f"Bundle directory already exists: {out_path}")
        out_path.mkdir(parents=True, exist_ok=True)
        control_fit, beta_fit = self._require_fit()
        metadata = {
            "producer": "perturbo",
            "bundle_version": 2,
            "likelihood": self.likelihood,
            "effect_prior_dist": self.effect_prior_dist,
            "efficiency_mode": self.efficiency_mode,
            "guide_effect_strategy": self.guide_effect_strategy,
            "guide_activity_mode": self.guide_activity_mode,
            "n_factors": self.n_factors,
            "clip_gene_expression_percentile": self.clip_gene_expression_percentile,
            "winsorize_gene_expression": self.winsorize_gene_expression,
            "gene_outlier_threshold_floor": self.gene_outlier_threshold_floor,
            "count_censoring_percentile": self.count_censoring_percentile,
            "guide_random_effects": self.guide_random_effects,
            "fit_perturbation_dispersion": self.fit_perturbation_dispersion,
            "perturbation_dispersion_prior_rate": self.perturbation_dispersion_prior_rate,
            "fit_guide_efficacy": self.fit_guide_efficacy,
            "library_size_center_log_mean": self.library_size_center_log_mean,
            "setup": self.setup.to_json_dict(),
            "svi_config": asdict(self.svi_config),
            "covariate_transform_state": (
                asdict(self.covariate_transform_state) if self.covariate_transform_state is not None else None
            ),
            "save_anndata": save_anndata,
        }
        guide_posteriors = {k: v for k, v in _guide_posterior_arrays(beta_fit).items() if v is not None}
        return save_fit_bundle(
            out_path,
            metadata=metadata,
            mdata=self.adata,
            control_arrays=control_fit_arrays(control_fit),
            beta_arrays=_beta_arrays(beta_fit),
            element_effects=self.get_element_effects(),
            guide_posteriors=guide_posteriors or None,
            guide_efficacy=self.guide_efficacy,
        )


PERTURBO = PerTurboModel


__all__ = [
    "PerTurboModel",
    "PERTURBO",
    "PosteriorMedians",
]
