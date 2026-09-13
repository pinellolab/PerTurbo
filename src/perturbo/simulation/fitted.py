"""Simulation helpers backed by perturbo fitted models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import jax
import jax.numpy as jnp
import mudata as md
import numpy as np
import numpyro.distributions as dist
import pandas as pd

from perturbo import core
from perturbo.inference import PerTurboModel
from perturbo.io import _SAVED_SIZE_FACTOR_KEY, setup_mudata
from perturbo.log_normal_negative_binomial import LogNormalNegativeBinomial


def _resolve_cell_indices(model: PerTurboModel, n_cells: int, cell_indices: np.ndarray | None) -> np.ndarray:
    if cell_indices is not None:
        return np.asarray(cell_indices, dtype=np.int32)
    source_n = model.adata[model.setup.rna_modality].n_obs
    if n_cells == source_n:
        return np.arange(source_n, dtype=np.int32)
    rng = np.random.default_rng(0)
    return rng.choice(source_n, size=n_cells, replace=True).astype(np.int32)


def _resolve_gene_indices(model: PerTurboModel, gene_indices: np.ndarray | None) -> np.ndarray:
    if gene_indices is not None:
        return np.asarray(gene_indices, dtype=np.int32)
    return np.arange(model.adata[model.setup.rna_modality].n_vars, dtype=np.int32)


def _resolve_size_factors(model: PerTurboModel, cell_indices: np.ndarray, read_depth_adjust_factor: float) -> np.ndarray:
    obs = model.adata[model.setup.rna_modality].obs.iloc[cell_indices]
    if getattr(model.setup, "size_factor_mode", None) == "none":
        size_factor = np.zeros((cell_indices.shape[0], 1), dtype=np.float32)
    elif model.setup.size_factor_key is not None and model.setup.size_factor_key in obs.columns:
        size_factor_raw = pd.to_numeric(obs[model.setup.size_factor_key], errors="coerce").to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(size_factor_raw)):
            raise ValueError(
                f"size_factor_key '{model.setup.size_factor_key}' contains non-finite values in simulated source cells."
            )
        if model.setup.size_factor_key != _SAVED_SIZE_FACTOR_KEY and core._is_count_like(size_factor_raw):
            raise ValueError(
                f"size_factor_key '{model.setup.size_factor_key}' appears to contain integer counts. "
                "Use library_size_key for count data, or provide real-valued size factors centered around zero."
            )
        size_factor = size_factor_raw.reshape(-1, 1)
    elif model.setup.library_size_key is not None and model.setup.library_size_key in obs.columns:
        library_size = np.asarray(obs[model.setup.library_size_key], dtype=np.float32)
        log_lib = np.log1p(library_size.astype(np.float64, copy=False))
        center = getattr(model, "library_size_center_log_mean", None)
        if center is None:
            center = float(np.mean(log_lib)) if log_lib.size > 0 else 0.0
        size_factor = (log_lib - float(center)).astype(np.float32, copy=False)[:, None]
    else:
        size_factor = np.zeros((cell_indices.shape[0], 1), dtype=np.float32)
    if read_depth_adjust_factor != 1.0:
        size_factor = size_factor + np.log(float(read_depth_adjust_factor))
    return size_factor


def _resolve_covariates(model: PerTurboModel, cell_indices: np.ndarray) -> np.ndarray | None:
    control_fit = model.control_fit
    if control_fit is None or control_fit.covariate_coef is None:
        return None
    transform_state = getattr(model, "covariate_transform_state", None)
    if transform_state is None:
        return None
    obs = model.adata[model.setup.rna_modality].obs.iloc[cell_indices]
    covariates, _ = core.apply_covariate_transform(obs, transform_state)
    return np.asarray(covariates, dtype=np.float32)


_SAMPLE_CHUNK_ROWS = 4096


def _sample_counts(
    *,
    model: PerTurboModel,
    mu: np.ndarray,
    outlier_mu: np.ndarray | None = None,
    gene_indices: np.ndarray,
    theta_override: np.ndarray | None = None,
    seed: int = 0,
    chunk_rows: int = _SAMPLE_CHUNK_ROWS,
) -> np.ndarray:
    """Sample counts for ``mu`` (cells, genes), in row chunks so any dataset size fits on the device.

    A single draw over 100k cells x 2k genes exhausted a 40 GB GPU inside the
    gamma sampler; chunks of ``chunk_rows`` cells each use their own fold of the
    seed. Datasets up to one chunk are drawn exactly as before.
    """
    if model.control_fit is None:
        raise RuntimeError("Model must be trained before simulation.")
    n_cells = int(np.asarray(mu).shape[0])
    if n_cells > chunk_rows:
        parts = []
        for index, start in enumerate(range(0, n_cells, chunk_rows)):
            stop = min(start + chunk_rows, n_cells)
            override = None if theta_override is None else np.asarray(theta_override)[start:stop]
            outlier_mu_part = None if outlier_mu is None else np.asarray(outlier_mu)[start:stop]
            parts.append(
                _sample_counts(
                    model=model, mu=np.asarray(mu)[start:stop], outlier_mu=outlier_mu_part,
                    gene_indices=gene_indices,
                    theta_override=override,
                    seed=int(jax.random.key_data(jax.random.fold_in(jax.random.PRNGKey(seed), index))[-1]),
                    chunk_rows=chunk_rows,
                )
            )
        return np.concatenate(parts, axis=0)
    control_fit = model.control_fit
    theta = np.asarray(control_fit.theta)[gene_indices] if theta_override is None else np.asarray(theta_override)
    if theta.shape != mu.shape:
        theta = np.broadcast_to(theta, mu.shape)
    if not np.all(np.isfinite(theta)) or np.any(theta <= 0.0):
        raise ValueError("Simulation theta values must be finite and positive.")
    key = jax.random.PRNGKey(seed)
    logits = jnp.asarray(mu - np.log(theta), dtype=jnp.float32)
    total_count = jnp.asarray(theta, dtype=jnp.float32)

    if model.likelihood.lower() in {"negbin", "nb", "censored_nb"}:
        dist_obj = dist.NegativeBinomialLogits(total_count=total_count, logits=logits)
    elif model.likelihood.lower() in {"lnnb", "lognormal_nb"}:
        noise_scale = np.asarray(control_fit.noise_scale)[gene_indices]
        dist_obj = LogNormalNegativeBinomial(
            total_count=total_count,
            logits=logits,
            multiplicative_noise_scale=jnp.asarray(noise_scale, dtype=jnp.float32),
        )
    elif model.likelihood.lower() == "mixture_nb":
        if control_fit.pi_outlier is None or control_fit.theta_outlier is None:
            raise ValueError("mixture_nb simulation requires outlier parameters from the control fit.")
        theta_outlier = np.asarray(control_fit.theta_outlier)[gene_indices]
        pi_outlier = np.asarray(control_fit.pi_outlier)[gene_indices]
        outlier_shift = (
            np.asarray(control_fit.outlier_mean_shift)[gene_indices]
            if control_fit.outlier_mean_shift is not None
            else np.zeros_like(theta_outlier)
        )
        outlier_predictor = mu if outlier_mu is None else np.asarray(outlier_mu)
        logits_outlier = jnp.asarray(
            outlier_predictor + outlier_shift[None, :] - np.log(theta_outlier)[None, :],
            dtype=jnp.float32,
        )
        component_distribution = dist.NegativeBinomialLogits(
            logits=jnp.stack([logits, logits_outlier], axis=-1),
            total_count=jnp.stack(
                [
                    jnp.broadcast_to(total_count, logits.shape),
                    jnp.broadcast_to(jnp.asarray(theta_outlier, dtype=jnp.float32), logits.shape),
                ],
                axis=-1,
            ),
        )
        mixing_distribution = dist.CategoricalProbs(
            probs=jnp.stack(
                [
                    1.0 - jnp.asarray(pi_outlier, dtype=jnp.float32),
                    jnp.asarray(pi_outlier, dtype=jnp.float32),
                ],
                axis=-1,
            )
        )
        dist_obj = dist.MixtureSameFamily(mixing_distribution, component_distribution)
    else:
        raise ValueError(f"Unsupported likelihood for simulation: {model.likelihood}")

    sampled = dist_obj.sample(key)
    if model.likelihood.lower() == "censored_nb":
        if control_fit.count_censoring_threshold is None:
            raise ValueError("censored_nb simulation requires count_censoring_threshold from the control fit.")
        threshold = jnp.asarray(np.asarray(control_fit.count_censoring_threshold)[gene_indices], dtype=sampled.dtype)
        sampled = jnp.minimum(sampled, threshold[None, :])
    return np.asarray(sampled, dtype=np.int32)


def _resolve_perturbation_dispersion_theta(
    *,
    model: PerTurboModel,
    element_membership: np.ndarray,
    guide_obs: np.ndarray,
    gene_indices: np.ndarray,
) -> np.ndarray | None:
    """Return cell-gene NB total counts after fitted excess-dispersion shifts."""
    if model.beta_fit is None or model.control_fit is None:
        raise RuntimeError("Model must be trained before simulation.")
    guide_excess = model.beta_fit.guide_dispersion_excess_inverse
    excess = model.beta_fit.dispersion_excess_inverse
    if guide_excess is not None:
        guide_values = np.asarray(guide_obs, dtype=np.float32)
        excess_arr = np.asarray(guide_excess, dtype=np.float32)[:, gene_indices]
        if guide_values.ndim != 2 or guide_values.shape[1] != excess_arr.shape[0]:
            raise ValueError("guide_obs must have one column per fitted guide dispersion term.")
        if np.any(~np.isfinite(guide_values)) or np.any(guide_values < 0):
            raise ValueError("guide_obs must be finite and non-negative for fitted guide dispersion.")
        total_excess = guide_values @ excess_arr
    elif excess is not None:
        membership = np.asarray(element_membership, dtype=np.float32)
        excess_arr = np.asarray(excess, dtype=np.float32)[:, gene_indices]
        if membership.ndim != 2 or membership.shape[1] != excess_arr.shape[0]:
            raise ValueError("element membership must have one column per fitted perturbation.")
        if (
            np.any(membership < -1e-6)
            or not np.allclose(membership, np.round(membership), atol=1e-6)
            or np.any(membership.sum(axis=1) > 1.0 + 1e-6)
        ):
            raise ValueError("Element-level fitted dispersion requires low-MOI one-hot element membership per cell.")
        total_excess = membership @ excess_arr
    else:
        return None
    base_inverse = np.reciprocal(np.asarray(model.control_fit.theta, dtype=np.float32)[gene_indices])
    total_inverse = base_inverse[None, :] + total_excess
    return np.reciprocal(total_inverse)


def _sample_guide_random_effect_contribution(
    *,
    model: PerTurboModel,
    guide_obs: np.ndarray,
    gene_indices: np.ndarray,
    seed: int = 1,
) -> np.ndarray | None:
    if not bool(getattr(model, "guide_random_effects", False)):
        return None
    if model.control_fit is None:
        raise RuntimeError("Model must be trained before simulation.")
    tau = model.control_fit.guide_random_effect_tau
    if tau is None:
        raise ValueError("guide-random-effects simulation requires guide_random_effect_tau from the control fit.")
    tau_arr = np.asarray(tau, dtype=np.float32).reshape(-1)[gene_indices]
    if not np.all(np.isfinite(tau_arr)) or np.any(tau_arr < 0.0):
        raise ValueError("guide_random_effect_tau must be finite and non-negative for simulation.")
    key = jax.random.PRNGKey(seed)
    effects = jax.random.normal(
        key,
        shape=(int(guide_obs.shape[1]), int(tau_arr.shape[0])),
        dtype=jnp.float32,
    ) * jnp.asarray(tau_arr, dtype=jnp.float32)[None, :]
    return np.asarray(guide_obs, dtype=np.float32) @ np.asarray(effects, dtype=np.float32)


def simulate_data_from_trained_model(
    model: PerTurboModel,
    guide_obs: np.ndarray,
    guide_by_element: np.ndarray,
    element_by_gene_lfc: np.ndarray,
    guide_efficacy: np.ndarray,
    read_depth_adjust_factor: float = 1.0,
    module_kwargs: dict[str, Any] | None = None,
    module_init_kwargs: dict[str, Any] | None = None,
    gene_indices: np.ndarray | None = None,
    cell_indices: np.ndarray | None = None,
    param_values: dict[str, Any] | None = None,
    accelerator: str = "auto",
    device: int | str = "auto",
    seed: int = 0,
) -> md.MuData:
    """Simulate a MuData object from a trained perturbo model.

    This mirrors the public PerTurbo simulator signature, but it is stateless and
    bundle-oriented rather than PyTorch-module-oriented. ``seed`` drives the
    count sampler (and the guide random effects, from ``seed + 1``); callers that
    simulate several datasets from one model should vary it, otherwise every
    dataset shares one set of underlying draws.

    Parameters
    ----------
    model
        Trained or loaded model with both control and perturbation fits.
    guide_obs
        Cell-by-guide observation matrix.
    guide_by_element
        Guide-by-element mapping matrix.
    element_by_gene_lfc
        Element-by-model-gene log fold changes.
    guide_efficacy
        One weight per guide.
    read_depth_adjust_factor
        Multiplicative adjustment applied to source-cell library sizes.
    gene_indices, cell_indices
        Optional indices selecting model genes and source cells.
    seed
        Seed for count sampling; guide random effects use ``seed + 1``.

    Returns
    -------
    mudata.MuData
        RNA counts and supplied guide observations, configured with the trained
        model's modality and metadata keys.
    """
    del module_kwargs, module_init_kwargs, param_values, accelerator, device
    if model.beta_fit is None or model.control_fit is None:
        raise RuntimeError("Model must be trained or loaded before simulation.")

    guide_obs_arr = np.asarray(guide_obs, dtype=np.float32)
    guide_by_element_arr = np.asarray(guide_by_element, dtype=np.float32)
    guide_eff_arr = np.asarray(guide_efficacy, dtype=np.float32).reshape(-1)
    if guide_obs_arr.shape[1] != guide_by_element_arr.shape[0]:
        raise ValueError("guide_obs columns must match guide_by_element rows.")
    if guide_eff_arr.shape[0] != guide_obs_arr.shape[1]:
        raise ValueError("guide_efficacy length must match guide_obs columns.")

    gene_idx = _resolve_gene_indices(model, gene_indices)
    cell_idx = _resolve_cell_indices(model, guide_obs_arr.shape[0], cell_indices)
    lfc = np.asarray(element_by_gene_lfc, dtype=np.float32)[:, gene_idx]
    element_membership = guide_obs_arr @ guide_by_element_arr
    weighted_guides = guide_obs_arr * guide_eff_arr[None, :]
    # The fitted shared-effect design records whether an element is present,
    # so two observed guides for the same element still contribute one beta.
    # Unit efficacy is the saved/default shared strategy. A caller-provided
    # non-unit vector remains an explicit guide weighting override.
    shared_unit_efficacy = (
        str(getattr(model, "guide_effect_strategy", "shared")).lower() == "shared"
        and np.allclose(guide_eff_arr, 1.0, rtol=0.0, atol=1e-6)
    )
    if shared_unit_efficacy:
        element_scores = np.asarray(element_membership > 0, dtype=np.float32)
    else:
        element_scores = weighted_guides @ guide_by_element_arr

    beta_0 = np.asarray(model.control_fit.beta_0, dtype=np.float32)[gene_idx]
    mu = beta_0[None, :] + _resolve_size_factors(model, cell_idx, read_depth_adjust_factor)
    covariates = _resolve_covariates(model, cell_idx)
    if covariates is not None and model.control_fit.covariate_coef is not None:
        cov_coef = np.asarray(model.control_fit.covariate_coef, dtype=np.float32)[:, gene_idx]
        mu = mu + covariates @ cov_coef
    outlier_mu = np.asarray(mu).copy()
    mu = mu + element_scores @ lfc
    guide_random_effect_contrib = _sample_guide_random_effect_contribution(
        model=model,
        guide_obs=guide_obs_arr,
        gene_indices=gene_idx,
        seed=int(seed) + 1,
    )
    if guide_random_effect_contrib is not None:
        mu = mu + guide_random_effect_contrib
        outlier_mu = outlier_mu + guide_random_effect_contrib

    theta_override = _resolve_perturbation_dispersion_theta(
        model=model,
        element_membership=element_membership,
        guide_obs=guide_obs_arr,
        gene_indices=gene_idx,
    )
    counts = _sample_counts(
        model=model,
        mu=mu,
        outlier_mu=outlier_mu,
        gene_indices=gene_idx,
        theta_override=theta_override,
        seed=int(seed),
    )

    rna_source = model.adata[model.setup.rna_modality]
    pert_source = model.adata[model.setup.perturbation_modality]
    obs = rna_source.obs.iloc[cell_idx].copy()
    var = rna_source.var.iloc[gene_idx].copy()
    rna = ad.AnnData(X=counts, obs=obs, var=var)
    rna.varm["lfc"] = np.asarray(element_by_gene_lfc, dtype=np.float32)[:, gene_idx].T
    if model.setup.gene_by_element_key is not None:
        rna.varm[model.setup.gene_by_element_key] = (
            np.asarray(element_by_gene_lfc, dtype=np.float32)[:, gene_idx] != 0
        ).T.astype(np.float32, copy=False)

    pert_var_names = list(pert_source.var_names.astype(str))
    if len(pert_var_names) != guide_obs_arr.shape[1]:
        pert_var_names = [f"guide_{i}" for i in range(guide_obs_arr.shape[1])]
    guide_var = pd.DataFrame(index=pert_var_names)
    pert = ad.AnnData(X=guide_obs_arr.astype(np.float32), obs=obs.copy(), var=guide_var)
    if model.setup.guide_by_element_key is not None:
        pert.varm[model.setup.guide_by_element_key] = guide_by_element_arr
    else:
        pert.varm["element_targeted"] = guide_by_element_arr
    if model.setup.guide_element_uns_key is not None:
        element_names = model.element_names
        if len(element_names) == guide_by_element_arr.shape[1]:
            pert.uns[model.setup.guide_element_uns_key] = np.asarray(element_names, dtype=object)
    pert.uns["guide_efficacy"] = guide_eff_arr

    mdata = md.MuData(
        {
            model.setup.rna_modality: rna,
            model.setup.perturbation_modality: pert,
        }
    )
    setup_mudata(
        mdata,
        batch_key=model.setup.batch_key,
        library_size_key=model.setup.library_size_key,
        size_factor_key=model.setup.size_factor_key,
        continuous_covariates_keys=model.setup.continuous_covariates_keys,
        gene_by_element_key=model.setup.gene_by_element_key,
        guide_by_element_key=model.setup.guide_by_element_key,
        rna_element_uns_key=model.setup.rna_element_uns_key,
        guide_element_uns_key=model.setup.guide_element_uns_key,
        gene_name_key=model.setup.gene_name_key,
        control_substring=model.setup.control_substring,
        modalities={
            "rna_layer": model.setup.rna_modality,
            "perturbation_layer": model.setup.perturbation_modality,
        },
        perturbation_layer=model.setup.perturbation_layer,
        size_factor_mode=model.setup.size_factor_mode,
        size_factor_provenance=model.setup.size_factor_provenance,
    )
    return mdata


def save_simulated_mudata(mdata: md.MuData, path: str | Path) -> Path:
    """Write simulated MuData to H5MU, creating parent directories.

    Returns the normalized destination :class:`pathlib.Path`.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mdata.write_h5mu(out_path)
    return out_path


__all__ = [
    "save_simulated_mudata",
    "simulate_data_from_trained_model",
]
