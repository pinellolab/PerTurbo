"""The common count generator: conditioned execution of the inference model.

Design/fit adapters supply named latent sites; this module never reconstructs
the linear predictor or implements an observation distribution. Chunked draws
retain one set of global latents and return host arrays to bound device memory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import Predictive

from perturbo.model import BaseModel, GuideSharedEffectModel

GENERATOR_VERSION = "perturbo_numpyro_predictive_v1"


def simulator_provenance() -> dict[str, Any]:
    from perturbo import model
    from perturbo import log_normal_negative_binomial

    paths = [
        Path(__file__),
        Path(__file__).with_name("fitted.py"),
        Path(model.__file__),
        Path(log_normal_negative_binomial.__file__),
    ]
    return {
        "generator": GENERATOR_VERSION,
        "model_module": "perturbo.model",
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "numpyro_version": numpyro.__version__,
    }


def validate_simulation_output_root(root: str | Path) -> None:
    """Prevent a resumed grid from relabeling old counts as new-model draws."""
    root = Path(root)
    design = root / "design.json"
    if not design.exists():
        if (root / "truth.parquet").exists() or any(root.glob("*/sim.h5*")):
            raise ValueError(f"{root}: existing simulation files lack a generator receipt; choose a new output root.")
        return
    recorded = json.loads(design.read_text()).get("simulation_generator", {})
    current = simulator_provenance()
    if any(recorded.get(k) != current[k] for k in ("generator", "source_sha256")):
        raise ValueError(f"{root}: existing simulation generator differs; preserve it and choose a new output root.")


def sample_predictive_counts(
    conditioned_sites: dict[str, Any],
    pert_id,
    rng_key,
    *,
    size_factors=None,
    covariates=None,
    guide_matrix=None,
    guide_to_element=None,
    likelihood: str = "nb",
    num_factors: int | None = None,
    guide_effect_strategy: str = "shared",
    guide_random_effects: bool = False,
    fit_perturbation_dispersion: bool = False,
    chunk_rows: int | None = 4096,
):
    """Draw once from BaseModel/GuideSharedEffectModel with fixed fitted sites.

    ``factor_scores`` (F,C,1) and ``size_factor`` (C,1) are the only supported
    cell-local conditioned sites. Unconditioned local sites are drawn by the
    model; global sites are sampled once, including when cells are chunked.
    Changing chunk size changes random draws, not the generating parameters.
    Censored-NB fitting masks outliers: its predictive distribution is raw NB,
    exactly as defined in the model, with no post-generation winsorization.
    """
    sites = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in conditioned_sites.items()}
    if not {"beta_0", "theta", "beta"} <= sites.keys():
        raise ValueError("Simulation requires fixed beta_0, theta and beta.")
    allowed = {"beta_0", "theta", "beta", "size_factor"}
    if likelihood in {"lnnb", "lognormal_nb"}:
        allowed.add("noise_scale")
    if likelihood == "mixture_nb":
        allowed.update(("pi_outlier", "theta_outlier", "outlier_mean_shift"))
    if num_factors is not None:
        allowed.update(("factor_scale", "factor_loadings", "factor_scores"))
    if covariates is not None:
        allowed.add("covariate_coef")
    if guide_random_effects:
        allowed.update(
            (
                "guide_random_effect_log_tau_loc",
                "guide_random_effect_log_tau_scale",
                "guide_random_effect_tau",
                "guide_random_effect",
            )
        )
    if fit_perturbation_dispersion:
        allowed.add("guide_dispersion_excess_inverse" if guide_to_element is not None else "dispersion_excess_inverse")
    if guide_to_element is not None and guide_effect_strategy in {"relative", "offset"}:
        allowed.add("guide_relative_efficiency" if guide_effect_strategy == "relative" else "guide_offset")
    if sites.keys() - allowed:
        raise ValueError(f"Conditioned sites unused by this model configuration: {sorted(sites.keys() - allowed)}")
    for name, value in sites.items():
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"Non-finite simulation site: {name}")
    if sites["beta"].ndim != 2 or getattr(pert_id, "ndim", None) not in {1, 2}:
        raise ValueError("beta must be (elements, genes); pert_id must be indices or a design matrix.")
    n_cells, n_genes = int(pert_id.shape[0]), int(sites["beta"].shape[1])
    if n_cells < 1 or n_genes < 1 or sites["beta"].shape[0] < 1:
        raise ValueError("Simulation needs at least one cell, gene and element.")
    design = np.asarray(pert_id)
    if design.ndim == 1:
        if design.dtype.kind not in "iu" or np.any(design < 0) or np.any(design >= sites["beta"].shape[0]):
            raise ValueError("pert_id indices must be integers on the beta element axis.")
    elif design.shape[1] != sites["beta"].shape[0] or not np.isfinite(design).all():
        raise ValueError("pert_id design must be finite and follow the beta element axis.")
    if chunk_rows is not None and (not isinstance(chunk_rows, int) or chunk_rows < 1):
        raise ValueError("chunk_rows must be a positive integer or None.")
    if sites["beta_0"].shape != (n_genes,) or sites["theta"].shape != (n_genes,):
        raise ValueError("beta_0 and theta must follow the beta gene axis.")
    if not np.isfinite(np.asarray(sites["theta"])).all() or np.any(np.asarray(sites["theta"]) <= 0):
        raise ValueError("Simulation theta values must be finite and positive.")
    if size_factors is not None:
        if "size_factor" in sites and not np.array_equal(
            sites["size_factor"], np.asarray(size_factors, dtype=np.float32)
        ):
            raise ValueError("Conflicting size_factors and conditioned size_factor.")
        sites["size_factor"] = jnp.asarray(size_factors, dtype=jnp.float32)
    if "size_factor" in sites and sites["size_factor"].shape != (n_cells, 1):
        raise ValueError("size_factor must have shape (num_cells, 1).")
    if "size_factor" in sites and not np.isfinite(np.asarray(sites["size_factor"])).all():
        raise ValueError("size_factor must be finite.")
    if num_factors is not None:
        if num_factors < 1 or "factor_loadings" not in sites:
            raise ValueError("Positive num_factors requires fitted factor_loadings.")
        if sites["factor_loadings"].shape != (num_factors, 1, n_genes):
            raise ValueError("factor_loadings must have shape (num_factors, 1, num_genes).")
        if "factor_scores" in sites and sites["factor_scores"].shape != (num_factors, n_cells, 1):
            raise ValueError(
                "factor_scores must match simulated cell count and have shape (num_factors, num_cells, 1)."
            )
    elif "factor_scores" in sites or "factor_loadings" in sites:
        raise ValueError("Factor sites require num_factors.")
    if covariates is not None:
        if covariates.ndim != 2 or covariates.shape[0] != n_cells or "covariate_coef" not in sites:
            raise ValueError("covariates require matching cells and fitted covariate_coef.")
        if sites["covariate_coef"].shape != (covariates.shape[1], n_genes):
            raise ValueError("covariate_coef must match the covariate and gene axes.")
    elif "covariate_coef" in sites:
        raise ValueError("covariate_coef supplied without covariates.")
    for name, value in (("covariates", covariates), ("guide_matrix", guide_matrix)):
        if value is not None and (
            value.ndim != 2 or value.shape[0] != n_cells or not np.isfinite(np.asarray(value)).all()
        ):
            raise ValueError(f"{name} must be finite, two-dimensional and follow the cell axis.")
    guide_aware = guide_to_element is not None
    model_fn = GuideSharedEffectModel if guide_aware else BaseModel
    if guide_aware and (
        guide_matrix is None or guide_to_element.shape != (guide_matrix.shape[1], sites["beta"].shape[0])
    ):
        raise ValueError("guide_to_element must map every guide to the beta element axis.")

    def arguments(start, stop):
        return dict(
            size_factors=None if size_factors is None else jnp.asarray(size_factors[start:stop], dtype=jnp.float32),
            covariates=None if covariates is None else jnp.asarray(covariates[start:stop], dtype=jnp.float32),
            guide_matrix=None if guide_matrix is None else jnp.asarray(guide_matrix[start:stop], dtype=jnp.float32),
            guide_to_element=None if guide_to_element is None else jnp.asarray(guide_to_element, dtype=jnp.float32),
            num_cells=stop - start,
            num_genes=n_genes,
            num_perts=int(sites["beta"].shape[0]),
            num_factors=num_factors,
            likelihood=likelihood,
            guide_effect_strategy=guide_effect_strategy,
            guide_random_effects=guide_random_effects,
            fit_perturbation_dispersion=fit_perturbation_dispersion,
        )

    def sliced_sites(start, stop):
        return {
            k: (v[:, start:stop, :] if k == "factor_scores" else v[start:stop] if k == "size_factor" else v)
            for k, v in sites.items()
        }

    rows = n_cells if chunk_rows is None else min(chunk_rows, n_cells)
    # Discover global latent sites through the model, rather than reproducing
    # priors here. Freeze them once so chunk boundaries cannot change a guide
    # effect, loading, or hyperparameter within the same simulated dataset.
    if rows < n_cells:
        global_key, local_key = jax.random.split(rng_key)
        trace = numpyro.handlers.trace(
            numpyro.handlers.seed(numpyro.handlers.condition(model_fn, data=sliced_sites(0, rows)), global_key)
        ).get_trace(None, jnp.asarray(pert_id[:rows]), skip_obs_sampling=True, **arguments(0, rows))
        for name, site in trace.items():
            if site["type"] == "sample" and not any(p.name == "cells" for p in site["cond_indep_stack"]):
                sites.setdefault(name, site["value"])
    keys = [rng_key] if rows == n_cells else jax.random.split(local_key, (n_cells + rows - 1) // rows)
    counts, depths = [], []
    for key, start in zip(keys, range(0, n_cells, rows)):
        stop = min(start + rows, n_cells)
        predictive = Predictive(
            numpyro.handlers.condition(model_fn, data=sliced_sites(start, stop)),
            num_samples=1,
            return_sites=["obs", "size_factor"],
        )
        draws = predictive(key, None, jnp.asarray(pert_id[start:stop]), **arguments(start, stop))
        counts.append(np.asarray(draws["obs"][0], dtype=np.int32))
        depths.append(np.asarray(draws["size_factor"][0], dtype=np.float32))
    return np.concatenate(counts), np.concatenate(depths)
