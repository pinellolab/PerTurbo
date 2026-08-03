from collections import namedtuple
from functools import partial

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from perturbo.log_normal_negative_binomial import LogNormalNegativeBinomial
from perturbo.plate import plate


def create_plates(
    counts: jnp.ndarray | None,
    pert_id: jnp.ndarray | None,
    covariates: jnp.ndarray | None = None,
    guide_matrix: jnp.ndarray | None = None,
    num_cells: int | None = None,
    num_genes: int | None = None,
    num_perts: int | None = None,
    num_guides: int | None = None,
    num_factors: int | None = None,
    num_covariates: int | None = None,
    subsample_size: int | None = None,
    cell_idx: jnp.ndarray | None = None,
    **kwargs,
):
    del kwargs
    Plates = namedtuple("Plates", ["cells", "genes", "perts", "guides", "factors", "covariates"])
    if counts is not None:
        inferred_cells, inferred_genes = counts.shape
        if num_cells is None:
            num_cells = int(inferred_cells)
        if num_genes is None:
            num_genes = int(inferred_genes)
    if num_perts is None and pert_id is not None and getattr(pert_id, "ndim", None) == 2:
        num_perts = int(pert_id.shape[1])
    if num_guides is None and guide_matrix is not None and getattr(guide_matrix, "ndim", None) == 2:
        num_guides = int(guide_matrix.shape[1])
    if num_factors is None:
        num_factors = 1
    if num_covariates is None and covariates is not None and getattr(covariates, "ndim", None) == 2:
        num_covariates = int(covariates.shape[1])
    covariate_plate_size = int(num_covariates) if num_covariates is not None and int(num_covariates) > 0 else 1
    covariate_plate = plate("covariates", covariate_plate_size, dim=-2)
    return Plates(
        plate("cells", num_cells, dim=-2, subsample_size=subsample_size, subsample=cell_idx),
        plate("genes", num_genes, dim=-1),
        plate("perts", num_perts, dim=-2),
        plate("guides", int(num_guides) if num_guides is not None and int(num_guides) > 0 else 1, dim=-2),
        plate("factors", num_factors, dim=-3),
        covariate_plate,
    )


def _sample_effect_site(name: str, prior: str):
    if prior == "normal":
        return numpyro.sample(name, dist.Normal(0, 1.0))
    if prior == "cauchy":
        return numpyro.sample(name, dist.Cauchy(0, 0.1))
    raise ValueError(f"Unknown prior: {prior}")


def _resolve_count_censoring_threshold(counts, count_censoring_threshold):
    del counts
    if count_censoring_threshold is None:
        raise ValueError(
            "count_censoring_threshold must be provided for censored_nb; "
            "compute once from control cells and reuse across training."
        )
    return jnp.asarray(count_censoring_threshold)


def _subsample_cell_axis(
    values,
    sampled_cell_idx: jnp.ndarray,
    *,
    num_cells: int | None,
    name: str,
):
    if values is None:
        return None
    if getattr(values, "ndim", None) == 0:
        raise ValueError(f"{name} must have a leading cell axis.")
    batch_size = int(sampled_cell_idx.shape[0])
    if values.shape[0] == batch_size:
        return values
    if num_cells is not None and values.shape[0] == num_cells:
        return values[sampled_cell_idx, ...]
    raise ValueError(
        f"{name} has incompatible shape {values.shape}; expected leading axis {batch_size} "
        f"(already subsampled) or {num_cells} (full dataset)."
    )


def _sample_observations(
    *,
    counts,
    likelihood: str,
    logits,
    theta,
    noise_scale=None,
    logits_outlier=None,
    theta_outlier=None,
    pi_outlier=None,
    count_censoring_threshold=None,
):
    uses_mixture_nb = likelihood in {"mixture_nb"}
    uses_censored_nb = likelihood in {"censored_nb", "censored_negbin"}
    if likelihood in {"nb", "negbin"}:
        return numpyro.sample(
            "obs",
            dist.NegativeBinomialLogits(logits=logits, total_count=theta),
            obs=counts,
        )
    if uses_censored_nb:
        if counts is None:
            return numpyro.sample(
                "obs",
                dist.NegativeBinomialLogits(logits=logits, total_count=theta),
            )
        count_censoring_threshold = _resolve_count_censoring_threshold(counts, count_censoring_threshold)
        uncensored_mask = jnp.asarray(counts) <= count_censoring_threshold
        return numpyro.sample(
            "obs",
            dist.NegativeBinomialLogits(logits=logits, total_count=theta).mask(uncensored_mask),
            obs=counts,
        )
    if likelihood in {"lnnb", "lognormal_nb"}:
        return numpyro.sample(
            "obs",
            LogNormalNegativeBinomial(
                total_count=theta,
                logits=logits,
                multiplicative_noise_scale=noise_scale,
            ),
            obs=counts,
        )
    if uses_mixture_nb:
        logits_components = jnp.stack([logits, logits_outlier], axis=-1)
        total_count_components = jnp.stack(
            [
                jnp.broadcast_to(theta, logits.shape),
                jnp.broadcast_to(theta_outlier, logits_outlier.shape),
            ],
            axis=-1,
        )
        component_distribution = dist.NegativeBinomialLogits(
            logits=logits_components,
            total_count=total_count_components,
        )
        mixing_distribution = dist.CategoricalProbs(
            probs=jnp.stack([1.0 - pi_outlier, pi_outlier], axis=-1)
        )
        return numpyro.sample(
            "obs",
            dist.MixtureSameFamily(mixing_distribution, component_distribution),
            obs=counts,
        )
    raise ValueError(
        "likelihood must be one of: 'nb', 'negbin', 'censored_nb', 'censored_negbin', "
        "'lnnb', 'lognormal_nb', 'mixture_nb'."
    )


def BaseModel(
    counts,
    pert_id,
    size_factors=None,
    covariates=None,
    guide_matrix=None,
    guide_to_element=None,
    num_cells=None,
    num_genes=None,
    num_perts=None,
    num_guides=None,
    num_factors=None,
    prior="normal",
    subsample_size: int | None = None,
    likelihood: str = "nb",
    cell_idx: jnp.ndarray | None = None,
    skip_obs_sampling: bool = False,
    guide_effect_strategy: str = "shared",
    guide_random_effects: bool = False,
    fit_perturbation_dispersion: bool = False,
    perturbation_dispersion_prior_rate: float = 10.0,
    count_censoring_threshold=None,
):
    del guide_to_element, guide_effect_strategy
    plates = create_plates(
        counts,
        pert_id,
        covariates=covariates,
        guide_matrix=guide_matrix,
        num_cells=num_cells,
        num_genes=num_genes,
        num_perts=num_perts,
        num_guides=num_guides,
        num_factors=num_factors,
        subsample_size=subsample_size,
        cell_idx=cell_idx,
    )
    cell_plate = plates.cells
    gene_plate = plates.genes
    pert_plate = plates.perts
    guide_plate = plates.guides
    factor_plate = plates.factors
    covariate_plate = plates.covariates
    full_num_cells = int(num_cells) if num_cells is not None else (int(counts.shape[0]) if counts is not None else None)
    uses_mixture_nb = likelihood in {"mixture_nb"}

    if num_factors is not None:
        factor_scale = numpyro.sample("factor_scale", dist.HalfNormal(0.1))
        with factor_plate:
            with gene_plate:
                factor_loadings = numpyro.sample("factor_loadings", dist.Normal(0, factor_scale))
    guide_random_effect_log_tau_loc = None
    guide_random_effect_log_tau_scale = None
    if guide_random_effects and guide_matrix is not None:
        # Global (cross-gene) hierarchical hyperparameters.
        guide_random_effect_log_tau_loc = numpyro.sample(
            "guide_random_effect_log_tau_loc",
            dist.Normal(0.0, 1.5),
        )
        guide_random_effect_log_tau_scale = numpyro.sample(
            "guide_random_effect_log_tau_scale",
            dist.HalfNormal(1.0),
        )

    with gene_plate:
        if uses_mixture_nb:
            pi_outlier = numpyro.sample("pi_outlier", dist.Beta(1.0, 10.0))
        beta_0 = numpyro.sample("beta_0", dist.Normal(0, 3))
        theta = numpyro.sample("theta", dist.LogNormal(0.0, 2.0))
        guide_random_effect = None
        if guide_random_effects and guide_matrix is not None:
            guide_random_effect_tau = numpyro.sample(
                "guide_random_effect_tau",
                dist.LogNormal(guide_random_effect_log_tau_loc, guide_random_effect_log_tau_scale),
            )
            with guide_plate:
                guide_random_effect = numpyro.sample(
                    "guide_random_effect",
                    dist.Normal(0.0, guide_random_effect_tau),
                )
        if likelihood in {"lnnb", "lognormal_nb"}:
            noise_scale = numpyro.sample("noise_scale", dist.HalfNormal(0.5))
        if uses_mixture_nb:
            theta_outlier = numpyro.sample("theta_outlier", dist.LogNormal(0.0, 2.0))
            outlier_mean_shift = numpyro.sample("outlier_mean_shift", dist.HalfNormal(1.0))
        with pert_plate:
            beta = _sample_effect_site("beta", prior)
            if fit_perturbation_dispersion:
                dispersion_excess_inverse = numpyro.sample(
                    "dispersion_excess_inverse", dist.Exponential(perturbation_dispersion_prior_rate)
                )

    covariate_coef = None
    if covariates is not None:
        if covariates.ndim != 2:
            raise ValueError("covariates must have shape (num_cells, num_covariates).")
        num_covariates = int(covariates.shape[1])
        if num_covariates > 0:
            with covariate_plate:
                with gene_plate:
                    covariate_coef = numpyro.sample("covariate_coef", dist.Normal(0, 1.0))

    with cell_plate as sampled_cell_idx:
        counts = _subsample_cell_axis(counts, sampled_cell_idx, num_cells=full_num_cells, name="counts")
        pert_id = _subsample_cell_axis(pert_id, sampled_cell_idx, num_cells=full_num_cells, name="pert_id")
        size_factor_obs = _subsample_cell_axis(
            size_factors,
            sampled_cell_idx,
            num_cells=full_num_cells,
            name="size_factors",
        )
        covariates = _subsample_cell_axis(
            covariates,
            sampled_cell_idx,
            num_cells=full_num_cells,
            name="covariates",
        )
        guide_matrix = _subsample_cell_axis(
            guide_matrix,
            sampled_cell_idx,
            num_cells=full_num_cells,
            name="guide_matrix",
        )
        size_factor = numpyro.sample("size_factor", dist.Normal(0.0, 2.0), obs=size_factor_obs)

        if num_factors is not None:
            with factor_plate:
                factor_scores = numpyro.sample("factor_scores", dist.Normal(0, 1.0))

        if pert_id is None:
            raise ValueError("pert_id must be provided.")
        if pert_id.ndim == 1:
            pert_effect = beta[pert_id, :]
            if fit_perturbation_dispersion:
                effective_theta = jnp.reciprocal(jnp.reciprocal(theta)[None, :] + dispersion_excess_inverse[pert_id, :])
        elif pert_id.ndim == 2:
            pert_matrix = jnp.asarray(pert_id, dtype=beta.dtype)
            pert_effect = pert_matrix @ beta
            if fit_perturbation_dispersion:
                effective_theta = jnp.reciprocal(jnp.reciprocal(theta)[None, :] + pert_matrix @ dispersion_excess_inverse)
        else:
            raise ValueError("pert_id must be 1D (indices) or 2D (binary matrix).")

        mu = beta_0 + pert_effect + size_factor
        # Outlier component is a right-shifted baseline mode and does not
        # depend on perturbation beta.
        mu_outlier = beta_0 + size_factor
        if covariate_coef is not None and covariates is not None:
            covariate_contrib = covariates @ covariate_coef
            mu = mu + covariate_contrib
            mu_outlier = mu_outlier + covariate_contrib
        if num_factors is not None:
            factor_contrib = jnp.einsum("fig,fcj->cg", factor_loadings, factor_scores)
            mu = mu + factor_contrib
            mu_outlier = mu_outlier + factor_contrib
        if guide_random_effect is not None and guide_matrix is not None:
            guide_random_effect_contrib = jnp.asarray(guide_matrix, dtype=guide_random_effect.dtype) @ guide_random_effect
            mu = mu + guide_random_effect_contrib
            mu_outlier = mu_outlier + guide_random_effect_contrib

        theta_for_observations = effective_theta if fit_perturbation_dispersion else theta
        logits = mu - jnp.log(theta_for_observations)
        logits_outlier = (mu_outlier + outlier_mean_shift) - jnp.log(theta_outlier) if uses_mixture_nb else None
        if skip_obs_sampling:
            return None

        with gene_plate:
            return _sample_observations(
                counts=counts,
                likelihood=likelihood,
                logits=logits,
                theta=theta_for_observations,
                noise_scale=noise_scale if likelihood in {"lnnb", "lognormal_nb"} else None,
                logits_outlier=logits_outlier,
                theta_outlier=theta_outlier if uses_mixture_nb else None,
                pi_outlier=pi_outlier if uses_mixture_nb else None,
                count_censoring_threshold=count_censoring_threshold,
            )


def GuideSharedEffectModel(
    counts,
    pert_id,
    size_factors=None,
    covariates=None,
    guide_matrix=None,
    guide_to_element=None,
    num_cells=None,
    num_genes=None,
    num_perts=None,
    num_guides=None,
    num_factors=None,
    prior="normal",
    subsample_size: int | None = None,
    likelihood: str = "nb",
    cell_idx: jnp.ndarray | None = None,
    skip_obs_sampling: bool = False,
    guide_effect_strategy: str = "shared",
    guide_random_effects: bool = False,
    fit_perturbation_dispersion: bool = False,
    perturbation_dispersion_prior_rate: float = 10.0,
    count_censoring_threshold=None,
):
    if guide_matrix is None or guide_to_element is None:
        raise ValueError("guide_matrix and guide_to_element must be provided for guide-aware effect models.")

    plates = create_plates(
        counts,
        pert_id,
        covariates=covariates,
        guide_matrix=guide_matrix,
        num_cells=num_cells,
        num_genes=num_genes,
        num_perts=num_perts,
        num_guides=num_guides,
        num_factors=num_factors,
        subsample_size=subsample_size,
        cell_idx=cell_idx,
    )
    cell_plate = plates.cells
    gene_plate = plates.genes
    pert_plate = plates.perts
    guide_plate = plates.guides
    factor_plate = plates.factors
    covariate_plate = plates.covariates
    full_num_cells = int(num_cells) if num_cells is not None else (int(counts.shape[0]) if counts is not None else None)
    uses_mixture_nb = likelihood in {"mixture_nb"}

    if num_factors is not None:
        factor_scale = numpyro.sample("factor_scale", dist.HalfNormal(0.1))
        with factor_plate:
            with gene_plate:
                factor_loadings = numpyro.sample("factor_loadings", dist.Normal(0, factor_scale))
    guide_random_effect_log_tau_loc = None
    guide_random_effect_log_tau_scale = None
    if guide_random_effects:
        # Global (cross-gene) hierarchical hyperparameters.
        guide_random_effect_log_tau_loc = numpyro.sample(
            "guide_random_effect_log_tau_loc",
            dist.Normal(0.0, 1.5),
        )
        guide_random_effect_log_tau_scale = numpyro.sample(
            "guide_random_effect_log_tau_scale",
            dist.HalfNormal(1.0),
        )

    with gene_plate:
        if uses_mixture_nb:
            pi_outlier = numpyro.sample("pi_outlier", dist.Beta(2.0, 100.0))
        beta_0 = numpyro.sample("beta_0", dist.Normal(0, 3))
        theta = numpyro.sample("theta", dist.LogNormal(0.0, 2.0))
        guide_random_effect = None
        if guide_random_effects:
            guide_random_effect_tau = numpyro.sample(
                "guide_random_effect_tau",
                dist.LogNormal(guide_random_effect_log_tau_loc, guide_random_effect_log_tau_scale),
            )
            with guide_plate:
                guide_random_effect = numpyro.sample(
                    "guide_random_effect",
                    dist.Normal(0.0, guide_random_effect_tau),
                )
        if likelihood in {"lnnb", "lognormal_nb"}:
            noise_scale = numpyro.sample("noise_scale", dist.HalfNormal(0.5))
        if uses_mixture_nb:
            theta_outlier = numpyro.sample("theta_outlier", dist.LogNormal(0.0, 2.0))
            outlier_mean_shift = numpyro.sample("outlier_mean_shift", dist.HalfNormal(1.0))
        with pert_plate:
            beta = _sample_effect_site("beta", prior)
        if fit_perturbation_dispersion:
            with guide_plate:
                guide_dispersion_excess_inverse = numpyro.sample(
                    "guide_dispersion_excess_inverse", dist.Exponential(perturbation_dispersion_prior_rate)
                )

    covariate_coef = None
    if covariates is not None:
        if covariates.ndim != 2:
            raise ValueError("covariates must have shape (num_cells, num_covariates).")
        num_covariates = int(covariates.shape[1])
        if num_covariates > 0:
            with covariate_plate:
                with gene_plate:
                    covariate_coef = numpyro.sample("covariate_coef", dist.Normal(0, 1.0))

    guide_to_element = jnp.asarray(guide_to_element, dtype=beta.dtype)
    parent_effect = guide_to_element @ beta
    if guide_effect_strategy == "shared":
        guide_effect = parent_effect
    elif guide_effect_strategy == "relative":
        with gene_plate:
            with guide_plate:
                guide_relative_efficiency = numpyro.sample(
                    "guide_relative_efficiency",
                    dist.Beta(5.0, 1.0),
                )
        guide_effect = parent_effect * guide_relative_efficiency
    elif guide_effect_strategy == "offset":
        with gene_plate:
            with guide_plate:
                guide_offset = _sample_effect_site("guide_offset", prior)
        guide_effect = parent_effect + guide_offset
    else:
        raise ValueError(f"Unknown guide_effect_strategy: {guide_effect_strategy}")

    guide_effect = numpyro.deterministic("guide_effect", guide_effect)

    with cell_plate as sampled_cell_idx:
        counts = _subsample_cell_axis(counts, sampled_cell_idx, num_cells=full_num_cells, name="counts")
        guide_matrix = _subsample_cell_axis(
            guide_matrix,
            sampled_cell_idx,
            num_cells=full_num_cells,
            name="guide_matrix",
        )
        size_factor_obs = _subsample_cell_axis(
            size_factors,
            sampled_cell_idx,
            num_cells=full_num_cells,
            name="size_factors",
        )
        covariates = _subsample_cell_axis(
            covariates,
            sampled_cell_idx,
            num_cells=full_num_cells,
            name="covariates",
        )
        size_factor = numpyro.sample("size_factor", dist.Normal(0.0, 2.0), obs=size_factor_obs)

        if num_factors is not None:
            with factor_plate:
                factor_scores = numpyro.sample("factor_scores", dist.Normal(0, 1.0))

        guide_effect_matrix = jnp.asarray(guide_matrix, dtype=guide_effect.dtype) @ guide_effect
        if fit_perturbation_dispersion:
            effective_theta = jnp.reciprocal(
                jnp.reciprocal(theta)[None, :] + jnp.asarray(guide_matrix, dtype=theta.dtype) @ guide_dispersion_excess_inverse
            )
        mu = beta_0 + guide_effect_matrix + size_factor
        # Outlier component is a right-shifted baseline mode and does not
        # depend on guide/element effect beta.
        mu_outlier = beta_0 + size_factor
        if covariate_coef is not None and covariates is not None:
            covariate_contrib = covariates @ covariate_coef
            mu = mu + covariate_contrib
            mu_outlier = mu_outlier + covariate_contrib
        if num_factors is not None:
            factor_contrib = jnp.einsum("fig,fcj->cg", factor_loadings, factor_scores)
            mu = mu + factor_contrib
            mu_outlier = mu_outlier + factor_contrib
        if guide_random_effect is not None:
            guide_random_effect_contrib = jnp.asarray(guide_matrix, dtype=guide_random_effect.dtype) @ guide_random_effect
            mu = mu + guide_random_effect_contrib
            mu_outlier = mu_outlier + guide_random_effect_contrib

        theta_for_observations = effective_theta if fit_perturbation_dispersion else theta
        logits = mu - jnp.log(theta_for_observations)
        logits_outlier = (mu_outlier + outlier_mean_shift) - jnp.log(theta_outlier) if uses_mixture_nb else None
        if skip_obs_sampling:
            return None

        with gene_plate:
            return _sample_observations(
                counts=counts,
                likelihood=likelihood,
                logits=logits,
                theta=theta_for_observations,
                noise_scale=noise_scale if likelihood in {"lnnb", "lognormal_nb"} else None,
                logits_outlier=logits_outlier,
                theta_outlier=theta_outlier if uses_mixture_nb else None,
                pi_outlier=pi_outlier if uses_mixture_nb else None,
                count_censoring_threshold=count_censoring_threshold,
            )


NegBinModel = partial(BaseModel, likelihood="nb")
CensoredNegativeBinomialModel = partial(BaseModel, likelihood="censored_nb")
LogNormalNegativeBinomialModel = partial(BaseModel, likelihood="lnnb")
MixtureNegativeBinomialModel = partial(BaseModel, likelihood="mixture_nb")
GuideSharedNegativeBinomialModel = partial(GuideSharedEffectModel, likelihood="nb")
GuideSharedCensoredNegativeBinomialModel = partial(GuideSharedEffectModel, likelihood="censored_nb")
GuideSharedLogNormalNegativeBinomialModel = partial(GuideSharedEffectModel, likelihood="lnnb")
GuideSharedMixtureNegativeBinomialModel = partial(GuideSharedEffectModel, likelihood="mixture_nb")
