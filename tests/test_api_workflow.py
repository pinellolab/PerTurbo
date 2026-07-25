"""Integration-style tests for the main API workflow."""

from __future__ import annotations

from dataclasses import replace

import anndata as ad
import perturbo.api as api_module
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from perturbo.api import (
    PerTurboData,
    fit_control,
    fit_perturbation_effects,
    load_controls,
    load_analysis_cells,
)


def _make_guide_shared_data() -> PerTurboData:
    counts = jnp.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
        ],
        dtype=jnp.int32,
    )
    guide_matrix = jnp.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=jnp.float32,
    )
    guide_to_element = jnp.array(
        [
            [1, 0],
            [1, 0],
            [0, 1],
        ],
        dtype=jnp.float32,
    )
    pert_id = (guide_matrix @ guide_to_element > 0).astype(jnp.int8)
    return PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["element_a", "ntc0"],
        gene_names=["g1", "g2", "g3"],
        guide_matrix=guide_matrix,
        guide_names=["guide_a_1", "guide_a_2", "ntc_1"],
        guide_to_element=guide_to_element,
    )


def test_two_stage_fit_with_factors_conditions_loadings() -> None:
    counts = jnp.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
            [2, 1, 0],
            [0, 2, 1],
        ],
        dtype=jnp.int32,
    )
    pert_id = jnp.array([0, 1, 0, 1, 0, 1])
    data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2", "g3"],
    )

    num_factors = 2
    control_fit = fit_control(
        data,
        num_steps=1,
        prior="cauchy",
        model_name="negbin",
        num_factors=num_factors,
    )
    assert control_fit.baseline_posterior is not None
    assert control_fit.baseline_posterior.beta_0_loc.shape == (counts.shape[1],)
    assert control_fit.baseline_posterior.beta_0_scale.shape == (counts.shape[1],)
    assert control_fit.baseline_posterior.theta_log_loc.shape == (counts.shape[1],)
    assert control_fit.baseline_posterior.theta_log_scale.shape == (counts.shape[1],)
    assert control_fit.factor_loadings is not None
    assert control_fit.factor_loadings.shape == (num_factors, 1, counts.shape[1])
    assert control_fit.pca_loadings is not None
    assert control_fit.pca_loadings.shape == (num_factors, 1, counts.shape[1])

    beta_fit = fit_perturbation_effects(
        data,
        control_fit,
        num_steps=1,
        prior="normal",
        model_name="negbin",
        num_factors=num_factors,
    )
    assert beta_fit.posterior_mean.shape == (2, counts.shape[1])


def test_two_stage_fit_with_baseline_uncertainty_marginalization() -> None:
    counts = jnp.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
            [2, 1, 0],
            [0, 2, 1],
        ],
        dtype=jnp.int32,
    )
    pert_id = jnp.array([0, 1, 0, 1, 0, 1])
    data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2", "g3"],
    )
    control_fit = fit_control(
        data,
        num_steps=1,
        prior="cauchy",
        model_name="negbin",
    )
    beta_fit = fit_perturbation_effects(
        data,
        control_fit,
        num_steps=1,
        prior="normal",
        model_name="negbin",
        propagate_baseline_uncertainty=True,
    )

    assert beta_fit.posterior_mean.shape == (2, counts.shape[1])
    assert beta_fit.posterior_scale.shape == (2, counts.shape[1])
    assert jnp.all(jnp.isfinite(beta_fit.posterior_mean))
    assert jnp.all(jnp.isfinite(beta_fit.posterior_scale))


def test_two_stage_fit_supports_mixture_nb() -> None:
    counts = jnp.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
            [2, 1, 0],
            [0, 2, 1],
        ],
        dtype=jnp.int32,
    )
    pert_id = jnp.array([0, 1, 0, 1, 0, 1])
    data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2", "g3"],
    )
    control_fit = fit_control(
        data,
        num_steps=1,
        prior="normal",
        model_name="mixture_nb",
    )
    assert control_fit.pi_outlier is not None
    assert control_fit.pi_outlier.shape == (counts.shape[1],)
    assert control_fit.theta_outlier is not None
    assert control_fit.outlier_mean_shift is not None
    assert control_fit.outlier_mean_shift.shape == (counts.shape[1],)

    beta_fit = fit_perturbation_effects(
        data,
        control_fit,
        num_steps=1,
        prior="normal",
        model_name="mixture_nb",
    )
    assert beta_fit.posterior_mean.shape == (2, counts.shape[1])
    assert beta_fit.posterior_scale.shape == (2, counts.shape[1])


def test_mixture_nb_rejects_baseline_uncertainty_propagation() -> None:
    counts = jnp.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
        ],
        dtype=jnp.int32,
    )
    pert_id = jnp.array([0, 1, 0, 1])
    data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2", "g3"],
    )
    control_fit = fit_control(data, num_steps=1, prior="normal", model_name="mixture_nb")
    with pytest.raises(ValueError, match="not supported"):
        fit_perturbation_effects(
            data,
            control_fit,
            num_steps=1,
            prior="normal",
            model_name="mixture_nb",
            propagate_baseline_uncertainty=True,
        )


def test_two_stage_fit_with_covariates_conditions_covariate_coef() -> None:
    counts = jnp.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
            [2, 1, 0],
            [0, 2, 1],
        ],
        dtype=jnp.int32,
    )
    pert_id = jnp.array([0, 1, 0, 1, 0, 1], dtype=jnp.int32)
    covariates = jnp.array(
        [
            [0.1, -1.0],
            [0.2, -0.2],
            [0.0, 0.1],
            [0.4, 1.2],
            [-0.3, 0.0],
            [0.8, -0.5],
        ],
        dtype=jnp.float32,
    )
    data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2", "g3"],
        covariates=covariates,
        covariate_names=["x1", "x2"],
    )
    control_fit = fit_control(
        data,
        num_steps=1,
        prior="normal",
        model_name="negbin",
    )
    assert control_fit.covariate_coef is not None
    assert control_fit.covariate_coef.shape == (covariates.shape[1], counts.shape[1])

    beta_fit = fit_perturbation_effects(
        data,
        control_fit,
        num_steps=1,
        prior="normal",
        model_name="negbin",
    )
    assert beta_fit.posterior_mean.shape == (2, counts.shape[1])


def test_stage1_guide_random_effects_estimates_gene_tau() -> None:
    data = _make_guide_shared_data()
    control_fit = fit_control(
        data,
        num_steps=1,
        prior="normal",
        model_name="negbin",
        guide_random_effects=True,
    )
    assert control_fit.guide_random_effect_tau is not None
    tau = np.asarray(control_fit.guide_random_effect_tau)
    assert tau.shape == (data.counts.shape[1],)
    assert np.all(np.isfinite(tau))
    assert np.all(tau > 0.0)


def test_two_stage_fit_supports_guide_random_effects_with_mapping() -> None:
    data = _make_guide_shared_data()
    control_fit = fit_control(
        data,
        num_steps=1,
        prior="normal",
        model_name="negbin",
        guide_random_effects=True,
    )
    beta_fit = fit_perturbation_effects(
        data,
        control_fit,
        num_steps=1,
        prior="normal",
        model_name="negbin",
        guide_random_effects=True,
    )
    assert beta_fit.posterior_mean.shape == (2, data.counts.shape[1])
    assert beta_fit.posterior_scale.shape == (2, data.counts.shape[1])
    assert np.all(np.isfinite(np.asarray(beta_fit.posterior_scale)))


def test_chunked_fit_forwards_guide_random_effects() -> None:
    """Regression: chunked beta fitting must accept and forward guide_random_effects.

    The chunked path is taken for large scenarios (cells exceeding max_chunk_size).
    Its signature previously omitted guide_random_effects, so benchmark runs with
    guide-random-effects inference enabled crashed with
    "_fit_perturbation_effects_chunked() got an unexpected keyword argument
    'guide_random_effects'". max_perturbations_per_chunk=1 forces multiple chunks
    on this tiny dataset so the per-chunk delegation is exercised.
    """
    from perturbo.core import _fit_perturbation_effects_chunked

    data = _make_guide_shared_data()
    control_fit = fit_control(
        data,
        num_steps=1,
        prior="normal",
        model_name="negbin",
        guide_random_effects=True,
    )
    beta_fit = _fit_perturbation_effects_chunked(
        data,
        control_fit,
        num_steps=1,
        prior="normal",
        model_name="negbin",
        guide_random_effects=True,
        max_perturbations_per_chunk=1,
    )
    assert beta_fit.posterior_mean.shape == (len(data.pert_names), data.counts.shape[1])
    assert np.all(np.isfinite(np.asarray(beta_fit.posterior_scale)))


def test_stage2_guide_random_effects_fallback_inflates_scale_without_mapping() -> None:
    counts = jnp.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
            [2, 1, 0],
            [0, 2, 1],
        ],
        dtype=jnp.int32,
    )
    pert_id = jnp.array([0, 1, 0, 1, 0, 1], dtype=jnp.int32)
    data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2", "g3"],
    )
    control_fit = fit_control(
        data,
        num_steps=1,
        prior="normal",
        model_name="negbin",
        guide_random_effects=True,
    )
    assert control_fit.guide_random_effect_tau is not None
    beta_fit_raw = fit_perturbation_effects(
        data,
        control_fit,
        num_steps=1,
        prior="normal",
        model_name="negbin",
        guide_random_effects=False,
    )
    beta_fit_inflated = fit_perturbation_effects(
        data,
        control_fit,
        num_steps=1,
        prior="normal",
        model_name="negbin",
        guide_random_effects=True,
    )
    scale_raw = np.asarray(beta_fit_raw.posterior_scale, dtype=np.float64)
    scale_inflated = np.asarray(beta_fit_inflated.posterior_scale, dtype=np.float64)
    tau = np.asarray(control_fit.guide_random_effect_tau, dtype=np.float64)
    expected = np.sqrt(np.square(scale_raw) + np.square(tau)[None, :])
    np.testing.assert_allclose(scale_inflated, expected, rtol=1e-6, atol=1e-6)
    assert np.all(scale_inflated >= scale_raw)


def test_fit_perturbation_effects_errors_when_covariates_missing_stage1_coefficients() -> None:
    counts = jnp.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
        ],
        dtype=jnp.int32,
    )
    pert_id = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    covariates = jnp.array(
        [
            [0.1],
            [0.2],
            [0.3],
            [0.4],
        ],
        dtype=jnp.float32,
    )
    data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2", "g3"],
        covariates=covariates,
        covariate_names=["x1"],
    )
    control_fit = fit_control(data, num_steps=1, prior="normal", model_name="negbin")
    broken_control_fit = replace(control_fit, covariate_coef=None)
    with pytest.raises(ValueError, match="covariate_coef"):
        fit_perturbation_effects(
            data,
            broken_control_fit,
            num_steps=1,
            prior="normal",
            model_name="negbin",
        )


def test_two_stage_fit_supports_relative_guide_sharing() -> None:
    data = _make_guide_shared_data()
    control_fit = fit_control(data, num_steps=1, prior="normal", model_name="negbin")

    beta_fit = fit_perturbation_effects(
        data,
        control_fit,
        num_steps=1,
        prior="normal",
        model_name="negbin",
        guide_effect_strategy="relative",
    )

    assert beta_fit.posterior_mean.shape == (2, data.counts.shape[1])
    assert beta_fit.guide_effect_mean is not None
    assert beta_fit.guide_effect_mean.shape == (3, data.counts.shape[1])
    assert beta_fit.guide_relative_efficiency_mean is not None
    assert beta_fit.guide_relative_efficiency_mean.shape == (3, data.counts.shape[1])


def test_two_stage_fit_rejects_offset_guide_sharing() -> None:
    data = _make_guide_shared_data()
    control_fit = fit_control(data, num_steps=1, prior="normal", model_name="negbin")

    with pytest.raises(ValueError, match="guide_effect_strategy"):
        fit_perturbation_effects(
            data,
            control_fit,
            num_steps=1,
            prior="normal",
            model_name="negbin",
            guide_effect_strategy="offset",
        )


def test_absolute_guide_activity_mode_is_guarded() -> None:
    data = _make_guide_shared_data()
    control_fit = fit_control(data, num_steps=1, prior="normal", model_name="negbin")

    with pytest.raises(ValueError, match="not implemented yet"):
        fit_perturbation_effects(
            data,
            control_fit,
            num_steps=1,
            prior="normal",
            model_name="negbin",
            guide_activity_mode="absolute",
        )


def test_loader_covariates_stage_consistency_with_control_fitted_transform() -> None:
    counts = np.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
            [2, 1, 0],
            [0, 2, 1],
            [1, 1, 0],
            [0, 1, 2],
        ],
        dtype=np.int32,
    )
    obs = pd.DataFrame(
        {
            "perturbation": ["ctrl", "ctrl", "pertA", "pertA", "pertB", "ctrl", "pertB", "pertA"],
            "percent_mito": [0.1, 0.2, 0.15, 0.25, 0.18, 0.05, 0.31, 0.22],
            "guide_count": [0, 1, 2, 1, 3, 0, 1, 2],
            "prep_batch": ["b1", "b2", "b1", "b3", "b2", "b1", None, "b3"],
        },
        index=[f"c{i}" for i in range(counts.shape[0])],
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    adata = ad.AnnData(X=counts, obs=obs, var=var)

    controls_loaded = load_controls(
        adata,
        perturbation_key="perturbation",
        control_selector="ctrl",
        continuous_covariates=["percent_mito", "guide_count"],
        batch_covariate="prep_batch",
        max_control_cells=None,
        return_covariate_transform_state=True,
    )
    assert isinstance(controls_loaded, tuple)
    controls, cov_state = controls_loaded
    assert controls.covariates is not None
    assert controls.covariate_names is not None
    assert cov_state is not None

    analysis_data = load_analysis_cells(
        adata,
        perturbation_key="perturbation",
        covariate_transform_state=cov_state,
        continuous_covariates=["percent_mito", "guide_count"],
        batch_covariate="prep_batch",
    )
    assert analysis_data.covariates is not None
    assert analysis_data.covariate_names == controls.covariate_names
    assert analysis_data.covariates.shape[1] == len(analysis_data.covariate_names)

    control_fit = fit_control(
        controls,
        num_steps=1,
        prior="normal",
        model_name="negbin",
    )
    assert control_fit.covariate_coef is not None
    beta_fit = fit_perturbation_effects(
        analysis_data,
        control_fit,
        num_steps=1,
        prior="normal",
        model_name="negbin",
    )
    assert beta_fit.posterior_mean.shape[1] == counts.shape[1]


def test_loaders_can_winsorize_gene_counts_to_shared_thresholds() -> None:
    counts = np.array(
        [
            [0, 7],
            [9, 1],
            [4, 8],
            [6, 2],
        ],
        dtype=np.int32,
    )
    obs = pd.DataFrame(
        {"perturbation": ["ctrl", "pertA", "ctrl", "pertB"]},
        index=["c0", "c1", "c2", "c3"],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    gene_clip_thresholds = np.array([5, 3], dtype=np.int32)

    controls = load_controls(
        adata,
        perturbation_key="perturbation",
        control_selector="ctrl",
        max_control_cells=None,
        clip_gene_expression_percentile=99.0,
        winsorize_gene_expression=True,
        gene_outlier_threshold_floor=10,
        gene_clip_thresholds=gene_clip_thresholds,
    )
    analysis_data = load_analysis_cells(
        adata,
        perturbation_key="perturbation",
        clip_gene_expression_percentile=99.0,
        winsorize_gene_expression=True,
        gene_outlier_threshold_floor=10,
        gene_clip_thresholds=gene_clip_thresholds,
    )

    assert np.array_equal(
        np.asarray(controls.counts),
        np.array(
            [
                [0, 3],
                [4, 3],
            ],
            dtype=np.int32,
        ),
    )
    assert np.array_equal(
        np.asarray(analysis_data.counts),
        np.array(
            [
                [0, 3],
                [5, 1],
                [4, 3],
                [5, 2],
            ],
            dtype=np.int32,
        ),
    )


def test_gene_outlier_thresholds_have_minimum_of_two() -> None:
    counts = np.array([[0, 0], [1, 1], [1, 2], [0, 1]], dtype=np.int32)
    thresholds = api_module._compute_gene_outlier_thresholds(counts, percentile=99.0)
    assert np.array_equal(thresholds, np.array([2, 2], dtype=np.int32))


def test_loaders_apply_cell_keep_mask_before_subsetting() -> None:
    counts = np.array(
        [
            [0, 1],
            [4, 0],
            [2, 3],
            [5, 1],
        ],
        dtype=np.int32,
    )
    obs = pd.DataFrame(
        {"perturbation": ["ctrl", "pertA", "ctrl", "pertB"]},
        index=["c0", "c1", "c2", "c3"],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    keep_mask = np.array([True, False, True, True], dtype=bool)

    controls = load_controls(
        adata,
        perturbation_key="perturbation",
        control_selector="ctrl",
        max_control_cells=None,
        cell_keep_mask=keep_mask,
    )
    analysis_data = load_analysis_cells(
        adata,
        perturbation_key="perturbation",
        cell_keep_mask=keep_mask,
    )

    assert controls.counts.shape == (2, 2)
    assert np.array_equal(np.asarray(controls.counts), counts[[0, 2]])
    assert analysis_data.counts.shape == (3, 2)
    assert np.array_equal(np.asarray(analysis_data.counts), counts[[0, 2, 3]])
    assert analysis_data.pert_names == ["ctrl", "pertB"]


def test_analysis_loader_selected_perturbations_matches_canonical_call() -> None:
    counts = np.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
        ],
        dtype=np.int32,
    )
    obs = pd.DataFrame({"perturbation": ["ctrl", "pertA", "pertB"]}, index=["c0", "c1", "c2"])
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    adata = ad.AnnData(X=counts, obs=obs, var=var)

    loaded = load_analysis_cells(adata, perturbation_key="perturbation", selected_perturbations=["pertA", "pertB"])

    assert loaded.pert_names == ["pertA", "pertB"]
    assert loaded.gene_names == ["g1", "g2", "g3"]
    assert np.array_equal(np.asarray(loaded.counts), counts[[1, 2]])
    assert np.array_equal(np.asarray(loaded.pert_id), np.array([0, 1]))


def test_analysis_library_size_centering_can_be_anchored_to_controls() -> None:
    counts = np.array(
        [
            [2, 1],
            [1, 2],
            [5, 4],
            [6, 5],
        ],
        dtype=np.int32,
    )
    obs = pd.DataFrame(
        {
            "perturbation": ["ctrl", "ctrl", "pertA", "pertB"],
            "umi_count": [10, 12, 80, 120],
        },
        index=["c0", "c1", "c2", "c3"],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    adata = ad.AnnData(X=counts, obs=obs, var=var)

    controls = load_controls(
        adata,
        perturbation_key="perturbation",
        control_selector="ctrl",
        max_control_cells=None,
        size_factor_key=None,
        library_size_key="umi_count",
    )
    assert controls.library_size_center_log_mean is not None

    analysis_data = load_analysis_cells(
        adata,
        perturbation_key="perturbation",
        size_factor_key=None,
        library_size_key="umi_count",
        library_size_center_log_mean=controls.library_size_center_log_mean,
    )
    expected = np.log1p(obs["umi_count"].to_numpy(dtype=np.float64)) - float(controls.library_size_center_log_mean)
    np.testing.assert_allclose(np.asarray(analysis_data.size_factors).reshape(-1), expected, atol=1e-6)


def test_library_size_key_requires_positive_integer_values() -> None:
    counts = np.array([[1, 0], [0, 1]], dtype=np.int32)
    obs = pd.DataFrame(
        {
            "perturbation": ["ctrl", "pertA"],
            "umi_count": [5.5, 10.0],
        },
        index=["c0", "c1"],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    adata = ad.AnnData(X=counts, obs=obs, var=var)

    with pytest.raises(ValueError, match="must contain only positive integers"):
        load_controls(
            adata,
            perturbation_key="perturbation",
            control_selector="ctrl",
            max_control_cells=None,
            size_factor_key=None,
            library_size_key="umi_count",
        )


def test_size_factor_key_rejects_all_positive_values() -> None:
    counts = np.array([[1, 0], [0, 1]], dtype=np.int32)
    obs = pd.DataFrame(
        {
            "perturbation": ["ctrl", "pertA"],
            "sf": [0.25, 0.5],
        },
        index=["c0", "c1"],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    adata = ad.AnnData(X=counts, obs=obs, var=var)

    with pytest.raises(ValueError, match="contains only positive values"):
        load_analysis_cells(
            adata,
            perturbation_key="perturbation",
            size_factor_key="sf",
            library_size_key=None,
        )


def test_size_factor_key_rejects_integer_count_like_values() -> None:
    counts = np.array([[1, 0], [0, 1]], dtype=np.int32)
    obs = pd.DataFrame(
        {
            "perturbation": ["ctrl", "pertA"],
            "sf": [10, 20],
        },
        index=["c0", "c1"],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    adata = ad.AnnData(X=counts, obs=obs, var=var)

    with pytest.raises(ValueError, match="appears to contain integer counts"):
        load_analysis_cells(
            adata,
            perturbation_key="perturbation",
            size_factor_key="sf",
            library_size_key=None,
        )
