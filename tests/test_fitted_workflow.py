from __future__ import annotations

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import jax.numpy as jnp
import pytest

import perturbo
import perturbo.core as core
import perturbo.inference as inference
from perturbo.io import MuDataSetup, control_fit_arrays, save_light_fit_bundle
from perturbo.simulation.fitted import _resolve_perturbation_dispersion_theta, _resolve_size_factors


def _make_registered_mdata(*, size_factor_key: str | None = None) -> md.MuData:
    obs = pd.DataFrame(
        {
            "prep_batch": ["b1", "b1", "b2", "b2"],
            "umi_count": [12, 10, 11, 13],
            "manual_size_factor": [-1.5, -0.5, 0.25, 1.75],
            "percent_mito": [0.05, 0.07, 0.03, 0.06],
            "log1p_guide_count": [0.0, np.log1p(1.0), np.log1p(1.0), np.log1p(2.0)],
        },
        index=[f"cell_{i}" for i in range(4)],
    )
    rna = ad.AnnData(
        X=np.array(
            [
                [5, 1],
                [4, 2],
                [2, 5],
                [1, 6],
            ],
            dtype=np.int32,
        ),
        obs=obs.copy(),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )
    guide = ad.AnnData(
        X=np.array(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 1],
            ],
            dtype=np.int8,
        ),
        obs=obs.copy(),
        var=pd.DataFrame(index=["guide_a_1", "guide_a_2", "ntc_1"]),
    )
    guide.varm["element_targeted"] = pd.DataFrame(
        np.array(
            [
                [1, 0],
                [1, 0],
                [0, 1],
            ],
            dtype=np.float32,
        ),
        index=guide.var_names,
        columns=["element_a", "ntc0"],
    )
    guide.uns["elements"] = np.array(["element_a", "ntc0"], dtype=object)
    rna.varm["element_tested"] = pd.DataFrame(
        np.array(
            [
                [1, 0],
                [0, 1],
            ],
            dtype=np.float32,
        ),
        index=rna.var_names,
        columns=["element_a", "ntc0"],
    )
    mdata = md.MuData({"rna": rna, "grna": guide})
    perturbo.setup_mudata(
        mdata,
        batch_key="prep_batch",
        library_size_key="umi_count",
        size_factor_key=size_factor_key,
        continuous_covariates_keys=["percent_mito", "log1p_guide_count"],
        gene_by_element_key="element_tested",
        guide_by_element_key="element_targeted",
        guide_element_uns_key="elements",
        modalities={"rna_layer": "rna", "perturbation_layer": "grna"},
    )
    return mdata


def test_load_controls_prefers_explicit_size_factor_column() -> None:
    adata = ad.AnnData(
        X=np.array([[5, 1], [4, 2], [2, 5], [1, 6]], dtype=np.int32),
        obs=pd.DataFrame(
            {
                "perturbation": ["ctrl", "ctrl", "pertA", "pertB"],
                "umi_count": [12, 10, 11, 13],
                "provided_sf": [-1.5, -0.5, 0.25, 1.75],
            },
            index=[f"cell_{i}" for i in range(4)],
        ),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )

    controls = core.load_controls(
        adata,
        perturbation_key="perturbation",
        control_selector="ctrl",
        size_factor_key="provided_sf",
        library_size_key="umi_count",
    )

    np.testing.assert_allclose(
        np.asarray(controls.size_factors),
        np.array([[-1.5], [-0.5]], dtype=np.float32),
    )


def test_fitted_model_exposes_public_posteriors() -> None:
    model = perturbo.PERTURBO(_make_registered_mdata(), likelihood="negbin", effect_prior_dist="normal")
    model.train(steps=1, batch_size=2, accelerator="cpu")

    medians = model.posterior_medians()
    assert "beta_0" in medians
    assert "element_effects" in medians
    assert "guide_effects" in medians
    assert "guide_efficacy" in medians
    assert medians["element_effects"].shape == (2, 2)
    assert medians["guide_effects"].shape == (3, 2)

    effects = model.get_element_effects()
    assert set(effects.columns) == {"element", "gene", "loc", "scale", "z_value", "q_value"}
    assert len(effects) == 4
    guide_effects = model.get_guide_effects()
    assert set(guide_effects.columns) == {"guide", "element", "gene", "loc", "scale", "z_value", "q_value"}
    assert len(guide_effects) == 6


def test_perturbo_normalizes_nb_likelihood_alias() -> None:
    model = perturbo.PERTURBO(_make_registered_mdata(), likelihood="nb", effect_prior_dist="normal")
    assert model.likelihood == "negbin"


def test_train_threads_censored_percentile_for_censored_nb(monkeypatch) -> None:
    model = perturbo.PERTURBO(
        _make_registered_mdata(),
        likelihood="censored_nb",
        effect_prior_dist="normal",
        count_censoring_percentile=97.5,
    )
    control_data = perturbo.PerTurboData(
        counts=jnp.ones((4, 2), dtype=jnp.int32),
        pert_id=jnp.zeros((4,), dtype=jnp.int32),
        pert_names=["ctrl"],
        gene_names=["g1", "g2"],
    )
    analysis_data = perturbo.PerTurboData(
        counts=jnp.ones((4, 2), dtype=jnp.int32),
        pert_id=jnp.zeros((4,), dtype=jnp.int32),
        pert_names=["ctrl"],
        gene_names=["g1", "g2"],
    )
    captured: dict[str, float] = {}

    monkeypatch.setattr(inference.core, "load_controls", lambda *args, **kwargs: (control_data, None))
    monkeypatch.setattr(inference.core, "load_analysis_cells", lambda *args, **kwargs: analysis_data)

    def _fake_fit_control(*args, **kwargs):
        captured["control"] = kwargs["count_censoring_percentile"]
        return perturbo.ControlFit(
            beta_0=jnp.zeros((2,)),
            theta=jnp.ones((2,)),
            noise_scale=jnp.ones((2,)),
            factor_loadings=None,
            factor_scores=None,
            factor_center=None,
            pca_loadings=None,
            size_factors=jnp.zeros((4, 1)),
            losses=jnp.array([1.0]),
            svi_result=None,
        )

    def _fake_fit_perturbation_effects(*args, **kwargs):
        captured["beta"] = kwargs["count_censoring_percentile"]
        return perturbo.BetaFit(
            posterior_mean=jnp.zeros((1, 2)),
            posterior_scale=jnp.ones((1, 2)),
            z_values=jnp.zeros((1, 2)),
            losses=jnp.array([1.0]),
            svi_result=None,
        )

    monkeypatch.setattr(inference.core, "fit_control", _fake_fit_control)
    monkeypatch.setattr(inference.core, "fit_perturbation_effects", _fake_fit_perturbation_effects)

    model.train(steps=1, batch_size=2, accelerator="cpu")

    assert captured["control"] == pytest.approx(97.5)
    assert captured["beta"] == pytest.approx(97.5)


def test_train_threads_gene_count_management_options(monkeypatch) -> None:
    model = perturbo.PERTURBO(
        _make_registered_mdata(),
        likelihood="negbin",
        effect_prior_dist="normal",
        clip_gene_expression_percentile=99.0,
        winsorize_gene_expression=True,
        gene_outlier_threshold_floor=7,
    )
    control_data = perturbo.PerTurboData(
        counts=jnp.ones((4, 2), dtype=jnp.int32),
        pert_id=jnp.zeros((4,), dtype=jnp.int32),
        pert_names=["ctrl"],
        gene_names=["g1", "g2"],
    )
    analysis_data = perturbo.PerTurboData(
        counts=jnp.ones((4, 2), dtype=jnp.int32),
        pert_id=jnp.zeros((4,), dtype=jnp.int32),
        pert_names=["ctrl"],
        gene_names=["g1", "g2"],
    )
    captured: dict[str, object] = {}

    def _fake_load_controls(*args, **kwargs):
        captured["control_clip_percentile"] = kwargs["clip_gene_expression_percentile"]
        captured["control_winsorize"] = kwargs["winsorize_gene_expression"]
        captured["control_floor"] = kwargs["gene_outlier_threshold_floor"]
        return control_data, None

    def _fake_load_analysis_cells(*args, **kwargs):
        captured["analysis_clip_percentile"] = kwargs["clip_gene_expression_percentile"]
        captured["analysis_winsorize"] = kwargs["winsorize_gene_expression"]
        captured["analysis_floor"] = kwargs["gene_outlier_threshold_floor"]
        return analysis_data

    monkeypatch.setattr(inference.core, "load_controls", _fake_load_controls)
    monkeypatch.setattr(inference.core, "load_analysis_cells", _fake_load_analysis_cells)
    monkeypatch.setattr(
        inference.core,
        "fit_control",
        lambda *args, **kwargs: perturbo.ControlFit(
            beta_0=jnp.zeros((2,)),
            theta=jnp.ones((2,)),
            noise_scale=jnp.ones((2,)),
            factor_loadings=None,
            factor_scores=None,
            factor_center=None,
            pca_loadings=None,
            size_factors=jnp.zeros((4, 1)),
            losses=jnp.array([1.0]),
            svi_result=None,
        ),
    )
    monkeypatch.setattr(
        inference.core,
        "fit_perturbation_effects",
        lambda *args, **kwargs: perturbo.BetaFit(
            posterior_mean=jnp.zeros((1, 2)),
            posterior_scale=jnp.ones((1, 2)),
            z_values=jnp.zeros((1, 2)),
            losses=jnp.array([1.0]),
            svi_result=None,
        ),
    )

    model.train(steps=1, batch_size=2, accelerator="cpu")

    assert captured["control_clip_percentile"] == pytest.approx(99.0)
    assert captured["analysis_clip_percentile"] == pytest.approx(99.0)
    assert captured["control_winsorize"] is True
    assert captured["analysis_winsorize"] is True
    assert captured["control_floor"] == 7
    assert captured["analysis_floor"] == 7


def test_train_uses_registered_size_factors_in_observed_mode(monkeypatch) -> None:
    model = perturbo.PERTURBO(
        _make_registered_mdata(size_factor_key="manual_size_factor"),
        likelihood="negbin",
        effect_prior_dist="normal",
    )
    control_data = perturbo.PerTurboData(
        counts=jnp.ones((4, 2), dtype=jnp.int32),
        pert_id=jnp.zeros((4,), dtype=jnp.int32),
        pert_names=["ctrl"],
        gene_names=["g1", "g2"],
        size_factors=jnp.arange(4, dtype=jnp.float32).reshape(4, 1),
    )
    analysis_data = perturbo.PerTurboData(
        counts=jnp.ones((4, 2), dtype=jnp.int32),
        pert_id=jnp.zeros((4,), dtype=jnp.int32),
        pert_names=["ctrl"],
        gene_names=["g1", "g2"],
        size_factors=jnp.arange(4, dtype=jnp.float32).reshape(4, 1),
    )
    captured: dict[str, object] = {}

    def _fake_load_controls(*args, **kwargs):
        captured["control_size_factor_key"] = kwargs["size_factor_key"]
        captured["control_library_size_key"] = kwargs["library_size_key"]
        return control_data, None

    def _fake_load_analysis_cells(*args, **kwargs):
        captured["analysis_size_factor_key"] = kwargs["size_factor_key"]
        captured["analysis_library_size_key"] = kwargs["library_size_key"]
        return analysis_data

    def _fake_fit_control(*args, **kwargs):
        captured["control_use_observed"] = kwargs["use_observed_size_factors"]
        return perturbo.ControlFit(
            beta_0=jnp.zeros((2,)),
            theta=jnp.ones((2,)),
            noise_scale=jnp.ones((2,)),
            factor_loadings=None,
            factor_scores=None,
            factor_center=None,
            pca_loadings=None,
            size_factors=jnp.zeros((4, 1)),
            losses=jnp.array([1.0]),
            svi_result=None,
        )

    def _fake_fit_perturbation_effects(*args, **kwargs):
        captured["beta_use_observed"] = kwargs["use_observed_size_factors"]
        return perturbo.BetaFit(
            posterior_mean=jnp.zeros((1, 2)),
            posterior_scale=jnp.ones((1, 2)),
            z_values=jnp.zeros((1, 2)),
            losses=jnp.array([1.0]),
            svi_result=None,
        )

    monkeypatch.setattr(inference.core, "load_controls", _fake_load_controls)
    monkeypatch.setattr(inference.core, "load_analysis_cells", _fake_load_analysis_cells)
    monkeypatch.setattr(inference.core, "fit_control", _fake_fit_control)
    monkeypatch.setattr(inference.core, "fit_perturbation_effects", _fake_fit_perturbation_effects)

    model.train(steps=1, batch_size=2, accelerator="cpu")

    assert captured["control_size_factor_key"] == "manual_size_factor"
    assert captured["analysis_size_factor_key"] == "manual_size_factor"
    assert captured["control_library_size_key"] == "umi_count"
    assert captured["analysis_library_size_key"] == "umi_count"
    assert captured["control_use_observed"] is True
    assert captured["beta_use_observed"] is True


def test_train_threads_control_library_size_center_into_analysis_loader(monkeypatch) -> None:
    model = perturbo.PERTURBO(
        _make_registered_mdata(),
        likelihood="negbin",
        effect_prior_dist="normal",
    )
    control_data = perturbo.PerTurboData(
        counts=jnp.ones((4, 2), dtype=jnp.int32),
        pert_id=jnp.zeros((4,), dtype=jnp.int32),
        pert_names=["ctrl"],
        gene_names=["g1", "g2"],
        size_factors=jnp.zeros((4, 1), dtype=jnp.float32),
        library_size_center_log_mean=1.2345,
    )
    analysis_data = perturbo.PerTurboData(
        counts=jnp.ones((4, 2), dtype=jnp.int32),
        pert_id=jnp.zeros((4,), dtype=jnp.int32),
        pert_names=["ctrl"],
        gene_names=["g1", "g2"],
        size_factors=jnp.zeros((4, 1), dtype=jnp.float32),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(inference.core, "load_controls", lambda *args, **kwargs: (control_data, None))

    def _fake_load_analysis_cells(*args, **kwargs):
        captured["library_size_center_log_mean"] = kwargs["library_size_center_log_mean"]
        return analysis_data

    monkeypatch.setattr(inference.core, "load_analysis_cells", _fake_load_analysis_cells)
    monkeypatch.setattr(
        inference.core,
        "fit_control",
        lambda *args, **kwargs: perturbo.ControlFit(
            beta_0=jnp.zeros((2,)),
            theta=jnp.ones((2,)),
            noise_scale=jnp.ones((2,)),
            factor_loadings=None,
            factor_scores=None,
            factor_center=None,
            pca_loadings=None,
            size_factors=jnp.zeros((4, 1)),
            losses=jnp.array([1.0]),
            svi_result=None,
        ),
    )
    monkeypatch.setattr(
        inference.core,
        "fit_perturbation_effects",
        lambda *args, **kwargs: perturbo.BetaFit(
            posterior_mean=jnp.zeros((1, 2)),
            posterior_scale=jnp.ones((1, 2)),
            z_values=jnp.zeros((1, 2)),
            losses=jnp.array([1.0]),
            svi_result=None,
        ),
    )

    model.train(steps=1, batch_size=2, accelerator="cpu")

    assert captured["library_size_center_log_mean"] == pytest.approx(1.2345)


def test_fit_bundle_roundtrip_preserves_effects_and_sidecars(tmp_path) -> None:
    model = perturbo.PERTURBO(
        _make_registered_mdata(),
        likelihood="negbin",
        effect_prior_dist="normal",
        clip_gene_expression_percentile=98.0,
        winsorize_gene_expression=True,
        gene_outlier_threshold_floor=5,
        count_censoring_percentile=97.5,
    )
    model.train(steps=1, batch_size=2, accelerator="cpu")

    bundle_dir = model.save(tmp_path / "bundle", overwrite=True)
    assert (bundle_dir / "mdata.h5mu").exists()
    assert (bundle_dir / "metadata.json").exists()
    assert (bundle_dir / "element_effects.parquet").exists()
    assert not (bundle_dir / "element_effects.csv").exists()
    assert not (bundle_dir / "guide_effects.parquet").exists()
    assert (bundle_dir / "guide_efficacy.npy").exists()

    loaded = perturbo.PERTURBO.load(bundle_dir)
    assert loaded.clip_gene_expression_percentile == pytest.approx(98.0)
    assert loaded.winsorize_gene_expression is True
    assert loaded.gene_outlier_threshold_floor == 5
    assert loaded.count_censoring_percentile == pytest.approx(97.5)
    np.testing.assert_allclose(
        loaded.posterior_medians()["element_effects"],
        model.posterior_medians()["element_effects"],
    )
    np.testing.assert_allclose(
        loaded.posterior_medians()["guide_effects"],
        model.posterior_medians()["guide_effects"],
    )
    np.testing.assert_allclose(model.guide_efficacy, np.ones((3,), dtype=np.float32))
    np.testing.assert_allclose(loaded.guide_efficacy, model.guide_efficacy)


def test_light_bundle_loads_low_moi_source_and_simulates_guide_random_effects(tmp_path) -> None:
    adata = ad.AnnData(
        X=np.array([[5, 1], [4, 2], [2, 5], [1, 6]], dtype=np.int32),
        obs=pd.DataFrame(
            {
                "perturbation": ["ctrl", "pertA", "pertB", "ctrl"],
                "umi_count": [12, 10, 11, 13],
                "batch": ["a", "a", "b", "b"],
            },
            index=[f"cell_{i}" for i in range(4)],
        ),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )
    source_path = tmp_path / "source.h5ad"
    adata.write_h5ad(source_path)

    control_fit = perturbo.ControlFit(
        beta_0=jnp.array([1.0, 1.2], dtype=jnp.float32),
        theta=jnp.array([2.0, 3.0], dtype=jnp.float32),
        noise_scale=jnp.ones((2,), dtype=jnp.float32),
        factor_loadings=None,
        factor_scores=None,
        factor_center=None,
        pca_loadings=None,
        size_factors=jnp.zeros((2, 1), dtype=jnp.float32),
        losses=jnp.array([1.0], dtype=jnp.float32),
        svi_result=None,
        guide_random_effect_tau=jnp.array([0.2, 0.3], dtype=jnp.float32),
        guide_random_effect_log_tau_loc=jnp.array([-1.0, -0.5], dtype=jnp.float32),
        guide_random_effect_log_tau_scale=jnp.array([0.1, 0.2], dtype=jnp.float32),
    )
    beta_arrays = {
        "posterior_mean": np.zeros((3, 2), dtype=np.float32),
        "posterior_scale": np.ones((3, 2), dtype=np.float32),
        "z_values": np.zeros((3, 2), dtype=np.float32),
        "losses": np.array([1.0], dtype=np.float32),
    }
    setup = MuDataSetup(
        rna_modality="rna",
        perturbation_modality="grna",
        library_size_key="umi_count",
        control_substring="ctrl",
    )
    center = float(np.mean(np.log1p([12, 13])))
    metadata = {
        "likelihood": "negbin",
        "effect_prior_dist": "normal",
        "efficiency_mode": "shared",
        "guide_effect_strategy": "shared",
        "guide_activity_mode": "always_on",
        "n_factors": None,
        "clip_gene_expression_percentile": 100.0,
        "winsorize_gene_expression": False,
        "gene_outlier_threshold_floor": 2,
        "count_censoring_percentile": None,
        "guide_random_effects": True,
        "library_size_center_log_mean": center,
        "setup": setup.to_json_dict(),
        "svi_config": {"step_size": 0.01},
        "covariate_transform_state": None,
        "source": {"path": str(source_path), "kind": "h5ad", "backed": True},
        "input_mode": "low_moi",
        "perturbation_key": "perturbation",
        "gene_names": ["gene_a", "gene_b"],
        "perturbation_names": ["ctrl", "pertA", "pertB"],
        "guide_names": ["ctrl", "pertA", "pertB"],
    }
    bundle_dir = save_light_fit_bundle(
        tmp_path / "bundle",
        metadata=metadata,
        control_arrays=control_fit_arrays(control_fit),
        beta_arrays=beta_arrays,
        guide_efficacy=np.ones((3,), dtype=np.float32),
    )
    pd.DataFrame({"element": ["ctrl"], "gene": ["gene_a"]}).to_parquet(bundle_dir / "element_effects.parquet")

    loaded_payload = perturbo.load_fit_bundle(bundle_dir)
    assert loaded_payload["element_effects"] is not None
    loaded = perturbo.PERTURBO.load(bundle_dir)
    assert loaded.setup.perturbation_modality == "grna"
    assert loaded.element_names == ["ctrl", "pertA", "pertB"]
    assert loaded.control_fit is not None
    np.testing.assert_allclose(np.asarray(loaded.control_fit.guide_random_effect_tau), [0.2, 0.3])
    np.testing.assert_allclose(
        _resolve_size_factors(loaded, np.array([0], dtype=np.int32), 1.0),
        np.array([[np.log1p(12.0) - center]], dtype=np.float32),
    )

    simulated = perturbo.simulate_data_from_trained_model(
        loaded,
        guide_obs=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        guide_by_element=np.eye(3, dtype=np.float32),
        element_by_gene_lfc=np.zeros((3, 2), dtype=np.float32),
        guide_efficacy=np.ones((3,), dtype=np.float32),
        cell_indices=np.array([0, 1], dtype=np.int32),
    )
    assert simulated["rna"].X.shape == (2, 2)


def test_censored_nb_bundle_simulation_caps_counts_at_saved_threshold(tmp_path) -> None:
    model = perturbo.PERTURBO(
        _make_registered_mdata(),
        likelihood="censored_nb",
        effect_prior_dist="normal",
        count_censoring_percentile=90.0,
    )
    model.control_fit = perturbo.ControlFit(
        beta_0=jnp.array([5.0, 5.0], dtype=jnp.float32),
        theta=jnp.array([2.0, 2.0], dtype=jnp.float32),
        noise_scale=jnp.ones((2,), dtype=jnp.float32),
        factor_loadings=None,
        factor_scores=None,
        factor_center=None,
        pca_loadings=None,
        size_factors=jnp.zeros((4, 1), dtype=jnp.float32),
        losses=jnp.array([], dtype=jnp.float32),
        svi_result=None,
        count_censoring_threshold=jnp.array([1, 2], dtype=jnp.int32),
    )
    model.beta_fit = perturbo.BetaFit(
        posterior_mean=jnp.zeros((2, 2), dtype=jnp.float32),
        posterior_scale=jnp.ones((2, 2), dtype=jnp.float32),
        z_values=jnp.zeros((2, 2), dtype=jnp.float32),
        losses=jnp.array([], dtype=jnp.float32),
        svi_result=None,
    )
    bundle_dir = model.save(tmp_path / "censored_bundle", overwrite=True)
    loaded = perturbo.PERTURBO.load(bundle_dir)

    simulated = perturbo.simulate_data_from_trained_model(
        loaded,
        guide_obs=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        guide_by_element=np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32),
        element_by_gene_lfc=np.zeros((2, 2), dtype=np.float32),
        guide_efficacy=np.ones((3,), dtype=np.float32),
        cell_indices=np.array([0, 1], dtype=np.int32),
    )
    assert int(np.asarray(simulated["rna"].X)[:, 0].max()) <= 1
    assert int(np.asarray(simulated["rna"].X)[:, 1].max()) <= 2


def test_control_array_roundtrip_preserves_guide_random_effect_fields() -> None:
    control_fit = perturbo.ControlFit(
        beta_0=jnp.zeros((2,)),
        theta=jnp.ones((2,)),
        noise_scale=jnp.ones((2,)),
        factor_loadings=None,
        factor_scores=None,
        factor_center=None,
        pca_loadings=None,
        size_factors=jnp.zeros((3, 1)),
        losses=jnp.array([1.0], dtype=jnp.float32),
        svi_result=None,
        guide_random_effect_tau=jnp.array([0.2, 0.4], dtype=jnp.float32),
        guide_random_effect_log_tau_loc=jnp.array([-1.0, -0.5], dtype=jnp.float32),
        guide_random_effect_log_tau_scale=jnp.array([0.3, 0.6], dtype=jnp.float32),
    )
    arrays = control_fit_arrays(control_fit)
    restored = inference._restore_control_fit({k: np.asarray(v) for k, v in arrays.items() if v is not None})
    np.testing.assert_allclose(
        np.asarray(restored.guide_random_effect_tau),
        np.asarray(control_fit.guide_random_effect_tau),
    )
    np.testing.assert_allclose(
        np.asarray(restored.guide_random_effect_log_tau_loc),
        np.asarray(control_fit.guide_random_effect_log_tau_loc),
    )
    np.testing.assert_allclose(
        np.asarray(restored.guide_random_effect_log_tau_scale),
        np.asarray(control_fit.guide_random_effect_log_tau_scale),
    )


def test_dispersion_excess_roundtrip_and_low_moi_simulation_theta() -> None:
    model = perturbo.PERTURBO(_make_registered_mdata(), likelihood="negbin", effect_prior_dist="normal")
    model.control_fit = perturbo.ControlFit(
        beta_0=jnp.zeros((2,)),
        theta=jnp.array([2.0, 4.0]),
        noise_scale=jnp.ones((2,)),
        factor_loadings=None,
        factor_scores=None,
        factor_center=None,
        pca_loadings=None,
        size_factors=jnp.zeros((4, 1)),
        losses=jnp.array([]),
        svi_result=None,
    )
    model.beta_fit = perturbo.BetaFit(
        posterior_mean=jnp.zeros((2, 2)),
        posterior_scale=jnp.ones((2, 2)),
        z_values=jnp.zeros((2, 2)),
        losses=jnp.array([]),
        svi_result=None,
        dispersion_excess_inverse=jnp.array([[0.0, 0.0], [0.5, 0.25]]),
    )
    arrays = inference._beta_arrays(model.beta_fit)
    restored = inference._restore_beta_fit({key: np.asarray(value) for key, value in arrays.items() if value is not None})
    np.testing.assert_allclose(restored.dispersion_excess_inverse, model.beta_fit.dispersion_excess_inverse)
    theta = _resolve_perturbation_dispersion_theta(
        model=model,
        element_membership=np.array([[0, 0], [0, 1]], dtype=np.float32),
        guide_obs=np.array([[0, 0], [0, 1]], dtype=np.float32),
        gene_indices=np.array([0, 1], dtype=np.int32),
    )
    np.testing.assert_allclose(theta, [[2.0, 4.0], [1.0, 2.0]])
    simulated = perturbo.simulate_data_from_trained_model(
        model,
        guide_obs=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        guide_by_element=np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float32),
        element_by_gene_lfc=np.zeros((2, 2), dtype=np.float32),
        guide_efficacy=np.ones((3,), dtype=np.float32),
        cell_indices=np.array([0, 1], dtype=np.int32),
    )
    assert simulated["rna"].X.shape == (2, 2)


def test_simulation_bundle_roundtrip_supports_retraining(tmp_path) -> None:
    model = perturbo.PERTURBO(_make_registered_mdata(), likelihood="negbin", effect_prior_dist="normal")
    model.train(steps=1, batch_size=2, accelerator="cpu")
    bundle_dir = model.save(tmp_path / "bundle", overwrite=True)
    loaded = perturbo.PERTURBO.load(bundle_dir)

    guide_obs = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.float32)
    guide_by_element = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    element_by_gene_lfc = np.array([[0.4, 0.0], [0.0, 0.0]], dtype=np.float32)

    simulated = perturbo.simulate_data_from_trained_model(
        loaded,
        guide_obs=guide_obs,
        guide_by_element=guide_by_element,
        element_by_gene_lfc=element_by_gene_lfc,
        guide_efficacy=np.array([1.0, 0.8, 0.5], dtype=np.float32),
        cell_indices=np.array([0, 1], dtype=np.int32),
    )
    assert set(simulated.mod.keys()) == {"rna", "grna"}
    assert "lfc" in simulated["rna"].varm
    assert "element_targeted" in simulated["grna"].varm
    assert "_perturbo_setup" in simulated.uns

    retrained = perturbo.PERTURBO(simulated, likelihood="negbin", effect_prior_dist="normal")
    retrained.train(steps=1, batch_size=2, accelerator="cpu")
    retrained_effects = retrained.get_element_effects()
    assert len(retrained_effects) == 4


def test_simulation_resolve_size_factors_rejects_count_like_size_factor_key() -> None:
    model = perturbo.PERTURBO(
        _make_registered_mdata(size_factor_key="umi_count"),
        likelihood="negbin",
        effect_prior_dist="normal",
    )
    with pytest.raises(ValueError, match="appears to contain integer counts"):
        _resolve_size_factors(
            model,
            cell_indices=np.array([0, 1], dtype=np.int32),
            read_depth_adjust_factor=1.0,
        )


def test_guide_efficacy_requires_fit() -> None:
    model = perturbo.PERTURBO(_make_registered_mdata(), likelihood="negbin", effect_prior_dist="normal")
    with pytest.raises(RuntimeError, match="trained"):
        _ = model.guide_efficacy


def test_relative_guide_sharing_roundtrip_preserves_guide_outputs(tmp_path) -> None:
    model = perturbo.PERTURBO(
        _make_registered_mdata(),
        likelihood="negbin",
        effect_prior_dist="normal",
        guide_effect_strategy="relative",
    )
    model.train(steps=1, batch_size=2, accelerator="cpu")

    medians = model.posterior_medians()
    assert "guide_relative_efficiency" in medians

    bundle_dir = model.save(tmp_path / "relative_bundle", overwrite=True)
    assert not (bundle_dir / "guide_effects.parquet").exists()
    assert (bundle_dir / "guide_posteriors.npz").exists()

    loaded = perturbo.PERTURBO.load(bundle_dir)
    np.testing.assert_allclose(loaded.posterior_medians()["guide_effects"], medians["guide_effects"])
    np.testing.assert_allclose(
        loaded.posterior_medians()["guide_relative_efficiency"],
        medians["guide_relative_efficiency"],
    )
    expected_eff = np.asarray(medians["guide_relative_efficiency"], dtype=np.float32).reshape(
        len(model.guide_names), -1
    ).mean(axis=1)
    np.testing.assert_allclose(model.guide_efficacy, expected_eff, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(loaded.guide_efficacy, expected_eff, rtol=1e-5, atol=1e-5)


def test_relative_bundle_without_guide_posteriors_loads_with_warning(tmp_path) -> None:
    """Bundles written before guide posteriors were saved must still load.

    The July 2026 at-scale Gasperini bundle is one: relative strategy, no
    guide_posteriors.npz. Its baselines and element effects drive the Figure 6
    simulation, which supplies its own guide efficacies, so the load must not
    fail on the accessor it cannot serve.
    """
    model = perturbo.PERTURBO(
        _make_registered_mdata(),
        likelihood="negbin",
        effect_prior_dist="normal",
        guide_effect_strategy="relative",
    )
    model.train(steps=1, batch_size=2, accelerator="cpu")
    bundle_dir = model.save(tmp_path / "legacy_bundle", overwrite=True)
    # A legacy bundle has neither the guide posteriors nor the derived efficacy sidecar.
    (bundle_dir / "guide_posteriors.npz").unlink()
    (bundle_dir / "guide_efficacy.npy").unlink(missing_ok=True)

    with pytest.warns(RuntimeWarning, match="guide_relative_efficiency_mean is missing"):
        loaded = perturbo.PERTURBO.load(bundle_dir)

    assert loaded.control_fit is not None and loaded.beta_fit is not None
    np.testing.assert_allclose(loaded.control_fit.beta_0, model.control_fit.beta_0)
    assert loaded._guide_efficacy is None
    with pytest.raises(RuntimeError, match="guide_relative_efficiency_mean is missing"):
        _ = loaded.guide_efficacy


def test_train_interprets_max_epochs_as_true_epochs(monkeypatch) -> None:
    model = perturbo.PERTURBO(_make_registered_mdata(), likelihood="negbin", effect_prior_dist="normal")
    control_data = perturbo.PerTurboData(
        counts=jnp.ones((4, 2), dtype=jnp.int32),
        pert_id=jnp.zeros((4,), dtype=jnp.int32),
        pert_names=["ctrl"],
        gene_names=["g1", "g2"],
    )
    analysis_data = perturbo.PerTurboData(
        counts=jnp.ones((4, 2), dtype=jnp.int32),
        pert_id=jnp.zeros((4,), dtype=jnp.int32),
        pert_names=["ctrl"],
        gene_names=["g1", "g2"],
    )
    captured: dict[str, int] = {}

    def _fake_load_controls(*args, **kwargs):
        return control_data, None

    def _fake_load_analysis_cells(*args, **kwargs):
        return analysis_data

    def _fake_fit_control(*args, **kwargs):
        captured["control_steps"] = kwargs["num_steps"]
        return perturbo.ControlFit(
            beta_0=jnp.zeros((2,)),
            theta=jnp.ones((2,)),
            noise_scale=jnp.ones((2,)),
            factor_loadings=None,
            factor_scores=None,
            factor_center=None,
            pca_loadings=None,
            size_factors=jnp.zeros((4, 1)),
            losses=jnp.array([1.0]),
            svi_result=None,
        )

    def _fake_fit_perturbation_effects(*args, **kwargs):
        captured["beta_steps"] = kwargs["num_steps"]
        return perturbo.BetaFit(
            posterior_mean=jnp.zeros((1, 2)),
            posterior_scale=jnp.ones((1, 2)),
            z_values=jnp.zeros((1, 2)),
            losses=jnp.array([1.0]),
            svi_result=None,
        )

    monkeypatch.setattr(inference.core, "load_controls", _fake_load_controls)
    monkeypatch.setattr(inference.core, "load_analysis_cells", _fake_load_analysis_cells)
    monkeypatch.setattr(inference.core, "fit_control", _fake_fit_control)
    monkeypatch.setattr(inference.core, "fit_perturbation_effects", _fake_fit_perturbation_effects)

    model.train(max_epochs=3, batch_size=2, accelerator="cpu")

    assert captured["control_steps"] == 6
    assert captured["beta_steps"] == 6


def test_train_supports_split_epoch_schedule(monkeypatch) -> None:
    model = perturbo.PERTURBO(_make_registered_mdata(), likelihood="negbin", effect_prior_dist="normal")
    control_data = perturbo.PerTurboData(
        counts=jnp.ones((5, 2), dtype=jnp.int32),
        pert_id=jnp.zeros((5,), dtype=jnp.int32),
        pert_names=["ctrl"],
        gene_names=["g1", "g2"],
    )
    analysis_data = perturbo.PerTurboData(
        counts=jnp.ones((7, 2), dtype=jnp.int32),
        pert_id=jnp.zeros((7,), dtype=jnp.int32),
        pert_names=["ctrl"],
        gene_names=["g1", "g2"],
    )
    captured: dict[str, int] = {}

    monkeypatch.setattr(inference.core, "load_controls", lambda *args, **kwargs: (control_data, None))
    monkeypatch.setattr(inference.core, "load_analysis_cells", lambda *args, **kwargs: analysis_data)

    def _fake_fit_control(*args, **kwargs):
        captured["control_steps"] = kwargs["num_steps"]
        return perturbo.ControlFit(
            beta_0=jnp.zeros((2,)),
            theta=jnp.ones((2,)),
            noise_scale=jnp.ones((2,)),
            factor_loadings=None,
            factor_scores=None,
            factor_center=None,
            pca_loadings=None,
            size_factors=jnp.zeros((5, 1)),
            losses=jnp.array([1.0]),
            svi_result=None,
        )

    def _fake_fit_perturbation_effects(*args, **kwargs):
        captured["beta_steps"] = kwargs["num_steps"]
        return perturbo.BetaFit(
            posterior_mean=jnp.zeros((1, 2)),
            posterior_scale=jnp.ones((1, 2)),
            z_values=jnp.zeros((1, 2)),
            losses=jnp.array([1.0]),
            svi_result=None,
        )

    monkeypatch.setattr(inference.core, "fit_control", _fake_fit_control)
    monkeypatch.setattr(inference.core, "fit_perturbation_effects", _fake_fit_perturbation_effects)

    model.train(control_epochs=2, beta_epochs=3, batch_size=3, accelerator="cpu")

    assert captured["control_steps"] == 4
    assert captured["beta_steps"] == 9


def test_train_supports_split_step_schedule(monkeypatch) -> None:
    model = perturbo.PERTURBO(_make_registered_mdata(), likelihood="negbin", effect_prior_dist="normal")
    control_data = perturbo.PerTurboData(
        counts=jnp.ones((5, 2), dtype=jnp.int32),
        pert_id=jnp.zeros((5,), dtype=jnp.int32),
        pert_names=["ctrl"],
        gene_names=["g1", "g2"],
    )
    analysis_data = perturbo.PerTurboData(
        counts=jnp.ones((7, 2), dtype=jnp.int32),
        pert_id=jnp.zeros((7,), dtype=jnp.int32),
        pert_names=["ctrl"],
        gene_names=["g1", "g2"],
    )
    captured: dict[str, int] = {}

    def _fake_fit_control(*args, **kwargs):
        captured["control_steps"] = kwargs["num_steps"]
        return perturbo.ControlFit(
            beta_0=jnp.zeros((2,)),
            theta=jnp.ones((2,)),
            noise_scale=jnp.ones((2,)),
            factor_loadings=None,
            factor_scores=None,
            factor_center=None,
            pca_loadings=None,
            size_factors=jnp.zeros((5, 1)),
            losses=jnp.array([1.0]),
            svi_result=None,
        )

    def _fake_fit_perturbation_effects(*args, **kwargs):
        captured["beta_steps"] = kwargs["num_steps"]
        return perturbo.BetaFit(
            posterior_mean=jnp.zeros((1, 2)),
            posterior_scale=jnp.ones((1, 2)),
            z_values=jnp.zeros((1, 2)),
            losses=jnp.array([1.0]),
            svi_result=None,
        )

    monkeypatch.setattr(inference.core, "load_controls", lambda *args, **kwargs: (control_data, None))
    monkeypatch.setattr(inference.core, "load_analysis_cells", lambda *args, **kwargs: analysis_data)
    monkeypatch.setattr(inference.core, "fit_control", _fake_fit_control)
    monkeypatch.setattr(inference.core, "fit_perturbation_effects", _fake_fit_perturbation_effects)

    model.train(control_steps=4, beta_steps=9, batch_size=3, accelerator="cpu")

    assert captured["control_steps"] == 4
    assert captured["beta_steps"] == 9


def test_train_rejects_mixed_step_and_epoch_schedule() -> None:
    model = perturbo.PERTURBO(_make_registered_mdata(), likelihood="negbin", effect_prior_dist="normal")

    with pytest.raises(ValueError, match="cannot be mixed"):
        model.train(max_epochs=2, steps=3, accelerator="cpu")
