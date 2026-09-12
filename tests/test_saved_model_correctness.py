from __future__ import annotations

import json
from types import SimpleNamespace

import anndata as ad
import jax.numpy as jnp
import mudata as md
import numpy as np
import pandas as pd

import perturbo.api as api
import perturbo.cli as cli_module
import perturbo.simulation.fitted as fitted_simulation
from perturbo.inference import PerTurboModel
from perturbo.io import MuDataSetup
from perturbo.simulation.fitted import _resolve_size_factors


def test_bundle_offsets_slice_original_rows_before_reading_counts() -> None:
    counts = np.arange(24, dtype=np.int32).reshape(6, 4)
    selected_rows = np.array([1, 3, 4], dtype=np.int64)
    reads: list[np.ndarray] = []

    class SparseLikeBlock:
        def __init__(self, values):
            self.values = values

        def sum(self, axis, dtype=None):
            return self.values.sum(axis=axis, dtype=dtype, keepdims=True)

        def toarray(self):
            raise AssertionError("non-winsorized bundle offsets must not densify sparse blocks")

    class GuardedBackedMatrix:
        def __getitem__(self, key):
            rows, columns = key
            assert columns == slice(None)
            rows = np.asarray(rows, dtype=np.int64)
            reads.append(rows.copy())
            return SparseLikeBlock(counts[rows])

    adata = SimpleNamespace(
        n_obs=counts.shape[0],
        obs=pd.DataFrame({"saved": np.linspace(-0.5, 0.5, counts.shape[0])}),
        X=GuardedBackedMatrix(),
    )
    center = 2.5
    offsets, provenance = cli_module._bundle_size_factors(
        adata,
        selected_row_indices=selected_rows,
        size_factor_mode="observed",
        size_factor_key=None,
        library_size_key=None,
        library_size_center_log_mean=center,
        gene_clip_thresholds=None,
        winsorize_gene_expression=False,
    )

    assert provenance == "analyzed_count_total"
    np.testing.assert_array_equal(reads, [selected_rows])
    np.testing.assert_allclose(
        offsets,
        (np.log1p(counts[selected_rows].sum(axis=1)) - center)[:, None],
        rtol=1e-6,
    )

    saved, provenance = cli_module._bundle_size_factors(
        adata,
        selected_row_indices=selected_rows,
        size_factor_mode="observed",
        size_factor_key="saved",
        library_size_key=None,
        library_size_center_log_mean=None,
        gene_clip_thresholds=None,
        winsorize_gene_expression=False,
    )
    assert provenance == "obs_size_factor"
    np.testing.assert_allclose(saved[:, 0], adata.obs["saved"].to_numpy()[selected_rows])


def _write_guide_fixture(path, *, constant_depth: bool = False) -> None:
    obs = pd.DataFrame(index=[f"c{i}" for i in range(6)])
    counts = (
        np.full((6, 2), 2, dtype=np.int32)
        if constant_depth
        else np.array([[2, 1], [2, 2], [1, 3], [2, 3], [1, 2], [1, 1]], dtype=np.int32)
    )
    rna = ad.AnnData(
        counts,
        obs=obs.copy(),
        var=pd.DataFrame(index=["gene1", "gene2"]),
    )
    grna = ad.AnnData(
        np.array(
            [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
            dtype=np.float32,
        ),
        obs=obs.copy(),
        var=pd.DataFrame(index=["ctrl", "guide_a", "guide_b"]),
    )
    grna.varm["mapping"] = pd.DataFrame(
        np.eye(3, dtype=np.float32),
        index=grna.var_names,
        columns=["ctrl", "element_a", "element_b"],
    )
    md.MuData({"rna": rna, "grna": grna}).write_h5mu(path)


def _stub_cli_fits(monkeypatch, captured_size_factors: list[np.ndarray]) -> None:
    def control_fit(data, **kwargs):
        del kwargs
        return api.ControlFit(
            beta_0=jnp.zeros(2),
            theta=jnp.ones(2),
            noise_scale=jnp.ones(2),
            factor_loadings=None,
            factor_scores=None,
            factor_center=None,
            pca_loadings=None,
            size_factors=data.size_factors,
            losses=jnp.array([1.0]),
            svi_result=None,
        )

    def beta_fit(data, control, **kwargs):
        del control, kwargs
        captured_size_factors.append(np.asarray(data.size_factors))
        guide_names = list(data.guide_names or [])
        guide_ids = np.asarray(
            [["ctrl", "guide_a", "guide_b"].index(name) for name in guide_names],
            dtype=np.float32,
        )[:, None]
        guide_values = np.broadcast_to(guide_ids + 0.1, (len(guide_names), 2)).copy()
        relative = np.broadcast_to(guide_ids * 0.1 + 0.2, (len(guide_names), 2)).copy()
        return api.BetaFit(
            posterior_mean=jnp.full((len(data.pert_names), 2), 0.4),
            posterior_scale=jnp.ones((len(data.pert_names), 2)),
            z_values=jnp.full((len(data.pert_names), 2), 0.4),
            losses=jnp.array([1.0]),
            svi_result=None,
            guide_effect_mean=jnp.asarray(guide_values),
            guide_effect_scale=jnp.ones_like(jnp.asarray(guide_values)),
            guide_effect_z_values=jnp.asarray(guide_values),
            guide_relative_efficiency_mean=jnp.asarray(relative),
            guide_relative_efficiency_scale=jnp.full_like(jnp.asarray(relative), 0.05),
            guide_offset_mean=jnp.asarray(guide_values + 1.0),
            guide_offset_scale=jnp.full_like(jnp.asarray(guide_values), 0.2),
            guide_dispersion_excess_inverse=jnp.asarray(guide_values + 2.0),
        )

    monkeypatch.setattr(api, "fit_control", control_fit)
    monkeypatch.setattr(api, "fit_perturbation_effects", beta_fit)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)


def test_chunked_cli_bundle_preserves_global_guide_arrays_and_derived_offsets(monkeypatch, tmp_path) -> None:
    source = tmp_path / "input.h5mu"
    out = tmp_path / "bundle"
    _write_guide_fixture(source)
    captured: list[np.ndarray] = []
    _stub_cli_fits(monkeypatch, captured)

    api.main(
        [
            "--input",
            str(source),
            "--out-dir",
            str(out),
            "--modality-key",
            "rna",
            "--perturbation-modality-key",
            "grna",
            "--perturbation-element-varm-key",
            "mapping",
            "--control-substring",
            "ctrl",
            "--guide-effect-strategy",
            "relative",
            "--perturbation-chunk-size",
            "1",
            "--no-crt",
        ]
    )

    loaded = PerTurboModel.load(out)
    assert loaded.beta_fit is not None
    expected_guides = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]], dtype=np.float32)
    np.testing.assert_allclose(loaded.beta_fit.guide_effect_mean, expected_guides)
    np.testing.assert_allclose(loaded.beta_fit.guide_offset_mean, expected_guides + 1.0)
    np.testing.assert_allclose(loaded.beta_fit.guide_dispersion_excess_inverse, expected_guides + 2.0)
    np.testing.assert_allclose(loaded.guide_efficacy, [0.2, 0.3, 0.4])

    fitted_offsets = np.concatenate(captured, axis=0)
    simulated_offsets = _resolve_size_factors(loaded, np.arange(6), 1.0)
    np.testing.assert_allclose(simulated_offsets, fitted_offsets)
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["size_factor_mode"] == "observed"
    assert metadata["size_factor_provenance"] == "analyzed_count_total"
    assert metadata["size_factors_file"] == "size_factors.npy"


def test_cli_bundle_keeps_explicit_none_offsets_zero(monkeypatch, tmp_path) -> None:
    source = tmp_path / "input.h5mu"
    out = tmp_path / "bundle"
    _write_guide_fixture(source)
    captured: list[np.ndarray] = []
    _stub_cli_fits(monkeypatch, captured)

    api.main(
        [
            "--input",
            str(source),
            "--out-dir",
            str(out),
            "--modality-key",
            "rna",
            "--perturbation-modality-key",
            "grna",
            "--perturbation-element-varm-key",
            "mapping",
            "--control-substring",
            "ctrl",
            "--guide-effect-strategy",
            "relative",
            "--size-factor-mode",
            "none",
            "--no-crt",
        ]
    )

    loaded = PerTurboModel.load(out)
    np.testing.assert_array_equal(captured[0], np.zeros((6, 1), dtype=np.float32))
    np.testing.assert_array_equal(
        _resolve_size_factors(loaded, np.arange(6), 1.0),
        np.zeros((6, 1), dtype=np.float32),
    )
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["size_factor_mode"] == "none"
    assert metadata["size_factor_provenance"] == "fixed_zero"


def test_saved_constant_observed_offsets_are_not_rejected_as_raw_counts(monkeypatch, tmp_path) -> None:
    source = tmp_path / "input.h5mu"
    out = tmp_path / "bundle"
    _write_guide_fixture(source, constant_depth=True)
    captured: list[np.ndarray] = []
    _stub_cli_fits(monkeypatch, captured)

    api.main(
        [
            "--input",
            str(source),
            "--out-dir",
            str(out),
            "--modality-key",
            "rna",
            "--perturbation-modality-key",
            "grna",
            "--perturbation-element-varm-key",
            "mapping",
            "--control-substring",
            "ctrl",
            "--guide-effect-strategy",
            "relative",
            "--no-crt",
        ]
    )

    loaded = PerTurboModel.load(out)
    np.testing.assert_array_equal(
        _resolve_size_factors(loaded, np.arange(6), 1.0),
        np.zeros((6, 1), dtype=np.float32),
    )


def test_mixture_simulation_excludes_element_effect_from_outlier_mean(monkeypatch, tmp_path) -> None:
    source = tmp_path / "input.h5mu"
    _write_guide_fixture(source)
    mdata = md.read_h5mu(source)
    setup = MuDataSetup(
        rna_modality="rna",
        perturbation_modality="grna",
        guide_by_element_key="mapping",
        size_factor_mode="none",
    )
    model = PerTurboModel(mdata, setup=setup, likelihood="mixture_nb", effect_prior_dist="normal")
    model.control_fit = api.ControlFit(
        beta_0=jnp.array([1.0, 1.0]),
        theta=jnp.ones(2),
        noise_scale=jnp.ones(2),
        factor_loadings=None,
        factor_scores=None,
        factor_center=None,
        pca_loadings=None,
        size_factors=jnp.zeros((6, 1)),
        losses=jnp.array([]),
        svi_result=None,
        pi_outlier=jnp.ones(2),
        theta_outlier=jnp.ones(2),
        outlier_mean_shift=jnp.zeros(2),
    )
    model.beta_fit = api.BetaFit(
        posterior_mean=jnp.zeros((3, 2)),
        posterior_scale=jnp.ones((3, 2)),
        z_values=jnp.zeros((3, 2)),
        losses=jnp.array([]),
        svi_result=None,
    )
    captured: dict[str, np.ndarray] = {}

    def capture_counts(*, mu, outlier_mu, **kwargs):
        del kwargs
        captured["mu"] = np.asarray(mu)
        captured["outlier_mu"] = np.asarray(outlier_mu)
        return np.zeros_like(mu, dtype=np.int32)

    monkeypatch.setattr(fitted_simulation, "_sample_counts", capture_counts)
    fitted_simulation.simulate_data_from_trained_model(
        model,
        guide_obs=np.array([[1, 0, 0]], dtype=np.float32),
        guide_by_element=np.eye(3, dtype=np.float32),
        element_by_gene_lfc=np.array([[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        guide_efficacy=np.ones(3, dtype=np.float32),
        cell_indices=np.array([0], dtype=np.int32),
    )

    np.testing.assert_allclose(captured["mu"], [[1.0, 3.0]])
    np.testing.assert_allclose(captured["outlier_mu"], [[1.0, 1.0]])


def test_shared_simulation_counts_duplicate_guides_once_with_unit_efficacy(monkeypatch, tmp_path) -> None:
    source = tmp_path / "input.h5mu"
    _write_guide_fixture(source)
    model = PerTurboModel(
        md.read_h5mu(source),
        setup=MuDataSetup(
            rna_modality="rna",
            perturbation_modality="grna",
            guide_by_element_key="mapping",
            size_factor_mode="none",
        ),
        likelihood="negbin",
        effect_prior_dist="normal",
        guide_effect_strategy="shared",
    )
    model.control_fit = api.ControlFit(
        beta_0=jnp.zeros(2),
        theta=jnp.ones(2),
        noise_scale=jnp.ones(2),
        factor_loadings=None,
        factor_scores=None,
        factor_center=None,
        pca_loadings=None,
        size_factors=jnp.zeros((6, 1)),
        losses=jnp.array([]),
        svi_result=None,
    )
    model.beta_fit = api.BetaFit(
        posterior_mean=jnp.zeros((2, 2)),
        posterior_scale=jnp.ones((2, 2)),
        z_values=jnp.zeros((2, 2)),
        losses=jnp.array([]),
        svi_result=None,
    )
    captured: list[np.ndarray] = []

    def capture_counts(*, mu, **kwargs):
        del kwargs
        captured.append(np.asarray(mu))
        return np.zeros_like(mu, dtype=np.int32)

    monkeypatch.setattr(fitted_simulation, "_sample_counts", capture_counts)
    common = {
        "model": model,
        "guide_obs": np.array([[0, 1, 1]], dtype=np.float32),
        "guide_by_element": np.array([[0, 1], [1, 0], [1, 0]], dtype=np.float32),
        "element_by_gene_lfc": np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        "cell_indices": np.array([0], dtype=np.int32),
    }
    fitted_simulation.simulate_data_from_trained_model(
        **common,
        guide_efficacy=np.ones(3, dtype=np.float32),
    )
    fitted_simulation.simulate_data_from_trained_model(
        **common,
        guide_efficacy=np.array([1.0, 0.5, 0.25], dtype=np.float32),
    )

    np.testing.assert_allclose(captured[0], [[2.0, 0.0]])
    np.testing.assert_allclose(captured[1], [[1.5, 0.0]])
