from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pandas as pd

import perturbo.api as api
from perturbo.api import BaselinePosteriorSummary, BetaFit, ControlFit, PerTurboData


def _dummy_control_fit(n_obs: int, n_genes: int = 2) -> ControlFit:
    return ControlFit(
        beta_0=jnp.zeros((n_genes,)),
        theta=jnp.ones((n_genes,)),
        noise_scale=jnp.ones((n_genes,)),
        factor_loadings=None,
        factor_scores=None,
        factor_center=None,
        pca_loadings=None,
        size_factors=jnp.zeros((n_obs, 1)),
        losses=jnp.array([1.0]),
        svi_result=None,
        baseline_posterior=BaselinePosteriorSummary(
            beta_0_loc=jnp.zeros((n_genes,)),
            beta_0_scale=jnp.ones((n_genes,)),
            theta_log_loc=jnp.zeros((n_genes,)),
            theta_log_scale=jnp.ones((n_genes,)),
        ),
        guide_random_effect_tau=jnp.linspace(0.2, 0.4, n_genes, dtype=jnp.float32),
        guide_random_effect_log_tau_loc=jnp.linspace(-1.0, -0.5, n_genes, dtype=jnp.float32),
        guide_random_effect_log_tau_scale=jnp.linspace(0.3, 0.6, n_genes, dtype=jnp.float32),
    )


def test_cli_writes_standard_outputs_relative(monkeypatch, tmp_path) -> None:
    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame(index=["c0", "c1"])
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = 2
            self.X = np.array([[1, 0], [0, 1]], dtype=np.int32)

    data = PerTurboData(
        counts=jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),
        pert_id=jnp.array([0, 1], dtype=jnp.int32),
        pert_names=["ntc0", "pertA"],
        gene_names=["g1", "g2"],
        guide_matrix=jnp.array([[1, 0], [0, 1]], dtype=jnp.float32),
        guide_to_element=jnp.array([[1, 0], [0, 1]], dtype=jnp.float32),
        guide_names=["guide_0", "guide_1"],
    )

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "_validate_cli_input_keys", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_resolve_adata", lambda *args, **kwargs: _FakeAdata())
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "load_analysis_cells", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "fit_control", lambda *args, **kwargs: _dummy_control_fit(2))
    monkeypatch.setattr(
        api,
        "fit_perturbation_effects",
        lambda *args, **kwargs: BetaFit(
            posterior_mean=jnp.array([[0.1, -0.2], [0.3, 0.4]], dtype=jnp.float32),
            posterior_scale=jnp.array([[0.2, 0.3], [0.4, 0.5]], dtype=jnp.float32),
            z_values=jnp.array([[0.5, -0.7], [0.75, 0.8]], dtype=jnp.float32),
            losses=jnp.array([1.0]),
            svi_result=None,
            guide_relative_efficiency_mean=jnp.array([[0.7, 0.9], [0.8, 0.85]], dtype=jnp.float32),
        ),
    )

    api.main(
        [
            "--input",
            "dummy.h5mu",
            "--out-dir",
            str(tmp_path),
            "--modality-key",
            "rna",
            "--perturbation-modality-key",
            "grna",
            "--perturbation-element-varm-key",
            "element_targeted",
            "--guide-effect-strategy",
            "relative",
        ]
    )

    element_path = tmp_path / "element_effects.parquet"
    guide_eff_path = tmp_path / "guide_efficiency.parquet"
    control_fit_path = tmp_path / "control_fit.npz"
    assert element_path.exists()
    assert guide_eff_path.exists()
    assert control_fit_path.exists()
    for old_output_name in [
        "posterior_summary_long.parquet",
        "posterior_mean.parquet",
        "posterior_scale.parquet",
        "posterior_prob.parquet",
        "guide_effects.parquet",
        "guide_effects.csv",
    ]:
        assert not (tmp_path / old_output_name).exists()

    element_df = pd.read_parquet(element_path)
    assert list(element_df.columns) == [
        "method",
        "element",
        "gene",
        "posterior_mean",
        "posterior_scale",
        "z_value",
        "posterior_prob",
        "empirical_p_value",
    ]
    assert "scenario_id" not in element_df.columns
    assert str(element_df["posterior_mean"].dtype) == "float32"

    with np.load(control_fit_path, allow_pickle=False) as control_bundle:
        assert "beta_0" in control_bundle.files
        assert "theta" in control_bundle.files
        assert "baseline_beta_0_loc" in control_bundle.files
        assert "baseline_theta_log_scale" in control_bundle.files
        np.testing.assert_allclose(control_bundle["guide_random_effect_tau"], np.array([0.2, 0.4], dtype=np.float32))


def test_cli_empirical_p_values_use_inferred_high_moi_control_elements(monkeypatch, tmp_path) -> None:
    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame(index=["c0", "c1"])
            self.var = pd.DataFrame(index=["g1", "g2", "g3", "g4"])
            self.n_obs = 2
            self.X = np.array([[1, 0, 2, 1], [0, 1, 1, 2]], dtype=np.int32)

    data = PerTurboData(
        counts=jnp.array([[1, 0, 2, 1], [0, 1, 1, 2]], dtype=jnp.int32),
        pert_id=jnp.array([[1, 0], [0, 1]], dtype=jnp.int8),
        pert_names=["random_1", "targetA"],
        gene_names=["g1", "g2", "g3", "g4"],
        guide_matrix=jnp.array([[1, 0], [0, 1]], dtype=jnp.float32),
        guide_to_element=jnp.array([[1, 0], [0, 1]], dtype=jnp.float32),
        guide_names=["guide_random", "guide_target"],
    )

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "_validate_cli_input_keys", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_resolve_adata", lambda *args, **kwargs: _FakeAdata())
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)

    def _fake_load_controls(*args, **kwargs):
        assert kwargs["infer_control_guides"] is True
        return data

    monkeypatch.setattr(api, "load_controls", _fake_load_controls)
    monkeypatch.setattr(api, "load_analysis_cells", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "fit_control", lambda *args, **kwargs: _dummy_control_fit(2, n_genes=4))
    monkeypatch.setattr(
        api,
        "fit_perturbation_effects",
        lambda *args, **kwargs: BetaFit(
            posterior_mean=jnp.array(
                [[0.05, -0.05, 0.10, -0.10], [0.30, 0.40, -0.50, 0.60]],
                dtype=jnp.float32,
            ),
            posterior_scale=jnp.full((2, 4), 0.5, dtype=jnp.float32),
            z_values=jnp.array(
                [[0.10, -0.10, 0.20, -0.20], [0.60, 0.80, -1.00, 1.20]],
                dtype=jnp.float32,
            ),
            losses=jnp.array([1.0]),
            svi_result=None,
        ),
    )

    api.main(
        [
            "--input",
            "dummy.h5mu",
            "--out-dir",
            str(tmp_path),
            "--modality-key",
            "rna",
            "--perturbation-modality-key",
            "grna",
            "--perturbation-element-varm-key",
            "element_targeted",
            "--guide-random-effects",
        ]
    )

    element_df = pd.read_parquet(tmp_path / "element_effects.parquet")
    assert element_df["empirical_p_value"].notna().all()


def test_cli_does_not_write_guide_efficiency_for_shared(monkeypatch, tmp_path) -> None:
    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame({"perturbation": ["ctrl", "pert"]}, index=["c0", "c1"])
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = 2
            self.X = np.array([[1, 0], [0, 1]], dtype=np.int32)

    data = PerTurboData(
        counts=jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),
        pert_id=jnp.array([0, 1], dtype=jnp.int32),
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2"],
    )

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: _FakeAdata())
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "load_analysis_cells", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "fit_control", lambda *args, **kwargs: _dummy_control_fit(2))
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        api,
        "fit_perturbation_effects",
        lambda *args, **kwargs: BetaFit(
            posterior_mean=jnp.array([[0.1, -0.2], [0.3, 0.4]], dtype=jnp.float32),
            posterior_scale=jnp.array([[0.2, 0.3], [0.4, 0.5]], dtype=jnp.float32),
            z_values=jnp.array([[0.5, -0.7], [0.75, 0.8]], dtype=jnp.float32),
            losses=jnp.array([1.0]),
            svi_result=None,
        ),
    )

    api.main(
        [
            "--input",
            "dummy.h5ad",
            "--out-dir",
            str(tmp_path),
            "--perturbation-key",
            "perturbation",
            "--control-substring",
            "ctrl",
        ]
    )

    assert (tmp_path / "element_effects.parquet").exists()
    assert not (tmp_path / "guide_efficiency.parquet").exists()


def test_cli_saves_model_params_by_default_and_supports_opt_out(monkeypatch, tmp_path) -> None:
    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame(
                {"perturbation": ["ctrl", "pert"], "umi_count": [10, 12]},
                index=["c0", "c1"],
            )
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = 2
            self.n_vars = 2
            self.X = np.array([[1, 0], [0, 1]], dtype=np.int32)

    data = PerTurboData(
        counts=jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),
        pert_id=jnp.array([0, 1], dtype=jnp.int32),
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2"],
        library_size_center_log_mean=float(np.mean(np.log1p([10]))),
    )

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: _FakeAdata())
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "load_analysis_cells", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "fit_control", lambda *args, **kwargs: _dummy_control_fit(2))
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        api,
        "fit_perturbation_effects",
        lambda *args, **kwargs: BetaFit(
            posterior_mean=jnp.array([[0.1, -0.2], [0.3, 0.4]], dtype=jnp.float32),
            posterior_scale=jnp.array([[0.2, 0.3], [0.4, 0.5]], dtype=jnp.float32),
            z_values=jnp.array([[0.5, -0.7], [0.75, 0.8]], dtype=jnp.float32),
            losses=jnp.array([1.0]),
            svi_result=None,
        ),
    )

    api.main(
        [
            "--input",
            "dummy.h5ad",
            "--out-dir",
            str(tmp_path),
            "--perturbation-key",
            "perturbation",
            "--control-substring",
            "ctrl",
            "--library-size-key",
            "umi_count",
            "--size-factor-mode",
            "observed",
            # This test stubs the loader, so the fabricated data carries no size
            # factors and the conditional randomization test cannot run on it. The
            # contract under test is which files a run writes.
            "--no-crt",
        ]
    )

    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "control_fit.npz").exists()
    assert (tmp_path / "beta_fit.npz").exists()
    assert (tmp_path / "guide_efficacy.npy").exists()
    assert (tmp_path / "element_effects.parquet").exists()
    assert not (tmp_path / "mdata.h5mu").exists()
    assert not (tmp_path / "element_effects.csv").exists()
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["bundle_format"] == "light_model_params"
    assert metadata["perturbation_key"] == "perturbation"
    assert metadata["setup"]["library_size_key"] == "umi_count"
    assert metadata["perturbation_names"] == ["ctrl", "pert"]

    opt_out_dir = tmp_path / "opt_out"
    api.main(
        [
            "--input",
            "dummy.h5ad",
            "--out-dir",
            str(opt_out_dir),
            "--perturbation-key",
            "perturbation",
            "--control-substring",
            "ctrl",
            "--library-size-key",
            "umi_count",
            "--size-factor-mode",
            "observed",
            # This test stubs the loader, so the fabricated data carries no size
            # factors and the conditional randomization test cannot run on it. The
            # contract under test is which files a run writes.
            "--no-crt",
            "--no-save-model-params",
        ]
    )
    assert (opt_out_dir / "element_effects.parquet").exists()
    assert not (opt_out_dir / "metadata.json").exists()
    assert not (opt_out_dir / "beta_fit.npz").exists()
    assert not (opt_out_dir / "guide_efficacy.npy").exists()
