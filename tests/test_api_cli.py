"""CLI wiring tests for PerTurbo API."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import perturbo.api as api
from perturbo.api import BaselinePosteriorSummary, BetaFit, ControlFit, PerTurboData, CovariateTransformState


@pytest.fixture(autouse=True)
def _wiring_tests_do_not_run_the_crt(monkeypatch):
    """These tests stub the loader and the fitters with sentinel objects to check how
    the CLI wires its flags. The conditional randomization test is on by default and
    would reach code the sentinels cannot satisfy, so it steps aside here, the way
    it does for any unsupported configuration."""
    import perturbo.cli as cli_module

    monkeypatch.setattr(
        cli_module, "_crt_configuration_problem", lambda args, size_factor_mode: "wiring test: loader and fitters are stubs"
    )



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
    )


def _dummy_beta_fit(n_perts: int, n_genes: int = 2) -> BetaFit:
    return BetaFit(
        posterior_mean=jnp.zeros((n_perts, n_genes)),
        posterior_scale=jnp.ones((n_perts, n_genes)),
        z_values=jnp.zeros((n_perts, n_genes)),
        losses=jnp.array([1.0]),
        svi_result=None,
    )


def test_main_wires_uncertainty_flags(monkeypatch, tmp_path) -> None:
    counts = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    pert_id = jnp.array([0, 1], dtype=jnp.int32)
    data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2"],
    )

    captured: dict[str, object] = {}

    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame({"perturbation": ["ctrl", "pert"]}, index=["c0", "c1"])
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = len(self.obs)
            self.X = np.array([[0, 1], [1, 0]], dtype=np.int32)

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: _FakeAdata())
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "load_analysis_cells", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)

    def _fake_fit_control(*args, **kwargs):
        cfg = kwargs["svi_config"]
        captured["control_elbo"] = cfg.elbo
        captured["control_num_particles"] = cfg.num_particles
        return ControlFit(
            beta_0=jnp.zeros((2,)),
            theta=jnp.ones((2,)),
            noise_scale=jnp.ones((2,)),
            factor_loadings=None,
            factor_scores=None,
            factor_center=None,
            pca_loadings=None,
            size_factors=jnp.zeros((2, 1)),
            losses=jnp.array([1.0]),
            svi_result=None,
            baseline_posterior=BaselinePosteriorSummary(
                beta_0_loc=jnp.zeros((2,)),
                beta_0_scale=jnp.ones((2,)),
                theta_log_loc=jnp.zeros((2,)),
                theta_log_scale=jnp.ones((2,)),
            ),
        )

    def _fake_fit_perturbation_effects(*args, **kwargs):
        captured["propagate"] = kwargs["propagate_baseline_uncertainty"]
        cfg = kwargs["svi_config"]
        captured["beta_elbo"] = cfg.elbo
        captured["beta_num_particles"] = cfg.num_particles
        return BetaFit(
            posterior_mean=jnp.zeros((2, 2)),
            posterior_scale=jnp.ones((2, 2)),
            z_values=jnp.zeros((2, 2)),
            losses=jnp.array([1.0]),
            svi_result=None,
        )

    def _fake_summarize(*args, **kwargs):
        frame = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=["ctrl", "pert"], columns=["g1", "g2"])
        return {
            "posterior_mean": frame,
            "posterior_scale": frame,
            "posterior_prob": frame,
        }

    monkeypatch.setattr(api, "fit_control", _fake_fit_control)
    monkeypatch.setattr(api, "fit_perturbation_effects", _fake_fit_perturbation_effects)
    monkeypatch.setattr(api, "summarize_betas", _fake_summarize)

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
            "--num-particles",
            "1",
            "--propagate-baseline-uncertainty",
        ]
    )

    assert captured["control_num_particles"] == 1
    assert captured["beta_num_particles"] == 1
    assert captured["control_elbo"] == "meanfield"
    assert captured["beta_elbo"] == "meanfield"
    assert captured["propagate"] is True


def test_fit_from_path_builds_cli_equivalent_replogle_command(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_main(argv):
        captured["argv"] = list(argv)

    monkeypatch.setattr(api, "main", _fake_main)

    api.fit_from_path(
        "/scratch/l/ljb80/K562_gwps_raw_singlecell_01.h5ad",
        "/scratch/l/ljb80/replogle_gwps_guide_recalibration_grid/nb_clipped/guide_re_on",
        perturbation_key="gene_transcript",
        control_substring="non-targeting",
        batch_covariate="gem_group",
        library_size_key="UMI_count",
        size_factor_mode="observed",
        likelihood="negbin",
        clip_gene_expression_percentile=99,
        winsorize_gene_expression_outliers=True,
        num_steps_control=2000,
        num_steps_betas=2000,
        step_size=0.01,
        max_control_cells=10000,
        max_chunk_size=10000,
        backed=True,
        device="cuda:0",
        guide_random_effects=True,
    )

    argv = captured["argv"]
    assert argv[:4] == [
        "--input",
        "/scratch/l/ljb80/K562_gwps_raw_singlecell_01.h5ad",
        "--out-dir",
        "/scratch/l/ljb80/replogle_gwps_guide_recalibration_grid/nb_clipped/guide_re_on",
    ]
    expected_pairs = {
        "--perturbation-key": "gene_transcript",
        "--control-substring": "non-targeting",
        "--batch-covariate": "gem_group",
        "--library-size-key": "UMI_count",
        "--size-factor-mode": "observed",
        "--likelihood": "negbin",
        "--clip-gene-expression-percentile": "99",
        "--num-steps-control": "2000",
        "--num-steps-betas": "2000",
        "--step-size": "0.01",
        "--max-control-cells": "10000",
        "--max-chunk-size": "10000",
        "--device": "cuda:0",
    }
    for flag, value in expected_pairs.items():
        idx = argv.index(flag)
        assert argv[idx + 1] == value
    for flag in [
        "--winsorize-gene-expression-outliers",
        "--backed",
        "--guide-random-effects",
        "--save-model-params",
    ]:
        assert flag in argv

    api.fit_from_path(
        "dummy.h5ad",
        "out",
        perturbation_key="perturbation",
        control_substring="ctrl",
        save_model_params=False,
    )
    assert "--no-save-model-params" in captured["argv"]
    assert "--save-model-params" not in captured["argv"]


def test_main_wires_guide_random_effects_flag(monkeypatch, tmp_path) -> None:
    counts = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    pert_id = jnp.array([0, 1], dtype=jnp.int32)
    data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2"],
    )
    captured: dict[str, object] = {}

    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame({"perturbation": ["ctrl", "pert"]}, index=["c0", "c1"])
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = len(self.obs)
            self.X = np.array([[0, 1], [1, 0]], dtype=np.int32)

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: _FakeAdata())
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "load_analysis_cells", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)

    def _fake_fit_control(*args, **kwargs):
        captured["control_guide_random_effects"] = kwargs.get("guide_random_effects")
        return _dummy_control_fit(n_obs=2)

    def _fake_fit_perturbation_effects(*args, **kwargs):
        captured["beta_guide_random_effects"] = kwargs.get("guide_random_effects")
        return _dummy_beta_fit(n_perts=2)

    def _fake_summarize(*args, **kwargs):
        frame = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=["ctrl", "pert"], columns=["g1", "g2"])
        return {
            "posterior_mean": frame,
            "posterior_scale": frame,
            "posterior_prob": frame,
        }

    monkeypatch.setattr(api, "fit_control", _fake_fit_control)
    monkeypatch.setattr(api, "fit_perturbation_effects", _fake_fit_perturbation_effects)
    monkeypatch.setattr(api, "summarize_betas", _fake_summarize)

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
            "--guide-random-effects",
        ]
    )

    assert captured["control_guide_random_effects"] is True
    assert captured["beta_guide_random_effects"] is True


def test_main_rejects_high_moi_guide_random_effects_when_control_guides_cannot_be_inferred(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "_validate_cli_input_keys", lambda *args, **kwargs: None)

    def _fake_load_controls(*args, **kwargs):
        assert kwargs["infer_control_guides"] is True
        raise ValueError("Could not infer control guides")

    monkeypatch.setattr(api, "load_controls", _fake_load_controls)

    with pytest.raises(ValueError, match="Could not infer control guides"):
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
                "--guide-random-effects",
            ]
        )


def test_main_rejects_guide_strategy_without_element_mapping(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: object())

    with pytest.raises(ValueError, match="Guide-sharing modes require --perturbation-element-varm-key"):
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
                "--guide-effect-strategy",
                "relative",
            ]
        )


def test_main_rejects_absolute_mode_for_non_negbin(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: object())

    with pytest.raises(ValueError, match="only compatible with negative-binomial likelihoods"):
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
                "--guide-activity-mode",
                "absolute",
                "--likelihood",
                "lognormal_nb",
            ]
        )


def test_main_wires_guide_strategy_flags(monkeypatch, tmp_path) -> None:
    counts = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    pert_id = jnp.array([0, 1], dtype=jnp.int32)
    data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["element_a", "ntc0"],
        gene_names=["g1", "g2"],
    )

    captured: dict[str, object] = {}

    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame(index=["c0", "c1"])
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = 2
            self.X = np.array([[0, 1], [1, 0]], dtype=np.int32)

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "_validate_cli_input_keys", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_resolve_adata", lambda *args, **kwargs: _FakeAdata())
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)

    def _fake_load_analysis_cells(*args, **kwargs):
        captured["retain_guide_structure"] = kwargs["retain_guide_structure"]
        return data

    def _fake_fit_control(*args, **kwargs):
        return ControlFit(
            beta_0=jnp.zeros((2,)),
            theta=jnp.ones((2,)),
            noise_scale=jnp.ones((2,)),
            factor_loadings=None,
            factor_scores=None,
            factor_center=None,
            pca_loadings=None,
            size_factors=jnp.zeros((2, 1)),
            losses=jnp.array([1.0]),
            svi_result=None,
            baseline_posterior=BaselinePosteriorSummary(
                beta_0_loc=jnp.zeros((2,)),
                beta_0_scale=jnp.ones((2,)),
                theta_log_loc=jnp.zeros((2,)),
                theta_log_scale=jnp.ones((2,)),
            ),
        )

    def _fake_fit_perturbation_effects(*args, **kwargs):
        captured["guide_effect_strategy"] = kwargs["guide_effect_strategy"]
        captured["guide_activity_mode"] = kwargs["guide_activity_mode"]
        return BetaFit(
            posterior_mean=jnp.zeros((2, 2)),
            posterior_scale=jnp.ones((2, 2)),
            z_values=jnp.zeros((2, 2)),
            losses=jnp.array([1.0]),
            svi_result=None,
        )

    def _fake_summarize(*args, **kwargs):
        frame = pd.DataFrame(np.zeros((2, 2), dtype=np.float32), index=["element_a", "ntc0"], columns=["g1", "g2"])
        return {
            "posterior_mean": frame,
            "posterior_scale": frame,
            "posterior_prob": frame,
        }

    monkeypatch.setattr(api, "load_analysis_cells", _fake_load_analysis_cells)
    monkeypatch.setattr(api, "fit_control", _fake_fit_control)
    monkeypatch.setattr(api, "fit_perturbation_effects", _fake_fit_perturbation_effects)
    monkeypatch.setattr(api, "summarize_betas", _fake_summarize)

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

    assert captured["retain_guide_structure"] is True
    assert captured["guide_effect_strategy"] == "relative"
    assert captured["guide_activity_mode"] == "always_on"


def test_main_chunking_path_passes_membership_argument(monkeypatch, tmp_path) -> None:
    counts = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    pert_id = jnp.array([0, 1], dtype=jnp.int32)
    control_data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pertA"],
        gene_names=["g1", "g2"],
    )

    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame(
                {"perturbation": ["ctrl", "pertA", "pertB"]},
                index=["c0", "c1", "c2"],
            )
            self.obs_names = self.obs.index
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = len(self.obs)
            self.X = np.array([[0, 1], [1, 0], [2, 3]], dtype=np.int32)

    fake_adata = _FakeAdata()
    captured: dict[str, object] = {"chunk_subsets": []}

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "_resolve_adata", lambda *args, **kwargs: fake_adata)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: control_data)
    monkeypatch.setattr(
        api,
        "fit_control",
        lambda *args, **kwargs: ControlFit(
            beta_0=jnp.zeros((2,)),
            theta=jnp.ones((2,)),
            noise_scale=jnp.ones((2,)),
            factor_loadings=None,
            factor_scores=None,
            factor_center=None,
            pca_loadings=None,
            size_factors=jnp.zeros((2, 1)),
            losses=jnp.array([1.0]),
            svi_result=None,
            baseline_posterior=BaselinePosteriorSummary(
                beta_0_loc=jnp.zeros((2,)),
                beta_0_scale=jnp.ones((2,)),
                theta_log_loc=jnp.zeros((2,)),
                theta_log_scale=jnp.ones((2,)),
            ),
        ),
    )

    def _fake_load_analysis_cells(*args, **kwargs):
        subset = kwargs["selected_perturbations"]
        captured["chunk_subsets"].append(tuple(subset))
        n = len(subset)
        return PerTurboData(
            counts=jnp.ones((max(n, 1), 2), dtype=jnp.int32),
            pert_id=jnp.arange(max(n, 1), dtype=jnp.int32),
            pert_names=list(subset),
            gene_names=["g1", "g2"],
        )

    def _fake_fit_perturbation_effects(*args, **kwargs):
        data_arg = args[0]
        n = len(data_arg.pert_names)
        return BetaFit(
            posterior_mean=jnp.zeros((n, 2)),
            posterior_scale=jnp.ones((n, 2)),
            z_values=jnp.zeros((n, 2)),
            losses=jnp.array([1.0]),
            svi_result=None,
        )

    def _fake_summarize(*args, **kwargs):
        frame = pd.DataFrame(
            np.zeros((3, 2), dtype=np.float32),
            index=["ctrl", "pertA", "pertB"],
            columns=["g1", "g2"],
        )
        return {
            "posterior_mean": frame,
            "posterior_scale": frame,
            "posterior_prob": frame,
        }

    monkeypatch.setattr(api, "load_analysis_cells", _fake_load_analysis_cells)
    monkeypatch.setattr(api, "fit_perturbation_effects", _fake_fit_perturbation_effects)
    monkeypatch.setattr(api, "summarize_betas", _fake_summarize)

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
            "--perturbation-chunk-size",
            "1",
        ]
    )

    assert captured["chunk_subsets"] == [("ctrl",), ("pertA",), ("pertB",)]


def test_main_converts_shared_epochs_to_stage_steps(monkeypatch, tmp_path) -> None:
    control_data = PerTurboData(
        counts=jnp.ones((5, 2), dtype=jnp.int32),
        pert_id=jnp.array([0, 0, 0, 1, 1], dtype=jnp.int32),
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2"],
    )
    analysis_data = PerTurboData(
        counts=jnp.ones((7, 2), dtype=jnp.int32),
        pert_id=jnp.array([0, 1, 0, 1, 0, 1, 0], dtype=jnp.int32),
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2"],
    )
    captured: dict[str, int] = {}

    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame({"perturbation": ["ctrl", "pert"]}, index=["c0", "c1"])
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = len(self.obs)
            self.X = np.array([[0, 1], [1, 0]], dtype=np.int32)

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: _FakeAdata())
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: control_data)
    monkeypatch.setattr(api, "load_analysis_cells", lambda *args, **kwargs: analysis_data)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)

    def _fake_fit_control(*args, **kwargs):
        captured["control_steps"] = kwargs["num_steps"]
        return _dummy_control_fit(n_obs=control_data.counts.shape[0])

    def _fake_fit_perturbation_effects(*args, **kwargs):
        captured["beta_steps"] = kwargs["num_steps"]
        return _dummy_beta_fit(n_perts=2)

    def _fake_summarize(*args, **kwargs):
        frame = pd.DataFrame(np.zeros((2, 2), dtype=np.float32), index=["ctrl", "pert"], columns=["g1", "g2"])
        return {
            "posterior_mean": frame,
            "posterior_scale": frame,
            "posterior_prob": frame,
        }

    monkeypatch.setattr(api, "fit_control", _fake_fit_control)
    monkeypatch.setattr(api, "fit_perturbation_effects", _fake_fit_perturbation_effects)
    monkeypatch.setattr(api, "summarize_betas", _fake_summarize)

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
            "--num-epochs",
            "2",
            "--minibatch-size-control",
            "2",
            "--minibatch-size-betas",
            "4",
        ]
    )

    assert captured["control_steps"] == 6
    assert captured["beta_steps"] == 4


def test_main_uses_shared_step_flag_for_both_stages(monkeypatch, tmp_path) -> None:
    data = PerTurboData(
        counts=jnp.ones((3, 2), dtype=jnp.int32),
        pert_id=jnp.array([0, 1, 0], dtype=jnp.int32),
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2"],
    )
    captured: dict[str, int] = {}

    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame({"perturbation": ["ctrl", "pert"]}, index=["c0", "c1"])
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = len(self.obs)
            self.X = np.array([[0, 1], [1, 0]], dtype=np.int32)

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: _FakeAdata())
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "load_analysis_cells", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)

    def _fake_fit_control(*args, **kwargs):
        captured["control_steps"] = kwargs["num_steps"]
        return _dummy_control_fit(n_obs=data.counts.shape[0])

    def _fake_fit_perturbation_effects(*args, **kwargs):
        captured["beta_steps"] = kwargs["num_steps"]
        return _dummy_beta_fit(n_perts=2)

    def _fake_summarize(*args, **kwargs):
        frame = pd.DataFrame(np.zeros((2, 2), dtype=np.float32), index=["ctrl", "pert"], columns=["g1", "g2"])
        return {
            "posterior_mean": frame,
            "posterior_scale": frame,
            "posterior_prob": frame,
        }

    monkeypatch.setattr(api, "fit_control", _fake_fit_control)
    monkeypatch.setattr(api, "fit_perturbation_effects", _fake_fit_perturbation_effects)
    monkeypatch.setattr(api, "summarize_betas", _fake_summarize)

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
            "--num-steps",
            "7",
        ]
    )

    assert captured["control_steps"] == 7
    assert captured["beta_steps"] == 7


def test_main_rejects_mixed_step_and_epoch_schedule(monkeypatch, tmp_path) -> None:
    data = PerTurboData(
        counts=jnp.ones((2, 2), dtype=jnp.int32),
        pert_id=jnp.array([0, 1], dtype=jnp.int32),
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2"],
    )

    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame({"perturbation": ["ctrl", "pert"]}, index=["c0", "c1"])
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = len(self.obs)
            self.X = np.array([[0, 1], [1, 0]], dtype=np.int32)

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: _FakeAdata())
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: data)

    with pytest.raises(ValueError, match="cannot be mixed"):
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
                "--num-steps",
                "5",
                "--num-epochs",
                "2",
            ]
        )


def test_main_resolves_chunked_beta_epochs_per_chunk(monkeypatch, tmp_path) -> None:
    control_data = PerTurboData(
        counts=jnp.ones((5, 2), dtype=jnp.int32),
        pert_id=jnp.array([0, 0, 0, 1, 1], dtype=jnp.int32),
        pert_names=["ctrl", "pertA"],
        gene_names=["g1", "g2"],
    )

    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame(
                {"perturbation": ["ctrl", "pertA", "pertB"]},
                index=["c0", "c1", "c2"],
            )
            self.obs_names = self.obs.index
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = len(self.obs)
            self.X = np.array([[0, 1], [1, 0], [2, 3]], dtype=np.int32)

    fake_adata = _FakeAdata()
    captured: dict[str, object] = {"beta_steps": []}
    chunk_rows = {"ctrl": 2, "pertA": 3, "pertB": 4}

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "_resolve_adata", lambda *args, **kwargs: fake_adata)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: control_data)

    def _fake_fit_control(*args, **kwargs):
        captured["control_steps"] = kwargs["num_steps"]
        return _dummy_control_fit(n_obs=control_data.counts.shape[0])

    def _fake_load_analysis_cells(*args, **kwargs):
        subset = list(kwargs["selected_perturbations"])
        n = chunk_rows[subset[0]]
        return PerTurboData(
            counts=jnp.ones((n, 2), dtype=jnp.int32),
            pert_id=jnp.zeros((n,), dtype=jnp.int32),
            pert_names=subset,
            gene_names=["g1", "g2"],
        )

    def _fake_fit_perturbation_effects(*args, **kwargs):
        beta_steps = captured["beta_steps"]
        assert isinstance(beta_steps, list)
        beta_steps.append(kwargs["num_steps"])
        data_arg = args[0]
        return _dummy_beta_fit(n_perts=len(data_arg.pert_names))

    def _fake_summarize(*args, **kwargs):
        frame = pd.DataFrame(
            np.zeros((3, 2), dtype=np.float32),
            index=["ctrl", "pertA", "pertB"],
            columns=["g1", "g2"],
        )
        return {
            "posterior_mean": frame,
            "posterior_scale": frame,
            "posterior_prob": frame,
        }

    monkeypatch.setattr(api, "fit_control", _fake_fit_control)
    monkeypatch.setattr(api, "load_analysis_cells", _fake_load_analysis_cells)
    monkeypatch.setattr(api, "fit_perturbation_effects", _fake_fit_perturbation_effects)
    monkeypatch.setattr(api, "summarize_betas", _fake_summarize)

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
            "--perturbation-chunk-size",
            "1",
            "--num-epochs-control",
            "3",
            "--num-epochs-betas",
            "2",
            "--minibatch-size-control",
            "2",
            "--minibatch-size-betas",
            "3",
        ]
    )

    assert captured["control_steps"] == 9
    assert captured["beta_steps"] == [2, 2, 4]


def test_construct_perturbation_chunks_dynamic_cell_cap() -> None:
    chunks = api._construct_perturbation_chunks(
        ["p0", "p1", "p2"],
        [
            np.array([0, 1], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            np.array([3, 4], dtype=np.int64),
        ],
        max_chunk_size=3,
        max_perturbations_per_chunk=None,
    )

    assert [chunk.pert_names for chunk in chunks] == [["p0", "p1"], ["p2"]]
    assert [chunk.cell_indices.size for chunk in chunks] == [3, 2]


def test_construct_perturbation_chunks_honors_perturbation_cap() -> None:
    chunks = api._construct_perturbation_chunks(
        ["p0", "p1", "p2", "p3"],
        [
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.array([3], dtype=np.int64),
        ],
        max_chunk_size=10,
        max_perturbations_per_chunk=2,
    )
    assert [chunk.pert_names for chunk in chunks] == [["p0", "p1"], ["p2", "p3"]]


def test_construct_perturbation_chunks_isolates_an_oversized_single_perturbation(capsys) -> None:
    """A perturbation bigger than the cap is kept whole, not refused.

    This used to raise. Screens exist with tens of thousands of cells behind one
    perturbation, and its cells cannot be split across chunks without breaking the
    estimate, so the cap yields and the run says which perturbation set peak memory.
    See tests/test_oversized_perturbation_chunking.py for the full behaviour.
    """
    chunks = api._construct_perturbation_chunks(
        ["p0"],
        [np.array([0, 1, 2], dtype=np.int64)],
        max_chunk_size=2,
        max_perturbations_per_chunk=None,
    )
    assert [chunk.pert_names for chunk in chunks] == [["p0"]]
    assert chunks[0].cell_indices.size == 3
    assert "exceed --max-chunk-size" in capsys.readouterr().out


class _FakeDevice:
    def __init__(self, platform: str, idx: int):
        self.platform = platform
        self.idx = idx


def test_get_device_supports_platform_index(monkeypatch) -> None:
    devices = [_FakeDevice("gpu", 0), _FakeDevice("gpu", 1), _FakeDevice("cpu", 0)]
    monkeypatch.setattr(api.jax, "devices", lambda: devices)
    selected = api._get_device("gpu:1")
    assert selected is devices[1]


def test_get_device_raises_for_out_of_bounds_index(monkeypatch) -> None:
    devices = [_FakeDevice("gpu", 0)]
    monkeypatch.setattr(api.jax, "devices", lambda: devices)
    with pytest.raises(ValueError, match="only 1 device\\(s\\) are available"):
        api._get_device("gpu:2")


def test_main_rejects_element_grouping_without_perturbation_modality(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: object())
    with pytest.raises(ValueError, match="--perturbation-element-varm-key requires --perturbation-modality-key"):
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
                "--perturbation-element-varm-key",
                "element_targeted",
            ]
        )


def test_main_rejects_missing_size_factor_key_early(monkeypatch, tmp_path) -> None:
    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame({"perturbation": ["ctrl", "pertA"]}, index=["c0", "c1"])
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = len(self.obs)
            self.X = np.array([[0, 1], [1, 0]], dtype=np.int32)

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: _FakeAdata())
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: pytest.fail("load_controls should not be called"))

    with pytest.raises(KeyError, match="size_factor_key 'missing_umi' not found in adata.obs"):
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
                "--size-factor-key",
                "missing_umi",
            ]
        )


def test_main_rejects_missing_library_size_key_early(monkeypatch, tmp_path) -> None:
    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame({"perturbation": ["ctrl", "pertA"]}, index=["c0", "c1"])
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = len(self.obs)
            self.X = np.array([[0, 1], [1, 0]], dtype=np.int32)

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: _FakeAdata())
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: pytest.fail("load_controls should not be called"))

    with pytest.raises(KeyError, match="library_size_key 'missing_umi' not found in adata.obs"):
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
                "missing_umi",
            ]
        )


def test_main_rejects_both_size_factor_and_library_size_keys(tmp_path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
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
                "--size-factor-key",
                "sf",
                "--library-size-key",
                "umi_count",
            ]
        )


def test_main_size_factor_mode_none_uses_fixed_zero_size_factors(monkeypatch, tmp_path) -> None:
    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame({"perturbation": ["ctrl", "pertA"]}, index=["c0", "c1"])
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = len(self.obs)
            self.X = np.array([[0, 1], [1, 0]], dtype=np.int32)

    data_with_nonzero_sf = PerTurboData(
        counts=jnp.array([[0, 1], [1, 0]], dtype=jnp.int32),
        pert_id=jnp.array([0, 1], dtype=jnp.int32),
        pert_names=["ctrl", "pertA"],
        gene_names=["g1", "g2"],
        size_factors=jnp.array([[3.0], [4.0]], dtype=jnp.float32),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: _FakeAdata())
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: data_with_nonzero_sf)
    monkeypatch.setattr(api, "load_analysis_cells", lambda *args, **kwargs: data_with_nonzero_sf)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)

    def _fake_fit_control(*args, **kwargs):
        data_arg = args[0]
        captured["control_use_observed"] = kwargs["use_observed_size_factors"]
        captured["control_size_factors"] = np.asarray(data_arg.size_factors)
        return _dummy_control_fit(n_obs=data_arg.counts.shape[0])

    def _fake_fit_perturbation_effects(*args, **kwargs):
        data_arg = args[0]
        captured["beta_use_observed"] = kwargs["use_observed_size_factors"]
        captured["beta_size_factors"] = np.asarray(data_arg.size_factors)
        return _dummy_beta_fit(n_perts=2)

    def _fake_summarize(*args, **kwargs):
        frame = pd.DataFrame(np.zeros((2, 2), dtype=np.float32), index=["ctrl", "pertA"], columns=["g1", "g2"])
        return {
            "posterior_mean": frame,
            "posterior_scale": frame,
            "posterior_prob": frame,
        }

    monkeypatch.setattr(api, "fit_control", _fake_fit_control)
    monkeypatch.setattr(api, "fit_perturbation_effects", _fake_fit_perturbation_effects)
    monkeypatch.setattr(api, "summarize_betas", _fake_summarize)

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
            "--size-factor-key",
            "missing_umi",
            "--size-factor-mode",
            "none",
        ]
    )

    assert captured["control_use_observed"] is True
    assert captured["beta_use_observed"] is True
    np.testing.assert_allclose(captured["control_size_factors"], np.zeros((2, 1), dtype=np.float32))
    np.testing.assert_allclose(captured["beta_size_factors"], np.zeros((2, 1), dtype=np.float32))


def test_main_rejects_size_factor_mode_none_with_use_observed_flag(tmp_path) -> None:
    with pytest.raises(ValueError, match="cannot be combined with --size-factor-mode=none"):
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
                "--size-factor-mode",
                "none",
                "--use-observed-size-factors",
            ]
        )


def test_main_rejects_missing_modality_key_early(monkeypatch, tmp_path) -> None:
    class _FakeMuData:
        def __init__(self) -> None:
            self.mod = {"gene": object(), "guide": object()}

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: _FakeMuData())
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: pytest.fail("load_controls should not be called"))

    with pytest.raises(KeyError, match="modality_key 'rna' not found in MuData.mod"):
        api.main(
            [
                "--input",
                "dummy.h5mu",
                "--out-dir",
                str(tmp_path),
                "--modality-key",
                "rna",
                "--perturbation-modality-key",
                "guide",
            ]
        )


def test_main_accepts_mixture_nb_likelihood(monkeypatch, tmp_path) -> None:
    counts = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    pert_id = jnp.array([0, 1], dtype=jnp.int32)
    data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2"],
    )
    captured: dict[str, object] = {}

    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame({"perturbation": ["ctrl", "pert"]}, index=["c0", "c1"])
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = len(self.obs)
            self.X = np.array([[0, 1], [1, 0]], dtype=np.int32)

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: _FakeAdata())
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "load_analysis_cells", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)

    def _fake_fit_control(*args, **kwargs):
        captured["control_model_name"] = kwargs["model_name"]
        return ControlFit(
            beta_0=jnp.zeros((2,)),
            theta=jnp.ones((2,)),
            noise_scale=jnp.ones((2,)),
            factor_loadings=None,
            factor_scores=None,
            factor_center=None,
            pca_loadings=None,
            size_factors=jnp.zeros((2, 1)),
            losses=jnp.array([1.0]),
            svi_result=None,
        )

    def _fake_fit_perturbation_effects(*args, **kwargs):
        captured["beta_model_name"] = kwargs["model_name"]
        return BetaFit(
            posterior_mean=jnp.zeros((2, 2)),
            posterior_scale=jnp.ones((2, 2)),
            z_values=jnp.zeros((2, 2)),
            losses=jnp.array([1.0]),
            svi_result=None,
        )

    monkeypatch.setattr(api, "fit_control", _fake_fit_control)
    monkeypatch.setattr(api, "fit_perturbation_effects", _fake_fit_perturbation_effects)
    monkeypatch.setattr(
        api,
        "summarize_betas",
        lambda *args, **kwargs: {
            "posterior_mean": pd.DataFrame(np.zeros((2, 2))),
            "posterior_scale": pd.DataFrame(np.zeros((2, 2))),
            "posterior_prob": pd.DataFrame(np.zeros((2, 2))),
        },
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
            "--likelihood",
            "mixture_nb",
        ]
    )

    assert captured["control_model_name"] == "mixture_nb"
    assert captured["beta_model_name"] == "mixture_nb"


def test_main_accepts_nb_alias_for_negbin(monkeypatch, tmp_path) -> None:
    counts = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    pert_id = jnp.array([0, 1], dtype=jnp.int32)
    data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2"],
    )
    captured: dict[str, object] = {}

    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame({"perturbation": ["ctrl", "pert"]}, index=["c0", "c1"])
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = len(self.obs)
            self.X = np.array([[0, 1], [1, 0]], dtype=np.int32)

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: _FakeAdata())
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "load_analysis_cells", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)

    def _fake_fit_control(*args, **kwargs):
        captured["control_model_name"] = kwargs["model_name"]
        return _dummy_control_fit(n_obs=2)

    def _fake_fit_perturbation_effects(*args, **kwargs):
        captured["beta_model_name"] = kwargs["model_name"]
        return _dummy_beta_fit(n_perts=2)

    monkeypatch.setattr(api, "fit_control", _fake_fit_control)
    monkeypatch.setattr(api, "fit_perturbation_effects", _fake_fit_perturbation_effects)
    monkeypatch.setattr(
        api,
        "summarize_betas",
        lambda *args, **kwargs: {
            "posterior_mean": pd.DataFrame(np.zeros((2, 2))),
            "posterior_scale": pd.DataFrame(np.zeros((2, 2))),
            "posterior_prob": pd.DataFrame(np.zeros((2, 2))),
        },
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
            "--likelihood",
            "nb",
        ]
    )

    assert captured["control_model_name"] == "negbin"
    assert captured["beta_model_name"] == "negbin"


def test_main_threads_censored_percentile_from_clip_flag(monkeypatch, tmp_path) -> None:
    counts = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    pert_id = jnp.array([0, 1], dtype=jnp.int32)
    data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pert"],
        gene_names=["g1", "g2"],
    )
    captured: dict[str, float] = {}

    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame({"perturbation": ["ctrl", "pert"]}, index=["c0", "c1"])
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = len(self.obs)
            self.X = np.array([[0, 1], [1, 0]], dtype=np.int32)

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: _FakeAdata())
    monkeypatch.setattr(api, "load_controls", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "load_analysis_cells", lambda *args, **kwargs: data)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)

    def _fake_fit_control(*args, **kwargs):
        captured["control"] = kwargs["count_censoring_percentile"]
        return _dummy_control_fit(n_obs=2)

    def _fake_fit_perturbation_effects(*args, **kwargs):
        captured["beta"] = kwargs["count_censoring_percentile"]
        return _dummy_beta_fit(n_perts=2)

    monkeypatch.setattr(api, "fit_control", _fake_fit_control)
    monkeypatch.setattr(api, "fit_perturbation_effects", _fake_fit_perturbation_effects)
    monkeypatch.setattr(
        api,
        "summarize_betas",
        lambda *args, **kwargs: {
            "posterior_mean": pd.DataFrame(np.zeros((2, 2))),
            "posterior_scale": pd.DataFrame(np.zeros((2, 2))),
            "posterior_prob": pd.DataFrame(np.zeros((2, 2))),
        },
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
            "--likelihood",
            "censored_nb",
            "--clip-gene-expression-percentile",
            "99.5",
        ]
    )

    assert captured["control"] == pytest.approx(99.5)
    assert captured["beta"] == pytest.approx(99.5)


def test_main_rejects_default_clip_percentile_for_censored_likelihood(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: object())

    with pytest.raises(ValueError, match="--clip-gene-expression-percentile < 100"):
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
                "--likelihood",
                "censored_nb",
            ]
        )


def test_main_covariate_flags_threaded_and_transform_reused_for_chunks(monkeypatch, tmp_path) -> None:
    counts = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    pert_id = jnp.array([0, 1], dtype=jnp.int32)
    control_data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pertA"],
        gene_names=["g1", "g2"],
    )
    cov_state = CovariateTransformState(
        continuous_covariates=["percent_mito", "guide_count"],
        batch_covariate="prep_batch",
        continuous_medians={"percent_mito": 0.1, "guide_count": 1.0},
        continuous_transforms={"percent_mito": "zscore", "guide_count": "log1p+zscore"},
        continuous_means={"percent_mito": 0.2, "guide_count": 1.0},
        continuous_stds={"percent_mito": 0.05, "guide_count": 0.2},
        batch_reference="b1",
        batch_levels=["b2"],
        all_feature_names=["percent_mito", "guide_count", "batch:prep_batch=b2"],
        feature_names=["percent_mito", "guide_count", "batch:prep_batch=b2"],
        dropped_features=[],
    )

    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame(
                {
                    "perturbation": ["ctrl", "pertA", "pertB"],
                    "percent_mito": [0.1, 0.2, 0.3],
                    "guide_count": [1, 2, 3],
                    "prep_batch": ["b1", "b2", "b1"],
                },
                index=["c0", "c1", "c2"],
            )
            self.obs_names = self.obs.index
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = len(self.obs)
            self.X = np.array([[0, 1], [1, 0], [2, 3]], dtype=np.int32)

    fake_adata = _FakeAdata()
    captured: dict[str, object] = {"load_analysis_cells_kwargs": []}

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "_resolve_adata", lambda *args, **kwargs: fake_adata)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)

    def _fake_load_controls(*args, **kwargs):
        assert kwargs["continuous_covariates"] == ["percent_mito", "guide_count"]
        assert kwargs["batch_covariate"] == "prep_batch"
        assert kwargs["return_covariate_transform_state"] is True
        return control_data, cov_state

    def _fake_load_analysis_cells(*args, **kwargs):
        captured["load_analysis_cells_kwargs"].append(kwargs)
        subset = kwargs["selected_perturbations"]
        n = len(subset)
        return PerTurboData(
            counts=jnp.ones((max(n, 1), 2), dtype=jnp.int32),
            pert_id=jnp.arange(max(n, 1), dtype=jnp.int32),
            pert_names=list(subset),
            gene_names=["g1", "g2"],
        )

    monkeypatch.setattr(api, "load_controls", _fake_load_controls)
    monkeypatch.setattr(api, "load_analysis_cells", _fake_load_analysis_cells)
    monkeypatch.setattr(
        api,
        "fit_control",
        lambda *args, **kwargs: ControlFit(
            beta_0=jnp.zeros((2,)),
            theta=jnp.ones((2,)),
            noise_scale=jnp.ones((2,)),
            factor_loadings=None,
            factor_scores=None,
            factor_center=None,
            pca_loadings=None,
            size_factors=jnp.zeros((2, 1)),
            losses=jnp.array([1.0]),
            svi_result=None,
        ),
    )
    monkeypatch.setattr(
        api,
        "fit_perturbation_effects",
        lambda *args, **kwargs: BetaFit(
            posterior_mean=jnp.zeros((len(args[0].pert_names), 2)),
            posterior_scale=jnp.ones((len(args[0].pert_names), 2)),
            z_values=jnp.zeros((len(args[0].pert_names), 2)),
            losses=jnp.array([1.0]),
            svi_result=None,
        ),
    )
    monkeypatch.setattr(
        api,
        "summarize_betas",
        lambda *args, **kwargs: {
            "posterior_mean": pd.DataFrame(np.zeros((3, 2))),
            "posterior_scale": pd.DataFrame(np.zeros((3, 2))),
            "posterior_prob": pd.DataFrame(np.zeros((3, 2))),
        },
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
            "--perturbation-chunk-size",
            "1",
            "--continuous-covariates",
            "percent_mito",
            "guide_count",
            "--batch-covariate",
            "prep_batch",
        ]
    )

    assert len(captured["load_analysis_cells_kwargs"]) == 3
    for kwargs in captured["load_analysis_cells_kwargs"]:
        assert kwargs["continuous_covariates"] == ["percent_mito", "guide_count"]
        assert kwargs["batch_covariate"] == "prep_batch"
        assert kwargs["covariate_transform_state"] is cov_state


def test_main_default_outlier_handling_is_noop(monkeypatch, tmp_path) -> None:
    counts = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    pert_id = jnp.array([0, 1], dtype=jnp.int32)
    control_data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pertA"],
        gene_names=["g1", "g2"],
    )

    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame(
                {"perturbation": ["ctrl", "pertA", "pertB"]},
                index=["c0", "c1", "c2"],
            )
            self.obs_names = self.obs.index
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = len(self.obs)
            self.X = np.array([[0, 10], [5, 0], [2, 8]], dtype=np.int32)

    fake_adata = _FakeAdata()
    captured: dict[str, object] = {"analysis_masks": []}

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "_resolve_adata", lambda *args, **kwargs: fake_adata)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)

    def _fake_load_controls(*args, **kwargs):
        captured["control_mask"] = kwargs["cell_keep_mask"]
        captured["control_winsorize"] = kwargs["winsorize_gene_expression"]
        return control_data, None

    def _fake_load_analysis_cells(*args, **kwargs):
        captured["analysis_masks"].append(kwargs["cell_keep_mask"])
        captured.setdefault("analysis_winsorize", []).append(kwargs["winsorize_gene_expression"])
        subset = kwargs["selected_perturbations"]
        return PerTurboData(
            counts=jnp.ones((len(subset), 2), dtype=jnp.int32),
            pert_id=jnp.arange(len(subset), dtype=jnp.int32),
            pert_names=list(subset),
            gene_names=["g1", "g2"],
        )

    monkeypatch.setattr(api, "load_controls", _fake_load_controls)
    monkeypatch.setattr(api, "load_analysis_cells", _fake_load_analysis_cells)
    monkeypatch.setattr(
        api,
        "fit_control",
        lambda *args, **kwargs: ControlFit(
            beta_0=jnp.zeros((2,)),
            theta=jnp.ones((2,)),
            noise_scale=jnp.ones((2,)),
            factor_loadings=None,
            factor_scores=None,
            factor_center=None,
            pca_loadings=None,
            size_factors=jnp.zeros((2, 1)),
            losses=jnp.array([1.0]),
            svi_result=None,
        ),
    )
    monkeypatch.setattr(
        api,
        "fit_perturbation_effects",
        lambda *args, **kwargs: BetaFit(
            posterior_mean=jnp.zeros((len(args[0].pert_names), 2)),
            posterior_scale=jnp.ones((len(args[0].pert_names), 2)),
            z_values=jnp.zeros((len(args[0].pert_names), 2)),
            losses=jnp.array([1.0]),
            svi_result=None,
        ),
    )
    monkeypatch.setattr(
        api,
        "summarize_betas",
        lambda *args, **kwargs: {
            "posterior_mean": pd.DataFrame(np.zeros((3, 2))),
            "posterior_scale": pd.DataFrame(np.zeros((3, 2))),
            "posterior_prob": pd.DataFrame(np.zeros((3, 2))),
        },
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
            "--perturbation-chunk-size",
            "1",
        ]
    )

    assert captured["control_mask"] is None
    assert captured["control_winsorize"] is False
    assert len(captured["analysis_masks"]) == 3
    assert all(mask is None for mask in captured["analysis_masks"])
    assert all(flag is False for flag in captured["analysis_winsorize"])


def test_main_filter_cells_mode_reuses_cell_mask_and_filtered_chunking(monkeypatch, tmp_path) -> None:
    counts = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    pert_id = jnp.array([0, 1], dtype=jnp.int32)
    control_data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pertB"],
        gene_names=["g1", "g2"],
    )
    outlier_thresholds = np.array([10, 10], dtype=np.int32)
    outlier_gene_counts = np.array([0, 120, 5], dtype=np.int32)

    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame(
                {"perturbation": ["ctrl", "pertA", "pertB"]},
                index=["c0", "c1", "c2"],
            )
            self.obs_names = self.obs.index
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = len(self.obs)
            self.X = np.array([[0, 1], [20, 0], [2, 8]], dtype=np.int32)

        def __getitem__(self, idx):
            subset = _FakeAdata()
            subset.obs = self.obs.iloc[idx].copy()
            subset.obs_names = subset.obs.index
            subset.var = self.var
            subset.n_obs = len(subset.obs)
            subset.X = self.X[idx]
            return subset

    fake_adata = _FakeAdata()
    captured: dict[str, object] = {"chunk_subsets": [], "control_masks": [], "analysis_masks": []}

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "_resolve_adata", lambda *args, **kwargs: fake_adata)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_compute_gene_outlier_thresholds", lambda *args, **kwargs: outlier_thresholds)
    monkeypatch.setattr(api, "_count_gene_outliers_per_cell", lambda *args, **kwargs: outlier_gene_counts)

    def _fake_load_controls(*args, **kwargs):
        captured["control_masks"].append(kwargs["cell_keep_mask"])
        captured["control_winsorize"] = kwargs["winsorize_gene_expression"]
        assert kwargs["gene_clip_thresholds"] is None
        return control_data, None

    def _fake_load_analysis_cells(*args, **kwargs):
        captured["analysis_masks"].append(kwargs["cell_keep_mask"])
        captured.setdefault("analysis_winsorize", []).append(kwargs["winsorize_gene_expression"])
        assert kwargs["gene_clip_thresholds"] is None
        subset = kwargs["selected_perturbations"]
        captured["chunk_subsets"].append(tuple(subset))
        return PerTurboData(
            counts=jnp.ones((len(subset), 2), dtype=jnp.int32),
            pert_id=jnp.arange(len(subset), dtype=jnp.int32),
            pert_names=list(subset),
            gene_names=["g1", "g2"],
        )

    monkeypatch.setattr(api, "load_controls", _fake_load_controls)
    monkeypatch.setattr(api, "load_analysis_cells", _fake_load_analysis_cells)
    monkeypatch.setattr(
        api,
        "fit_control",
        lambda *args, **kwargs: ControlFit(
            beta_0=jnp.zeros((2,)),
            theta=jnp.ones((2,)),
            noise_scale=jnp.ones((2,)),
            factor_loadings=None,
            factor_scores=None,
            factor_center=None,
            pca_loadings=None,
            size_factors=jnp.zeros((2, 1)),
            losses=jnp.array([1.0]),
            svi_result=None,
        ),
    )
    monkeypatch.setattr(
        api,
        "fit_perturbation_effects",
        lambda *args, **kwargs: BetaFit(
            posterior_mean=jnp.zeros((len(args[0].pert_names), 2)),
            posterior_scale=jnp.ones((len(args[0].pert_names), 2)),
            z_values=jnp.zeros((len(args[0].pert_names), 2)),
            losses=jnp.array([1.0]),
            svi_result=None,
        ),
    )
    monkeypatch.setattr(
        api,
        "summarize_betas",
        lambda *args, **kwargs: {
            "posterior_mean": pd.DataFrame(np.zeros((2, 2)), index=["ctrl", "pertB"], columns=["g1", "g2"]),
            "posterior_scale": pd.DataFrame(np.zeros((2, 2)), index=["ctrl", "pertB"], columns=["g1", "g2"]),
            "posterior_prob": pd.DataFrame(np.zeros((2, 2)), index=["ctrl", "pertB"], columns=["g1", "g2"]),
        },
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
            "--perturbation-chunk-size",
            "1",
            "--gene-outlier-action",
            "filter_cells",
            "--clip-gene-expression-percentile",
            "99",
            "--gene-outlier-threshold-floor",
            "10",
            "--outlier-cell-min-genes",
            "100",
        ]
    )

    expected_mask = np.array([True, False, True], dtype=bool)
    assert len(captured["control_masks"]) == 1
    assert np.array_equal(captured["control_masks"][0], expected_mask)
    assert captured["control_winsorize"] is False
    assert len(captured["analysis_masks"]) == 2
    assert all(np.array_equal(mask, expected_mask) for mask in captured["analysis_masks"])
    assert all(flag is False for flag in captured["analysis_winsorize"])
    assert captured["chunk_subsets"] == [("ctrl",), ("pertB",)]


def test_main_filter_cells_and_winsorize_reuses_mask_and_thresholds(monkeypatch, tmp_path) -> None:
    counts = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    pert_id = jnp.array([0, 1], dtype=jnp.int32)
    control_data = PerTurboData(
        counts=counts,
        pert_id=pert_id,
        pert_names=["ctrl", "pertB"],
        gene_names=["g1", "g2"],
    )
    outlier_thresholds = np.array([10, 10], dtype=np.int32)
    outlier_gene_counts = np.array([0, 120, 5], dtype=np.int32)

    class _FakeAdata:
        def __init__(self) -> None:
            self.obs = pd.DataFrame(
                {"perturbation": ["ctrl", "pertA", "pertB"]},
                index=["c0", "c1", "c2"],
            )
            self.obs_names = self.obs.index
            self.var = pd.DataFrame(index=["g1", "g2"])
            self.n_obs = len(self.obs)
            self.X = np.array([[0, 1], [20, 0], [2, 8]], dtype=np.int32)

        def __getitem__(self, idx):
            subset = _FakeAdata()
            subset.obs = self.obs.iloc[idx].copy()
            subset.obs_names = subset.obs.index
            subset.var = self.var
            subset.n_obs = len(subset.obs)
            subset.X = self.X[idx]
            return subset

    fake_adata = _FakeAdata()
    captured: dict[str, object] = {"chunk_subsets": [], "control_masks": [], "analysis_masks": []}

    monkeypatch.setattr(api, "_load_from_path_with_backing", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "_resolve_adata", lambda *args, **kwargs: fake_adata)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_compute_gene_outlier_thresholds", lambda *args, **kwargs: outlier_thresholds)
    monkeypatch.setattr(api, "_count_gene_outliers_per_cell", lambda *args, **kwargs: outlier_gene_counts)

    def _fake_load_controls(*args, **kwargs):
        captured["control_masks"].append(kwargs["cell_keep_mask"])
        captured["control_thresholds"] = kwargs["gene_clip_thresholds"]
        captured["control_winsorize"] = kwargs["winsorize_gene_expression"]
        return control_data, None

    def _fake_load_analysis_cells(*args, **kwargs):
        captured["analysis_masks"].append(kwargs["cell_keep_mask"])
        captured.setdefault("analysis_thresholds", []).append(kwargs["gene_clip_thresholds"])
        captured.setdefault("analysis_winsorize", []).append(kwargs["winsorize_gene_expression"])
        subset = kwargs["selected_perturbations"]
        captured["chunk_subsets"].append(tuple(subset))
        return PerTurboData(
            counts=jnp.ones((len(subset), 2), dtype=jnp.int32),
            pert_id=jnp.arange(len(subset), dtype=jnp.int32),
            pert_names=list(subset),
            gene_names=["g1", "g2"],
        )

    monkeypatch.setattr(api, "load_controls", _fake_load_controls)
    monkeypatch.setattr(api, "load_analysis_cells", _fake_load_analysis_cells)
    monkeypatch.setattr(
        api,
        "fit_control",
        lambda *args, **kwargs: ControlFit(
            beta_0=jnp.zeros((2,)),
            theta=jnp.ones((2,)),
            noise_scale=jnp.ones((2,)),
            factor_loadings=None,
            factor_scores=None,
            factor_center=None,
            pca_loadings=None,
            size_factors=jnp.zeros((2, 1)),
            losses=jnp.array([1.0]),
            svi_result=None,
        ),
    )
    monkeypatch.setattr(
        api,
        "fit_perturbation_effects",
        lambda *args, **kwargs: BetaFit(
            posterior_mean=jnp.zeros((len(args[0].pert_names), 2)),
            posterior_scale=jnp.ones((len(args[0].pert_names), 2)),
            z_values=jnp.zeros((len(args[0].pert_names), 2)),
            losses=jnp.array([1.0]),
            svi_result=None,
        ),
    )
    monkeypatch.setattr(
        api,
        "summarize_betas",
        lambda *args, **kwargs: {
            "posterior_mean": pd.DataFrame(np.zeros((2, 2)), index=["ctrl", "pertB"], columns=["g1", "g2"]),
            "posterior_scale": pd.DataFrame(np.zeros((2, 2)), index=["ctrl", "pertB"], columns=["g1", "g2"]),
            "posterior_prob": pd.DataFrame(np.zeros((2, 2)), index=["ctrl", "pertB"], columns=["g1", "g2"]),
        },
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
            "--perturbation-chunk-size",
            "1",
            "--gene-outlier-action",
            "filter_cells",
            "--clip-gene-expression-percentile",
            "99",
            "--gene-outlier-threshold-floor",
            "10",
            "--outlier-cell-min-genes",
            "100",
            "--winsorize-gene-expression-outliers",
        ]
    )

    expected_mask = np.array([True, False, True], dtype=bool)
    assert len(captured["control_masks"]) == 1
    assert np.array_equal(captured["control_masks"][0], expected_mask)
    assert captured["control_winsorize"] is True
    assert captured["control_thresholds"] is outlier_thresholds
    assert len(captured["analysis_masks"]) == 2
    assert all(np.array_equal(mask, expected_mask) for mask in captured["analysis_masks"])
    assert all(flag is True for flag in captured["analysis_winsorize"])
    assert all(thresholds is outlier_thresholds for thresholds in captured["analysis_thresholds"])
    assert captured["chunk_subsets"] == [("ctrl",), ("pertB",)]


def test_fit_from_path_maps_the_crt_keywords_onto_the_flags(monkeypatch) -> None:
    """The saddlepoint-only all-cells CRT with no stage two, as a power calculator would call it."""
    captured: dict[str, list[str]] = {}

    def _fake_main(argv):
        captured["argv"] = list(argv)

    monkeypatch.setattr(api, "main", _fake_main)
    api.fit_from_path(
        "screen.h5mu",
        "out",
        modality_key="gene",
        perturbation_modality_key="guide",
        perturbation_element_varm_key="guide_intended_target_pairs",
        perturbation_element_names_uns_key="intended_targets",
        crt=True,
        crt_only=True,
        crt_pool="all-cells",
        crt_mechanism="propensity",
        crt_tail_families=("saddlepoint",),
        crt_saddlepoint_only=True,
        crt_allow_unconverged_baseline=True,
        crt_polish_baseline=True,
    )
    argv = captured["argv"]
    for flag in ("--crt", "--crt-only", "--crt-saddlepoint-only", "--crt-allow-unconverged-baseline", "--crt-polish-baseline"):
        assert flag in argv
    for flag, value in (("--crt-pool", "all-cells"), ("--crt-mechanism", "propensity"), ("--crt-tail-families", "saddlepoint")):
        assert argv[argv.index(flag) + 1] == value

    api.fit_from_path("screen.h5mu", "out", modality_key="gene", perturbation_key="perturbation", control_substring="non-targeting")
    assert "--crt" not in captured["argv"]
    assert "--crt-only" not in captured["argv"]
