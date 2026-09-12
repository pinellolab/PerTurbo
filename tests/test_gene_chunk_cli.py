"""End-to-end CLI coverage for disk-backed stage-two gene blocks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import jax.numpy as jnp
import mudata as md
import numpy as np
import pandas as pd
import scipy.sparse as sp

import perturbo.api as api
import perturbo.cli as cli_module
from perturbo.crt import ChunkCRTResult
from perturbo.io import load_fit_bundle
from perturbo.sparse_design import IndexedDesignMatrix, indexed_design_to_dense


GENES = [f"gene_{index}" for index in range(5)]
GUIDES = ["ctrl", "guide_a", "guide_b"]
ELEMENTS = ["ctrl", "element_a", "element_b"]


def _write_cooccurring_screen(path: Path) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array(
        [
            [1, 2, 3, 4, 5],
            [3, 1, 4, 1, 5],
            [2, 7, 1, 8, 2],
            [6, 1, 8, 0, 3],
            [5, 9, 2, 6, 5],
            [3, 5, 8, 9, 7],
            [9, 3, 2, 3, 8],
            [4, 6, 2, 6, 4],
        ],
        dtype=np.int32,
    )
    assignments = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(index=[f"cell_{index}" for index in range(counts.shape[0])])
    rna = ad.AnnData(
        X=sp.csr_matrix(counts),
        obs=obs.copy(),
        var=pd.DataFrame(index=GENES),
    )
    guides = ad.AnnData(
        X=sp.csr_matrix(assignments),
        obs=obs.copy(),
        var=pd.DataFrame(index=GUIDES),
    )
    guides.varm["mapping"] = pd.DataFrame(
        np.eye(len(GUIDES), dtype=np.float32),
        index=GUIDES,
        columns=ELEMENTS,
    )
    md.MuData({"rna": rna, "grna": guides}).write_h5mu(path)
    return counts, assignments


def _control_fit(data, **kwargs):
    del kwargs
    n_genes = int(data.counts.shape[1])
    return api.ControlFit(
        beta_0=jnp.arange(n_genes, dtype=jnp.float32) + 10.0,
        theta=jnp.arange(n_genes, dtype=jnp.float32) + 20.0,
        noise_scale=jnp.ones(n_genes, dtype=jnp.float32),
        factor_loadings=None,
        factor_scores=None,
        factor_center=None,
        pca_loadings=None,
        size_factors=data.size_factors,
        losses=jnp.array([], dtype=jnp.float32),
        svi_result=None,
    )


def _gene_values(row_count: int, gene_names: list[str], *, base: float) -> np.ndarray:
    gene_ids = np.array([GENES.index(name) for name in gene_names], dtype=np.float32)
    row_ids = np.arange(row_count, dtype=np.float32)[:, None]
    return base + 100.0 * row_ids + gene_ids[None, :]


def test_backed_gene_blocks_keep_the_joint_design_offsets_and_combined_posterior(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "screen.h5mu"
    out = tmp_path / "out"
    counts, assignments = _write_cooccurring_screen(source)

    original_load = api.load_analysis_cells
    requested_gene_indices: list[slice | None] = []
    fit_calls: list[dict[str, np.ndarray | list[str] | tuple[int, ...]]] = []

    def recording_load(*args, **kwargs):
        requested_gene_indices.append(kwargs.get("selected_gene_indices"))
        return original_load(*args, **kwargs)

    def beta_fit(data, control, **kwargs):
        del kwargs
        assert isinstance(data.pert_id, IndexedDesignMatrix)
        dense_elements = np.asarray(indexed_design_to_dense(data.pert_id))
        np.testing.assert_array_equal(dense_elements, assignments)
        # Shared guide effects need only the grouped element design. Avoiding a
        # second indexed guide matrix is part of the memory-bounded path.
        assert data.guide_matrix is None

        names = list(data.gene_names)
        gene_ids = np.array([GENES.index(name) for name in names])
        np.testing.assert_array_equal(np.asarray(data.counts), counts[:, gene_ids])
        np.testing.assert_array_equal(np.asarray(control.beta_0), 10.0 + gene_ids)
        np.testing.assert_array_equal(np.asarray(control.theta), 20.0 + gene_ids)
        fit_calls.append(
            {
                "shape": tuple(data.counts.shape),
                "genes": names,
                "offsets": np.asarray(data.size_factors).copy(),
            }
        )

        n_elements = len(data.pert_names)
        return api.BetaFit(
            posterior_mean=jnp.asarray(_gene_values(n_elements, names, base=0.0)),
            posterior_scale=jnp.asarray(_gene_values(n_elements, names, base=1_000.0)),
            z_values=jnp.asarray(_gene_values(n_elements, names, base=2_000.0)),
            losses=jnp.array([], dtype=jnp.float32),
            svi_result=None,
        )

    monkeypatch.setattr(api, "load_analysis_cells", recording_load)
    monkeypatch.setattr(api, "fit_control", _control_fit)
    monkeypatch.setattr(api, "fit_perturbation_effects", beta_fit)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_save_multi_loss_plot", lambda *args, **kwargs: None)

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
            "--likelihood",
            "negbin",
            "--gene-chunk-size",
            "2",
            "--num-steps-control",
            "1",
            "--num-steps-betas",
            "1",
            "--backed",
            "--no-crt",
        ]
    )

    assert requested_gene_indices == [slice(0, 2), slice(2, 4), slice(4, 5)]
    assert [call["shape"] for call in fit_calls] == [(8, 2), (8, 2), (8, 1)]
    assert [call["genes"] for call in fit_calls] == [GENES[:2], GENES[2:4], GENES[4:]]

    control_center = np.log1p(counts[:2].sum(axis=1)).mean()
    expected_offsets = (np.log1p(counts.sum(axis=1)) - control_center)[:, None]
    for call in fit_calls:
        np.testing.assert_allclose(call["offsets"], expected_offsets, rtol=1e-6, atol=1e-6)

    bundle = load_fit_bundle(out)
    assert bundle["metadata"]["gene_names"] == GENES
    assert bundle["metadata"]["perturbation_names"] == ELEMENTS
    np.testing.assert_array_equal(
        bundle["beta_arrays"]["posterior_mean"],
        _gene_values(len(ELEMENTS), GENES, base=0.0),
    )
    np.testing.assert_array_equal(
        bundle["beta_arrays"]["posterior_scale"],
        _gene_values(len(ELEMENTS), GENES, base=1_000.0),
    )
    assert bundle["guide_posteriors"] == {}

    effects = pd.read_parquet(out / "element_effects.parquet")
    np.testing.assert_array_equal(effects["element"], np.repeat(ELEMENTS, len(GENES)))
    np.testing.assert_array_equal(effects["gene"], np.tile(GENES, len(ELEMENTS)))
    np.testing.assert_array_equal(
        effects["posterior_mean"], _gene_values(len(ELEMENTS), GENES, base=0.0).reshape(-1)
    )


def test_gene_block_crt_only_has_the_same_screen_wide_p_and_q_values(tmp_path, monkeypatch) -> None:
    source = tmp_path / "screen.h5mu"
    _write_cooccurring_screen(source)

    def baseline(data, control, **kwargs):
        del control, kwargs
        return SimpleNamespace(null_check=SimpleNamespace(describe=lambda: "stub null"))

    def crt_result(_baseline, data, **kwargs):
        del _baseline, kwargs
        gene_ids = np.array([GENES.index(name) for name in data.gene_names], dtype=np.float64)
        row_ids = np.arange(len(data.pert_names), dtype=np.float64)[:, None]
        p_value = 0.01 * (row_ids + 1.0) + 0.001 * (gene_ids[None, :] + 1.0)
        shape = p_value.shape
        zeros = np.zeros(shape, dtype=np.float64)
        return ChunkCRTResult(
            observed_score=1.0 - p_value,
            p_value=np.full(shape, np.nan),
            null_converged=np.ones(shape, dtype=bool),
            target_names=tuple(data.pert_names),
            gene_names=tuple(data.gene_names),
            num_resamples=0,
            parametric={
                "saddlepoint": {
                    "p_value": p_value,
                    "log_p_value": np.log(p_value),
                    "valid": np.ones(shape, dtype=bool),
                    "used_screen": np.zeros(shape, dtype=bool),
                }
            },
            null_summaries={
                "crt_null_mean": zeros,
                "crt_null_variance": np.ones(shape),
                "crt_null_skewness": zeros,
                "crt_null_excess_kurtosis": zeros,
            },
            resampling_mechanism="propensity",
            saddlepoint_only=True,
        )

    monkeypatch.setattr(api, "fit_control", _control_fit)
    monkeypatch.setattr(api, "_save_loss_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_module, "prepare_crt_baseline", baseline)
    monkeypatch.setattr(cli_module, "prepare_all_cells_propensity", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli_module, "run_crt_all_cells", crt_result)

    def run(out: Path, *, chunked: bool) -> pd.DataFrame:
        argv = [
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
            "--likelihood",
            "negbin",
            "--num-steps-control",
            "1",
            "--num-steps-betas",
            "1",
            "--backed",
            "--crt",
            "--crt-only",
            "--crt-pool",
            "all-cells",
            "--crt-mechanism",
            "propensity",
            "--crt-tail-families",
            "saddlepoint",
            "--crt-saddlepoint-only",
            "--no-save-model-params",
        ]
        if chunked:
            argv.extend(["--gene-chunk-size", "2"])
        api.main(argv)
        return pd.read_parquet(out / "element_effects.parquet").sort_values(
            ["element", "gene"], ignore_index=True
        )

    full = run(tmp_path / "full", chunked=False)
    chunked = run(tmp_path / "chunked", chunked=True)
    for column in ("crt_saddlepoint_p_value", "crt_saddlepoint_q_value"):
        np.testing.assert_allclose(chunked[column], full[column], rtol=0.0, atol=0.0)
