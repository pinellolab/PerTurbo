"""The CLI's all-cells (high-MOI) CRT pool on a small planted screen."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from perturbo.cli import main as cli_main


def _write_high_moi_screen(path: Path, *, n_cells=1500, n_genes=16, n_elements=10, guides_per_element=2, seed=0) -> None:
    rng = np.random.default_rng(seed)
    n_guides = n_elements * guides_per_element
    guide_to_element = np.zeros((n_guides, n_elements), dtype=np.float32)
    guide_to_element[np.arange(n_guides), np.repeat(np.arange(n_elements), guides_per_element)] = 1.0
    assignment = (rng.random((n_cells, n_guides)) < 0.2).astype(np.float32)
    membership = (assignment @ guide_to_element) > 0
    log_mu = rng.normal(1.5, 0.5, size=n_genes)[None, :] + rng.normal(0, 0.25, size=n_cells)[:, None]
    lfc = np.zeros((n_elements, n_genes)); lfc[0, 1] = -1.5; lfc[1, 2] = -1.5   # planted: element 0 on gene 1, element 1 on gene 2
    log_mu = log_mu + membership.astype(float) @ lfc
    theta = 4.0
    counts = rng.negative_binomial(theta, theta / (theta + np.exp(log_mu))).astype(np.float32)
    gene = ad.AnnData(
        X=sp.csr_matrix(counts),
        obs=pd.DataFrame({"total_umis": counts.sum(axis=1).astype(np.int64) + 50}, index=[f"cell{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )
    guide = ad.AnnData(X=sp.csr_matrix(assignment), obs=gene.obs.copy(), var=pd.DataFrame(index=[f"guide{i}" for i in range(n_guides)]))
    guide.varm["guide_intended_target_pairs"] = sp.csr_matrix(guide_to_element)
    guide.uns["intended_targets"] = np.array([f"elem_{e}" for e in range(n_elements - 2)] + ["non-targeting_a", "non-targeting_b"], dtype=object)
    guide.var["is_non_targeting"] = np.repeat(np.arange(n_elements) >= n_elements - 2, guides_per_element)
    md.MuData({"gene": gene, "guide": guide}).write_h5mu(path)


def _run(tmp_path: Path, *extra: str) -> pd.DataFrame:
    screen = tmp_path / "screen.h5mu"
    _write_high_moi_screen(screen)
    out = tmp_path / "out"
    cli_main([
        "--input", str(screen), "--out-dir", str(out),
        "--modality-key", "gene", "--perturbation-modality-key", "guide",
        "--perturbation-element-varm-key", "guide_intended_target_pairs",
        "--perturbation-element-names-uns-key", "intended_targets",
        "--library-size-key", "total_umis", "--size-factor-mode", "observed", "--likelihood", "nb",
        "--num-steps-control", "200", "--num-steps-betas", "50", "--no-save-model-params",
        "--crt", "--crt-mechanism", "propensity", "--crt-tail-families", "saddlepoint", "--crt-saddlepoint-only",
        "--crt-allow-unconverged-baseline", *extra,
    ])
    return pd.read_parquet(out / "element_effects.parquet")


def test_all_cells_pool_is_chosen_for_element_designs_and_finds_planted_effects(tmp_path):
    frame = _run(tmp_path)
    assert "crt_saddlepoint_p_value" in frame.columns and "crt_saddlepoint_q_value" in frame.columns
    tested = frame[np.isfinite(frame["crt_saddlepoint_p_value"])]
    assert tested["element"].nunique() == 10  # every element, non-targeting ones included, is a marginal test
    planted = frame[(frame["element"] == "elem_0") & (frame["gene"] == "gene_1")]["crt_saddlepoint_p_value"].iloc[0]
    planted2 = frame[(frame["element"] == "elem_1") & (frame["gene"] == "gene_2")]["crt_saddlepoint_p_value"].iloc[0]
    assert planted < 1e-3 and planted2 < 1e-3
    # The two non-targeting elements give 32 null pairs on this fixture, far too few to
    # pin a far-tail bound: a single p-value near 1e-4 is ordinary sampling. What the
    # test can hold is that the null is not systematically small.
    null = frame[frame["element"].str.startswith("non-targeting")]["crt_saddlepoint_p_value"].dropna()
    assert null.min() > 1e-6
    assert null.median() > 0.05
    assert (null < 0.01).mean() < 0.2
    assert frame["crt_p_value"].isna().all()  # no resamples were drawn


def test_all_cells_pool_refuses_resampling_configurations(tmp_path):
    with pytest.raises(ValueError, match="all-cells runs the propensity saddlepoint"):
        _run(tmp_path, "--crt-pool", "all-cells", "--crt-mechanism", "permutation", "--crt-tail-families", "skew_normal")


def test_control_anchored_pool_on_an_element_design_sets_aside_multi_element_cells(tmp_path, capsys):
    """An element map is no longer refused by the control-anchored test. The assignment is
    collapsed to elements and a cell carrying more than one is set aside, not reinterpreted;
    on this high-MOI fixture that is most cells, and the run says so."""
    frame = _run(tmp_path, "--crt-pool", "control-anchored", "--control-substring", "non-targeting")
    out = capsys.readouterr().out
    assert "carry only control guides" in out          # the pool is the pure controls
    assert "more than one perturbation" in out          # and the set-aside count was reported
    assert "crt_saddlepoint_p_value" in frame.columns
    tested = frame[np.isfinite(frame["crt_saddlepoint_p_value"])]["element"].unique()
    assert not any(str(e).startswith("non-targeting") for e in tested)  # controls are the pool, not targets


def test_crt_only_skips_stage_two_and_keeps_the_crt_columns(tmp_path):
    frame = _run(tmp_path, "--crt-only")
    assert np.isfinite(frame["crt_saddlepoint_p_value"]).sum() > 0
    assert frame["loc"].isna().all() if "loc" in frame.columns else True


def test_all_cells_pool_survives_auto_chunking(tmp_path):
    # A chunk-size cap far below the cell count forces the chunked stage-two path;
    # the all-cells CRT must still run once over every cell.
    frame = _run(tmp_path, "--max-chunk-size", "700", "--crt-only")
    tested = frame[np.isfinite(frame["crt_saddlepoint_p_value"])]
    assert tested["element"].nunique() == 10
    planted = frame[(frame["element"] == "elem_0") & (frame["gene"] == "gene_1")]["crt_saddlepoint_p_value"].iloc[0]
    assert planted < 1e-3


def test_auto_chunked_crt_only_runs_the_all_cells_crt_once(tmp_path, monkeypatch):
    """Auto-chunking must not send a --crt-only run through the unchunked branch a second time."""
    import perturbo.cli as cli_module

    original = cli_module.run_crt_all_cells
    calls: list[int] = []

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cli_module, "run_crt_all_cells", counting)
    frame = _run(tmp_path, "--max-chunk-size", "700", "--crt-only")
    assert len(calls) == 1
    assert frame["crt_saddlepoint_p_value"].notna().any()
