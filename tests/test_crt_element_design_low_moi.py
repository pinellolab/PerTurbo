"""The control-anchored CRT on a low-MOI screen that arrives with a guide-to-element map.

This is how the IGVF pipeline hands every screen to PerTurbo: guides grouped into
elements by an element map, control elements named 'non-targeting|...'. The
control-anchored test must (a) collapse the assignment to elements, (b) use as its
pool the cells that carry nothing but control guides, (c) set aside cells that carry
two elements rather than reinterpret them, and (d) give the same p-values as the
label-per-cell path on the cells both can use.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from perturbo.cli import main as cli_main

N_CELLS, N_GENES, N_ELEMENTS, GUIDES_PER_ELEMENT = 2400, 12, 8, 2   # elements 6 and 7 are controls


def _design(seed=0):
    rng = np.random.default_rng(seed)
    n_guides = N_ELEMENTS * GUIDES_PER_ELEMENT
    guide_to_element = np.zeros((n_guides, N_ELEMENTS), dtype=np.float32)
    guide_to_element[np.arange(n_guides), np.repeat(np.arange(N_ELEMENTS), GUIDES_PER_ELEMENT)] = 1.0
    # One guide per cell, controls over-represented so the pool is a few hundred cells;
    # then a second guide from a different element in 12% of cells, and a control guide
    # beside a targeting one in another 4%.
    first = np.where(rng.random(N_CELLS) < 0.3, rng.integers(12, n_guides, N_CELLS), rng.integers(0, 12, N_CELLS))
    assignment = np.zeros((N_CELLS, n_guides), dtype=np.float32)
    assignment[np.arange(N_CELLS), first] = 1.0
    second = rng.random(N_CELLS) < 0.12
    other = (first + rng.integers(2, n_guides - 2, N_CELLS)) % n_guides
    assignment[np.flatnonzero(second), other[second]] = 1.0
    mixed = (rng.random(N_CELLS) < 0.04) & (first < 12)
    assignment[np.flatnonzero(mixed), rng.integers(12, n_guides, int(mixed.sum()))] = 1.0
    membership = (assignment @ guide_to_element) > 0
    log_mu = rng.normal(1.6, 0.4, size=N_GENES)[None, :] + rng.normal(0, 0.2, size=N_CELLS)[:, None]
    lfc = np.zeros((N_ELEMENTS, N_GENES)); lfc[0, 1] = -1.4; lfc[2, 3] = 1.0
    log_mu = log_mu + membership.astype(float) @ lfc
    theta = 5.0
    counts = rng.negative_binomial(theta, theta / (theta + np.exp(log_mu))).astype(np.float32)
    names = [f"elem_{e}" for e in range(N_ELEMENTS - 2)] + ["non-targeting|1", "non-targeting|2"]
    return counts, assignment, guide_to_element, membership, names


def _write_mudata(path: Path, counts, assignment, guide_to_element, names):
    obs = pd.DataFrame({"total_umis": counts.sum(axis=1).astype(np.int64) + 50}, index=[f"cell{i}" for i in range(N_CELLS)])
    gene = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(N_GENES)]))
    guide = ad.AnnData(X=sp.csr_matrix(assignment), obs=obs.copy(), var=pd.DataFrame(index=[f"guide{i}" for i in range(assignment.shape[1])]))
    guide.varm["element_map"] = sp.csr_matrix(guide_to_element)
    guide.uns["element_names"] = np.array(names, dtype=object)
    md.MuData({"gene": gene, "guide": guide}).write_h5mu(path)


def _write_labelled_anndata(path: Path, counts, membership, names):
    """The label-per-cell view of the same screen, with exactly the semantics the
    element-mapped control-anchored path applies: a cell with one targeting element
    is labelled with it (whether or not a control guide rides along), a cell with no
    targeting element but a control guide is a control, and a cell with two or more
    targeting elements is set aside."""
    targeting = membership[:, :N_ELEMENTS - 2]
    n_targeting = targeting.sum(axis=1)
    is_control = (n_targeting == 0) & membership[:, N_ELEMENTS - 2:].any(axis=1)
    keep = (n_targeting == 1) | is_control
    label = np.where(is_control, "non-targeting", np.array(names, dtype=object)[targeting.argmax(axis=1)])
    obs = pd.DataFrame({"pert": label[keep], "total_umis": counts[keep].sum(axis=1).astype(np.int64) + 50},
                       index=[f"cell{i}" for i in np.flatnonzero(keep)])
    ad.AnnData(X=sp.csr_matrix(counts[keep]), obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(N_GENES)])).write_h5ad(path)
    return keep


COMMON = ["--library-size-key", "total_umis", "--size-factor-mode", "observed", "--likelihood", "nb",
          "--num-steps-control", "300", "--num-steps-betas", "50", "--no-save-model-params",
          "--crt", "--crt-only", "--crt-mechanism", "propensity", "--crt-tail-families", "saddlepoint",
          "--crt-saddlepoint-only", "--crt-polish-baseline", "--crt-allow-unconverged-baseline",
          "--control-substring", "non-targeting"]


@pytest.fixture(scope="module")
def screen(tmp_path_factory):
    root = tmp_path_factory.mktemp("element_low_moi")
    counts, assignment, g2e, membership, names = _design()
    _write_mudata(root / "screen.h5mu", counts, assignment, g2e, names)
    kept = _write_labelled_anndata(root / "screen.h5ad", counts, membership, names)
    return dict(root=root, membership=membership, kept=kept, names=names)


def _run(root: Path, name: str, *args) -> tuple[pd.DataFrame, str]:
    out = root / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cli_main([*args, "--out-dir", str(out), *COMMON])
    return pd.read_parquet(out / "element_effects.parquet"), str(out)


def test_element_design_matches_the_label_per_cell_path(screen, capsys):
    root = screen["root"]
    mapped, _ = _run(root, "mapped", "--input", str(root / "screen.h5mu"), "--modality-key", "gene",
                     "--perturbation-modality-key", "guide", "--perturbation-element-varm-key", "element_map",
                     "--perturbation-element-names-uns-key", "element_names", "--crt-pool", "control-anchored")
    out = capsys.readouterr().out
    # Control elements are excluded as targets before the count, so a cell with a
    # control guide beside one targeting guide is a single-perturbation cell here.
    n_multi = int((screen["membership"][:, :N_ELEMENTS - 2].sum(axis=1) > 1).sum())
    assert f"{n_multi:,} analysed cells carried more than one perturbation" in out
    labelled, _ = _run(root, "labelled", "--input", str(root / "screen.h5ad"), "--perturbation-key", "pert",
                       "--crt-pool", "control-anchored")
    key = ["element", "gene"]
    a = mapped[~mapped["element"].astype(str).str.startswith("non-targeting")].set_index(key)["crt_saddlepoint_p_value"].sort_index()
    b = labelled[~labelled["element"].astype(str).str.startswith("non-targeting")].set_index(key)["crt_saddlepoint_p_value"].sort_index()
    assert list(a.index) == list(b.index) and len(a) == 6 * N_GENES
    np.testing.assert_allclose(np.log(a.to_numpy()), np.log(b.to_numpy()), rtol=0, atol=0.05)
    planted = a.loc[("elem_0", "gene_1")], a.loc[("elem_2", "gene_3")]
    assert planted[0] < 1e-3 and planted[1] < 1e-2


def test_auto_picks_the_control_pool_and_warns_about_a_thin_one(screen, capsys):
    root = screen["root"]
    with pytest.warns(UserWarning, match="carry nothing but control guides"):
        cli_main(["--input", str(root / "screen.h5mu"), "--modality-key", "gene", "--perturbation-modality-key", "guide",
                  "--perturbation-element-varm-key", "element_map", "--perturbation-element-names-uns-key", "element_names",
                  "--out-dir", str(root / "auto"), *COMMON])
    out = capsys.readouterr().out
    assert "--crt-pool auto resolved to 'control-anchored'" in out
    assert "median 1.00 guides per cell" in out
    meta = json.loads((root / "auto" / "crt_metadata.json").read_text())
    assert meta["pool"] == "control-anchored" and meta["pool_requested"] == "auto"
    assert meta["measured"]["median_guides_per_cell"] == 1.0
    assert meta["multi_assignment_cells_set_aside"] == int((screen["membership"][:, :N_ELEMENTS - 2].sum(axis=1) > 1).sum())
