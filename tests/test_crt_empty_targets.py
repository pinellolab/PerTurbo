"""Targets with no assigned cell must not abort the control-anchored CRT.

The control-anchored design keeps only cells that carry exactly one perturbation and
pass the CRT's cell mask. Guides are sparse - a real TAP-seq screen had 37 of its
4,120 guides with no assigned cell before any masking at all, and 149 empty after -
so that filter empties some targets entirely. An empty target is a fact about the
screen, not a broken input: the run drops it, says so, and leaves its rows missing in
the output table rather than aborting a 126,000-cell analysis. Only an input in which
*every* target is empty is still fatal.
"""

from __future__ import annotations

import contextlib
import io
import json
import warnings
from pathlib import Path

import anndata as ad
import jax.numpy as jnp
import mudata as md
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from perturbo._internal.low_moi.design import prepare_low_moi_design
from perturbo.cli import main as cli_main
from perturbo.core import PerTurboData
from perturbo.crt import build_chunk_design
from perturbo.utils import summarize_names

N_GENES = 4


def _data(*, pert_names, pert_id, cell_mask=None, seed: int = 0) -> PerTurboData:
    labels = np.asarray(pert_id)
    n_cells = labels.shape[0]
    counts = np.random.default_rng(seed).poisson(6.0, size=(n_cells, N_GENES)).astype(np.float32)
    return PerTurboData(
        counts=jnp.asarray(counts),
        pert_id=jnp.asarray(labels),
        pert_names=list(pert_names),
        gene_names=[f"gene_{i}" for i in range(N_GENES)],
        size_factors=jnp.zeros((n_cells, 1), dtype=jnp.float32),
        cell_mask=None if cell_mask is None else jnp.asarray(np.asarray(cell_mask, dtype=bool)),
    )


def _theta() -> np.ndarray:
    return np.full(N_GENES, 5.0, dtype=np.float32)


# --- the design builder -----------------------------------------------------------


def test_a_named_target_no_cell_carries_is_dropped_and_announced(capsys) -> None:
    """The 37-empty-guides case: the name is in the screen, no cell has the label."""

    codes = np.repeat([0, 1, 2], 6).astype(np.int32)
    design = prepare_low_moi_design(
        _data(pert_names=["NTC", "t0", "t1", "t2"], pert_id=codes),
        control_perturbations=["NTC"],
        dispersion=_theta(),
    )

    assert design.empty_target_names == ("t2",)
    assert design.target_names == ("t0", "t1")
    assert design.num_targets == 2
    # The survivors keep contiguous codes and no cell points at a hole.
    np.testing.assert_array_equal(
        np.asarray(design.target_codes), np.repeat([-1, 0, 1], 6).astype(np.int32)
    )
    out = capsys.readouterr().out
    assert "Dropping 1 of 3 target perturbations with no active cell" in out
    assert "t2" in out


def test_a_target_the_cell_mask_empties_is_dropped(capsys) -> None:
    codes = np.repeat([0, 1, 2], 6).astype(np.int32)
    mask = np.ones(codes.size, dtype=bool)
    mask[codes == 2] = False

    design = prepare_low_moi_design(
        _data(pert_names=["NTC", "t0", "t1"], pert_id=codes, cell_mask=mask),
        control_perturbations=["NTC"],
        dispersion=_theta(),
    )

    assert design.empty_target_names == ("t1",)
    assert design.target_names == ("t0",)
    assert "Dropping 1 of 2 target perturbations" in capsys.readouterr().out


def test_a_target_only_multi_assigned_cells_carry_is_dropped(capsys) -> None:
    """The measured mechanism: 21.4% of the screen's cells carried two perturbations.

    ``build_chunk_design`` sets those cells aside, which is what empties a target
    that did have cells of its own.
    """

    membership = np.zeros((18, 3), dtype=np.int8)
    membership[:6, 0] = 1                     # t0 alone
    membership[6:12, 1] = 1                   # t1 alone
    membership[12:, 1] = 1                    # t2 only ever beside t1
    membership[12:, 2] = 1
    chunk = _data(pert_names=["t0", "t1", "t2"], pert_id=membership)
    controls = _data(pert_names=["NTC"], pert_id=np.zeros(20, dtype=np.int32), seed=1)

    design = build_chunk_design(None, chunk, control_data=controls, dispersion=_theta())

    assert design.empty_target_names == ("t2",)
    assert design.target_names == ("t0", "t1")
    out = capsys.readouterr().out
    assert "setting aside 6 of 18 chunk cells" in out
    assert "Dropping 1 of 3 target perturbations with no active cell" in out


def test_a_full_target_set_is_left_alone(capsys) -> None:
    codes = np.repeat([0, 1, 2], 6).astype(np.int32)
    design = prepare_low_moi_design(
        _data(pert_names=["NTC", "t0", "t1"], pert_id=codes),
        control_perturbations=["NTC"],
        dispersion=_theta(),
    )

    assert design.empty_target_names == ()
    assert design.target_names == ("t0", "t1")
    np.testing.assert_array_equal(
        np.asarray(design.target_codes), np.repeat([-1, 0, 1], 6).astype(np.int32)
    )
    assert "Dropping" not in capsys.readouterr().out


def test_dropping_a_target_moves_nothing_else() -> None:
    """A design that drops an empty target must equal one built without that name.

    This is the no-behaviour-change guarantee. An empty target owns no cell, so the
    cell axis, the nuisance design, the offsets and the control pool are untouched
    and only the target axis is renumbered.
    """

    codes = np.repeat([0, 1, 2], 6).astype(np.int32)
    with_hole = prepare_low_moi_design(
        _data(pert_names=["NTC", "t0", "t1", "t2"], pert_id=codes),
        control_perturbations=["NTC"],
        dispersion=_theta(),
    )
    without_hole = prepare_low_moi_design(
        _data(pert_names=["NTC", "t0", "t1"], pert_id=codes),
        control_perturbations=["NTC"],
        dispersion=_theta(),
    )

    assert with_hole.target_names == without_hole.target_names
    for field in (
        "counts",
        "target_codes",
        "nuisance_design",
        "offsets",
        "dispersion",
        "control_mask",
        "source_cell_indices",
    ):
        left = np.asarray(getattr(with_hole, field))
        right = np.asarray(getattr(without_hole, field))
        assert left.dtype == right.dtype, field
        np.testing.assert_array_equal(left, right, err_msg=field)


def test_the_dtype_conventions_survive_a_drop() -> None:
    codes = np.repeat([0, 1, 2], 6).astype(np.int32)
    design = prepare_low_moi_design(
        _data(pert_names=["NTC", "t0", "t1", "t2"], pert_id=codes),
        control_perturbations=["NTC"],
        dispersion=_theta(),
    )

    assert np.asarray(design.target_codes).dtype == np.int32
    for field in ("counts", "nuisance_design", "offsets", "dispersion"):
        assert np.asarray(getattr(design, field)).dtype == np.float32, field


def test_every_target_empty_is_still_fatal() -> None:
    codes = np.repeat([0, 1], 6).astype(np.int32)
    mask = np.ones(codes.size, dtype=bool)
    mask[codes == 1] = False

    with pytest.raises(ValueError, match="No target perturbation has an active cell"):
        prepare_low_moi_design(
            _data(pert_names=["NTC", "t0"], pert_id=codes, cell_mask=mask),
            control_perturbations=["NTC"],
            dispersion=_theta(),
        )


def test_a_long_dropped_list_is_truncated() -> None:
    """149 names on one line is what the run used to print. It is not a message."""

    summary = summarize_names(tuple(f"g{i}" for i in range(149)))
    assert summary.startswith("g0, g1, ")
    assert summary.endswith("(+141 more)")
    assert summarize_names(("a", "b")) == "a, b"
    assert summarize_names(()) == "none"


# --- the whole run ----------------------------------------------------------------

N_CELLS, SCREEN_GENES, N_ELEMENTS, GUIDES_PER_ELEMENT = 1600, 8, 7, 2
EMPTY_ELEMENT = 4          # elements 5 and 6 are the controls; element 4 gets no cell


def _screen(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_guides = N_ELEMENTS * GUIDES_PER_ELEMENT
    guide_to_element = np.zeros((n_guides, N_ELEMENTS), dtype=np.float32)
    guide_to_element[np.arange(n_guides), np.repeat(np.arange(N_ELEMENTS), GUIDES_PER_ELEMENT)] = 1.0

    control_guides = np.arange(GUIDES_PER_ELEMENT * (N_ELEMENTS - 2), n_guides)
    empty = set(range(GUIDES_PER_ELEMENT * EMPTY_ELEMENT, GUIDES_PER_ELEMENT * (EMPTY_ELEMENT + 1)))
    targeting_guides = np.asarray(
        [g for g in range(GUIDES_PER_ELEMENT * (N_ELEMENTS - 2)) if g not in empty]
    )
    # A third of the cells carry a control guide; nothing ever carries either guide of
    # element 4, so that element is in the screen with no cell of its own - the
    # measured TAP-seq case, where 37 of 4,120 guides had no assigned cell.
    guide = np.where(
        rng.random(N_CELLS) < 0.34,
        rng.choice(control_guides, N_CELLS),
        rng.choice(targeting_guides, N_CELLS),
    )
    assignment = np.zeros((N_CELLS, n_guides), dtype=np.float32)
    assignment[np.arange(N_CELLS), guide] = 1.0

    membership = (assignment @ guide_to_element) > 0
    assert not membership[:, EMPTY_ELEMENT].any()
    log_mu = (
        rng.normal(1.8, 0.3, size=SCREEN_GENES)[None, :]
        + rng.normal(0.0, 0.2, size=N_CELLS)[:, None]
    )
    lfc = np.zeros((N_ELEMENTS, SCREEN_GENES))
    lfc[0, 1] = -1.5
    log_mu = log_mu + membership.astype(float) @ lfc
    theta = 6.0
    counts = rng.negative_binomial(theta, theta / (theta + np.exp(log_mu))).astype(np.float32)
    names = [f"elem_{e}" for e in range(N_ELEMENTS - 2)] + ["non-targeting|1", "non-targeting|2"]
    return counts, assignment, guide_to_element, names


def _write(path: Path, counts, assignment, guide_to_element, names) -> None:
    obs = pd.DataFrame(
        {"total_umis": counts.sum(axis=1).astype(np.int64) + 50},
        index=[f"cell{i}" for i in range(N_CELLS)],
    )
    gene = ad.AnnData(
        X=sp.csr_matrix(counts),
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(SCREEN_GENES)]),
    )
    guide = ad.AnnData(
        X=sp.csr_matrix(assignment),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"guide{i}" for i in range(assignment.shape[1])]),
    )
    guide.varm["element_map"] = sp.csr_matrix(guide_to_element)
    guide.uns["element_names"] = np.array(names, dtype=object)
    md.MuData({"gene": gene, "guide": guide}).write_h5mu(path)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One real control-anchored run over a screen holding one empty element."""

    root = tmp_path_factory.mktemp("crt_empty_targets")
    _write(root / "screen.h5mu", *_screen())
    out = root / "out"
    log = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(log):
        warnings.simplefilter("ignore")
        cli_main(
            [
                "--input", str(root / "screen.h5mu"),
                "--out-dir", str(out),
                "--modality-key", "gene",
                "--perturbation-modality-key", "guide",
                "--perturbation-element-varm-key", "element_map",
                "--perturbation-element-names-uns-key", "element_names",
                "--control-substring", "non-targeting",
                "--library-size-key", "total_umis",
                "--size-factor-mode", "observed",
                "--likelihood", "nb",
                "--num-steps-control", "300",
                "--num-steps-betas", "50",
                "--no-save-model-params",
                "--crt", "--crt-only",
                "--crt-pool", "control-anchored",
                "--crt-mechanism", "propensity",
                "--crt-tail-families", "saddlepoint",
                "--crt-saddlepoint-only",
                "--crt-polish-baseline",
                "--crt-allow-unconverged-baseline",
            ]
        )
    return pd.read_parquet(out / "element_effects.parquet"), out, log.getvalue()


def test_the_run_finishes_and_keeps_every_element_row(run) -> None:
    frame, _, _ = run
    expected = [f"elem_{e}" for e in range(N_ELEMENTS - 2)] + ["non-targeting|1", "non-targeting|2"]
    assert sorted(frame["element"].astype(str).unique()) == sorted(expected)
    # The empty element keeps its whole row block, so a consumer indexing on a fixed
    # element set still finds it.
    assert len(frame[frame["element"] == f"elem_{EMPTY_ELEMENT}"]) == SCREEN_GENES


def test_the_empty_element_is_missing_not_zero(run) -> None:
    frame, _, _ = run
    empty = frame[frame["element"] == f"elem_{EMPTY_ELEMENT}"]
    assert empty["crt_saddlepoint_p_value"].isna().all()
    assert empty["crt_z_value"].isna().all()
    assert not empty["crt_saddlepoint_valid"].astype(bool).any()


def test_the_other_elements_were_tested_and_the_planted_effect_is_found(run) -> None:
    frame, _, _ = run
    tested = frame[frame["element"].astype(str).str.startswith("elem_")]
    tested = tested[tested["element"] != f"elem_{EMPTY_ELEMENT}"]
    assert np.isfinite(tested["crt_saddlepoint_p_value"].to_numpy(float)).all()
    planted = tested[(tested["element"] == "elem_0") & (tested["gene"] == "gene_1")]
    assert float(planted["crt_saddlepoint_p_value"].iloc[0]) < 0.01


def test_the_run_says_out_loud_what_it_dropped(run) -> None:
    _, _, log = run
    assert "Dropping 1 of 5 target perturbations with no active cell" in log
    assert f"elem_{EMPTY_ELEMENT}" in log
    assert "had no cell the control-anchored test could use" in log


def test_the_run_records_the_dropped_element_in_its_metadata(run) -> None:
    _, out, _ = run
    meta = json.loads((Path(out) / "crt_metadata.json").read_text())
    assert meta["targets_without_assigned_cells"] == 1
    assert meta["targets_without_assigned_cells_names"] == [f"elem_{EMPTY_ELEMENT}"]
