"""A batch covariate the control pool cannot identify has to be said out loud.

Measured on a 126,154-cell TAP-seq screen with 14 sequencing lanes in
``obs["batch"]``: all 30 non-targeting guides had been prepared in one lane, so
2,033 of the 2,049 control-only cells sat there and the other 13 lanes held between
zero and four each. ``--batch-covariate batch`` was supplied and did not help.
Control-anchored, every lane effect became an apparent knockdown - 1,241 enhancer
guides over 50 Mb "knocked down" MRPL13. All-cells, the non-targeting guides
themselves returned 21.9% of tests at p < 0.05 while the targeting guides, spread
over every lane, were calibrated. Inside the one lane both pools were exactly
calibrated at 4.8%.

Nobody caught it because no output described how the control cells sat across the
batch levels. These tests cover the diagnostic that now does.
"""

from __future__ import annotations

import contextlib
import io
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
from perturbo.crt import (
    CONTROL_BATCH_MIN_CELLS_PER_LEVEL,
    report_control_batch_confinement,
    summarize_control_batch_confinement,
)


def _masks(control_counts, level_sizes):
    """Per-cell control mask and batch labels from per-level (controls, cells)."""

    controls: list[bool] = []
    labels: list[str] = []
    for index, (n_control, n_cells) in enumerate(zip(control_counts, level_sizes, strict=True)):
        assert n_control <= n_cells
        level = f"lane{index}"
        controls.extend([True] * n_control + [False] * (n_cells - n_control))
        labels.extend([level] * n_cells)
    return np.asarray(controls, dtype=bool), np.asarray(labels, dtype=object)


def _summary(control_counts, level_sizes, **kwargs):
    return summarize_control_batch_confinement(
        *_masks(control_counts, level_sizes), batch_covariate="batch", **kwargs
    )


# --- the rule ---------------------------------------------------------------------


def test_controls_spread_over_every_level_are_not_confined() -> None:
    summary = _summary([30, 30, 30, 30], [500, 500, 500, 500])

    assert summary.confined is False
    assert summary.num_batch_levels == 4
    assert summary.num_represented_levels == 4
    assert summary.control_share_in_top_level == pytest.approx(0.25)
    assert summary.top_level_screen_share == pytest.approx(0.25)
    assert summary.warning(pool="control-anchored") is None


def test_the_measured_tap_seq_shape_is_confined() -> None:
    """2,033 of 2,049 controls in one lane of 14; that lane is 35% controls."""

    control_counts = [2033] + [4, 4, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    level_sizes = [5809] + [9257] * 13
    summary = _summary(control_counts, level_sizes)

    assert summary.confined is True
    assert summary.num_control_cells == 2049
    assert summary.num_batch_levels == 14
    # Fourteen levels hold a control cell; one holds enough to fit a coefficient on.
    assert summary.num_levels_with_controls == 10
    assert summary.num_represented_levels == 1
    assert summary.top_level == "lane0"
    assert summary.control_share_in_top_level == pytest.approx(2033 / 2049, abs=1e-4)
    assert summary.top_level_screen_share == pytest.approx(0.0459, abs=1e-3)
    assert summary.control_share_of_top_level == pytest.approx(0.35, abs=1e-2)


def test_a_scatter_of_thin_levels_does_not_count_as_coverage() -> None:
    """The share clause alone would pass this; the coverage clause catches it."""

    # 88% in the top level, so under the 90% share threshold.
    summary = _summary([440, 15, 15, 15, 15], [2000, 2000, 2000, 2000, 2000])

    assert summary.control_share_in_top_level == pytest.approx(0.88)
    assert summary.num_represented_levels == 1
    assert summary.confined is True


def test_two_well_populated_levels_are_enough() -> None:
    summary = _summary([440, 60], [2000, 2000])

    assert summary.control_share_in_top_level == pytest.approx(0.88)
    assert summary.num_represented_levels == 2
    assert summary.confined is False


def test_the_screen_share_guard_belongs_to_the_share_clause_alone() -> None:
    """94.6% of controls in one level means different things at both screen shares."""

    # The top level is 90% of the screen, and the other level has controls of its
    # own: a lopsided pool, but every level's effect is still identified.
    assert _summary([440, 25], [4500, 500]).confined is False
    # The same control split with the top level a tenth of the screen: 90% of the
    # cells sit in a level the controls barely enter.
    assert _summary([440, 25], [500, 4500]).confined is True


def test_one_represented_level_is_flagged_even_when_it_is_most_of_the_screen() -> None:
    """The share clause would not fire here; the coverage clause has to."""

    summary = _summary([100, 0], [9000, 1000])

    assert summary.top_level_screen_share == pytest.approx(0.9)
    assert summary.num_represented_levels == 1
    # The second level holds a thousand analysed cells and no control cell at all,
    # so its coefficient is not identified from controls whatever its size.
    assert summary.confined is True


def test_a_single_batch_level_cannot_be_confined() -> None:
    summary = _summary([100], [1000])

    assert summary.num_batch_levels == 1
    assert summary.confined is False


def test_a_small_even_dataset_does_not_warn_just_for_being_small() -> None:
    """No level clears the 20-control bar, which is a size fact, not a design one."""

    summary = _summary([5, 5, 5], [50, 50, 50])

    assert summary.num_represented_levels == 0
    assert summary.confined is False


def test_no_control_cell_at_all_reports_without_flagging() -> None:
    summary = _summary([0, 0], [100, 100])

    assert summary.num_control_cells == 0
    assert summary.confined is False
    assert "nothing to distribute" in summary.describe()


def test_the_bar_is_a_parameter() -> None:
    assert _summary([5, 5, 5], [50, 50, 50], min_control_cells_per_level=4).confined is False
    confined = _summary([30, 5, 5], [200, 200, 200], min_control_cells_per_level=4)
    assert confined.num_represented_levels == 3
    assert confined.confined is False
    assert _summary([30, 5, 5], [200, 200, 200]).confined is True


def test_unused_categories_are_not_counted_as_levels() -> None:
    labels = pd.Categorical(
        ["a"] * 40 + ["b"] * 40, categories=["a", "b", "never_sequenced"]
    )
    summary = summarize_control_batch_confinement(
        np.repeat([True, False], 40), labels, batch_covariate="batch"
    )

    assert summary.num_batch_levels == 2


def test_a_length_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError, match="one entry per analysed cell"):
        summarize_control_batch_confinement(np.ones(5, dtype=bool), np.zeros(4))


# --- what it says -----------------------------------------------------------------


def test_the_line_is_printed_whether_or_not_it_warns(capsys) -> None:
    summary = report_control_batch_confinement(
        *_masks([30, 30], [300, 300]), batch_covariate="batch", pool="control-anchored"
    )
    out = capsys.readouterr().out

    assert summary.confined is False
    assert "[perturbo] controls: 60 cells across 2 'batch' levels" in out
    assert "WARNING" not in out


def test_the_control_anchored_warning_names_the_identification_failure(capsys) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = report_control_batch_confinement(
            *_masks([98, 2], [1000, 3000]), batch_covariate="batch", pool="control-anchored"
        )
    out = capsys.readouterr().out

    assert summary.confined is True
    assert "[perturbo] controls: 100 cells across 2 'batch' levels" in out
    assert "[perturbo] WARNING: Control cells are confined to one batch level" in out
    assert "effectively a single batch" in out
    assert "cannot identify the other levels' effects from controls alone" in out
    # Same text on both channels, so a library caller sees it too.
    assert len(caught) == 1
    assert "effectively a single batch" in str(caught[0].message)


def test_the_all_cells_warning_is_about_calibration_instead(capsys) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report_control_batch_confinement(
            *_masks([98, 2], [1000, 3000]), batch_covariate="batch", pool="all-cells"
        )
    out = capsys.readouterr().out

    assert "Non-targeting calibration checks are confounded with batch" in out
    assert "read within that level" in out
    assert "effectively a single batch" not in out


def test_the_metadata_keys_are_the_numbers_in_the_line() -> None:
    meta = _summary([98, 2], [1000, 3000]).as_metadata()

    assert meta["control_batch_confined"] is True
    assert meta["control_batch_levels"] == 2
    assert meta["control_top_batch_level"] == "lane0"
    assert meta["control_top_batch_share"] == pytest.approx(0.98)
    assert meta["control_top_batch_screen_share"] == pytest.approx(0.25)
    assert meta["control_top_batch_control_fraction"] == pytest.approx(0.098)
    assert meta["control_batch_min_cells_per_level"] == CONTROL_BATCH_MIN_CELLS_PER_LEVEL
    # JSON-serializable as written, with no numpy scalars left in it.
    assert json.loads(json.dumps(meta))["control_batch_confined"] is True


# --- a whole run ------------------------------------------------------------------

N_CELLS, SCREEN_GENES, N_ELEMENTS, GUIDES_PER_ELEMENT = 400, 4, 4, 2
N_LANES = 4


def _screen(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_guides = N_ELEMENTS * GUIDES_PER_ELEMENT
    guide_to_element = np.zeros((n_guides, N_ELEMENTS), dtype=np.float32)
    guide_to_element[np.arange(n_guides), np.repeat(np.arange(N_ELEMENTS), GUIDES_PER_ELEMENT)] = 1.0

    control_guides = np.arange(GUIDES_PER_ELEMENT * (N_ELEMENTS - 2), n_guides)
    targeting_guides = np.arange(GUIDES_PER_ELEMENT * (N_ELEMENTS - 2))
    is_control = rng.random(N_CELLS) < 0.35
    guide = np.where(
        is_control,
        rng.choice(control_guides, N_CELLS),
        rng.choice(targeting_guides, N_CELLS),
    )
    assignment = np.zeros((N_CELLS, n_guides), dtype=np.float32)
    assignment[np.arange(N_CELLS), guide] = 1.0

    # The same cells, two batch columns. Spread: the lane is independent of the
    # guide. Confined: every control cell but two sits in lane0, while the
    # targeting cells are spread over all four lanes - the measured shape.
    lane_spread = rng.integers(0, N_LANES, size=N_CELLS)
    lane_confined = np.where(is_control, 0, rng.integers(0, N_LANES, size=N_CELLS))
    leaked = np.flatnonzero(is_control)[:2]
    lane_confined[leaked] = 1

    log_mu = (
        rng.normal(1.9, 0.2, size=SCREEN_GENES)[None, :]
        + rng.normal(0.0, 0.2, size=N_CELLS)[:, None]
    )
    theta = 6.0
    counts = rng.negative_binomial(theta, theta / (theta + np.exp(log_mu))).astype(np.float32)
    names = [f"elem_{e}" for e in range(N_ELEMENTS - 2)] + ["non-targeting|1", "non-targeting|2"]
    return counts, assignment, guide_to_element, names, lane_spread, lane_confined


def _write(path: Path) -> None:
    counts, assignment, guide_to_element, names, lane_spread, lane_confined = _screen()
    obs = pd.DataFrame(
        {
            "total_umis": counts.sum(axis=1).astype(np.int64) + 50,
            "batch_spread": pd.Categorical([f"lane{i}" for i in lane_spread]),
            "batch_confined": pd.Categorical([f"lane{i}" for i in lane_confined]),
        },
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


def _run(root: Path, name: str, batch_covariate: str | None):
    out = root / name
    log = io.StringIO()
    argv = [
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
        "--num-steps-control", "30",
        "--num-steps-betas", "10",
        "--no-save-model-params",
        "--crt", "--crt-only",
        "--crt-pool", "control-anchored",
        "--crt-mechanism", "propensity",
        "--crt-tail-families", "saddlepoint",
        "--crt-saddlepoint-only",
        "--crt-polish-baseline",
        "--crt-allow-unconverged-baseline",
    ]
    if batch_covariate is not None:
        argv += ["--batch-covariate", batch_covariate]
    with warnings.catch_warnings(), contextlib.redirect_stdout(log):
        warnings.simplefilter("ignore")
        cli_main(argv)
    return out, log.getvalue()


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("control_batch_confinement")
    _write(root / "screen.h5mu")
    return {
        "spread": _run(root, "spread", "batch_spread"),
        "confined": _run(root, "confined", "batch_confined"),
        "no_batch": _run(root, "no_batch", None),
    }


def _metadata(out: Path) -> dict:
    return json.loads((out / "crt_metadata.json").read_text())


def test_spread_controls_get_the_line_and_no_warning(runs) -> None:
    out, log = runs["spread"]

    assert "[perturbo] controls:" in log
    assert "'batch_spread' levels" in log
    assert "Control cells are confined to one batch level" not in log
    assert _metadata(out)["control_batch_confined"] is False


def test_spread_controls_record_every_level(runs) -> None:
    meta = _metadata(runs["spread"][0])

    assert meta["control_batch_covariate"] == "batch_spread"
    assert meta["control_batch_levels"] == N_LANES
    assert meta["control_batch_represented_levels"] == N_LANES
    assert meta["control_top_batch_share"] < 0.5
    assert 0 < meta["control_top_batch_screen_share"] < 1


def test_confined_controls_warn_in_the_run_log(runs) -> None:
    _, log = runs["confined"]

    assert "[perturbo] controls:" in log
    assert "[perturbo] WARNING: Control cells are confined to one batch level" in log
    assert "effectively a single batch" in log
    assert "cannot identify the other levels' effects from controls alone" in log


def test_confined_controls_are_flagged_in_the_crt_metadata(runs) -> None:
    meta = _metadata(runs["confined"][0])

    assert meta["control_batch_confined"] is True
    assert meta["control_batch_covariate"] == "batch_confined"
    assert meta["control_batch_levels"] == N_LANES
    assert meta["control_batch_represented_levels"] == 1
    assert meta["control_top_batch_level"] == "lane0"
    assert meta["control_top_batch_share"] > 0.9
    assert meta["control_top_batch_screen_share"] < 0.5
    assert meta["control_batch_control_cells"] == meta["control_top_batch_cells"] + 2


def test_the_covariate_metadata_carries_it_too(runs) -> None:
    """A run without the CRT writes no crt_metadata.json; this file it still writes."""

    meta = json.loads((runs["confined"][0] / "covariate_metadata.json").read_text())

    assert meta["batch_covariate_requested"] == "batch_confined"
    assert meta["control_batch_confined"] is True
    assert meta["control_top_batch_level"] == "lane0"


def test_without_a_batch_covariate_the_diagnostic_is_skipped(runs) -> None:
    out, log = runs["no_batch"]

    assert "[perturbo] controls:" not in log
    assert "Control cells are confined" not in log
    meta = _metadata(out)
    assert not [key for key in meta if key.startswith("control_batch_")]
    assert not [key for key in meta if key.startswith("control_top_batch_")]
    # And the run still finished the test it was asked for.
    assert meta["pool"] == "control-anchored"
    assert not (out / "covariate_metadata.json").exists()


# --- the all-cells pool, end to end -----------------------------------------------

ALL_CELLS_LANES = ("lane0", "lane1", "lane2")
UNSAMPLED_LANE = "lane2"


def _write_all_cells_screen(path: Path, *, n_cells: int = 900, n_genes: int = 6, seed: int = 3) -> None:
    """A high-MOI screen whose control cells never enter one lane.

    ``lane2`` holds a fifth of the analysed cells and not one control cell, so
    the stage-one fit cannot identify it and only the all-cells refit can. This
    is the TAP-seq shape reduced to something a test can run: a lane the
    non-targeting guides were simply never prepared in.
    """

    rng = np.random.default_rng(seed)
    n_elements, guides_per_element = 6, 2
    n_guides = n_elements * guides_per_element
    guide_to_element = np.zeros((n_guides, n_elements), dtype=np.float32)
    guide_to_element[np.arange(n_guides), np.repeat(np.arange(n_elements), guides_per_element)] = 1.0
    targeting_guides = np.arange(guides_per_element * (n_elements - 2))
    control_guides = np.arange(guides_per_element * (n_elements - 2), n_guides)

    is_control = rng.random(n_cells) < 0.3
    # Controls sit in the first two lanes only; targeting cells use all three.
    lane = np.where(
        is_control,
        rng.integers(0, len(ALL_CELLS_LANES) - 1, size=n_cells),
        rng.integers(0, len(ALL_CELLS_LANES), size=n_cells),
    )

    assignment = np.zeros((n_cells, n_guides), dtype=np.float32)
    for cell in range(n_cells):
        if is_control[cell]:
            assignment[cell, rng.choice(control_guides, size=2, replace=False)] = 1.0
        else:
            # High MOI: several targeting guides per cell, so the guide count
            # itself carries information the selection model has to absorb.
            drawn = rng.choice(targeting_guides, size=rng.integers(2, 5), replace=False)
            assignment[cell, drawn] = 1.0

    log_mu = (
        rng.normal(1.8, 0.3, size=n_genes)[None, :]
        + rng.normal(0.0, 0.2, size=n_cells)[:, None]
        # A real lane effect, so dropping the lane's column would be visible.
        + np.array([0.0, 0.3, -0.4])[lane][:, None]
    )
    theta = 5.0
    counts = rng.negative_binomial(theta, theta / (theta + np.exp(log_mu))).astype(np.float32)

    obs = pd.DataFrame(
        {
            "total_umis": counts.sum(axis=1).astype(np.int64) + 50,
            "lane": pd.Categorical([ALL_CELLS_LANES[i] for i in lane]),
        },
        index=[f"cell{i}" for i in range(n_cells)],
    )
    gene = ad.AnnData(
        X=sp.csr_matrix(counts),
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )
    guide = ad.AnnData(
        X=sp.csr_matrix(assignment),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"guide{i}" for i in range(n_guides)]),
    )
    guide.varm["element_map"] = sp.csr_matrix(guide_to_element)
    guide.uns["element_names"] = np.array(
        [f"elem_{e}" for e in range(n_elements - 2)] + ["non-targeting_a", "non-targeting_b"],
        dtype=object,
    )
    md.MuData({"gene": gene, "guide": guide}).write_h5mu(path)


@pytest.fixture(scope="module")
def all_cells_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("all_cells_unsampled_lane")
    _write_all_cells_screen(root / "screen.h5mu")
    out = root / "out"
    log = io.StringIO()
    argv = [
        "--input", str(root / "screen.h5mu"),
        "--out-dir", str(out),
        "--device", "cpu",
        "--modality-key", "gene",
        "--perturbation-modality-key", "guide",
        "--perturbation-element-varm-key", "element_map",
        "--perturbation-element-names-uns-key", "element_names",
        "--control-substring", "non-targeting",
        "--library-size-key", "total_umis",
        "--size-factor-mode", "observed",
        "--likelihood", "nb",
        "--batch-covariate", "lane",
        "--num-steps-control", "60",
        "--num-steps-betas", "10",
        "--no-save-model-params",
        "--crt", "--crt-only",
        "--crt-pool", "all-cells",
        "--crt-mechanism", "propensity",
        "--crt-tail-families", "saddlepoint",
        "--crt-saddlepoint-only",
        "--crt-allow-unconverged-baseline",
    ]
    with warnings.catch_warnings(), contextlib.redirect_stdout(log):
        warnings.simplefilter("ignore")
        cli_main(argv)
    return out, log.getvalue()


def test_the_all_cells_run_keeps_a_column_for_the_lane_the_controls_missed(all_cells_run) -> None:
    out, _ = all_cells_run
    meta = json.loads((out / "covariate_metadata.json").read_text())

    assert meta["batch_levels_source"] == "analysed-cells"
    assert meta["batch_all_levels"] == list(ALL_CELLS_LANES)
    assert meta["unidentifiable_batch_levels"] == [UNSAMPLED_LANE]
    # The level the controls never sampled is named *and* still has a design
    # column, which is the whole point of the all-cells refit.
    assert f"batch:lane={UNSAMPLED_LANE}" in meta["covariate_names"]
    # And it is not the reference, so the control design is not aliased with
    # its own intercept.
    assert meta["batch_reference"] != UNSAMPLED_LANE
    assert meta["batch_reference"] in ALL_CELLS_LANES


def test_the_all_cells_run_records_its_batch_support(all_cells_run) -> None:
    out, log = all_cells_run
    meta = json.loads((out / "crt_metadata.json").read_text())
    covariates = json.loads((out / "covariate_metadata.json").read_text())

    assert meta["pool"] == "all-cells"
    assert meta["all_cells_batch_support"] is True
    # The support is per batch level, and every level is a column or the
    # reference: its width is the number of levels the data has.
    assert meta["all_cells_batch_levels"] == len(covariates["batch_all_levels"])
    assert meta["all_cells_elements"] == 6
    # The two non-targeting elements are the ones built to miss ``lane2``; the
    # four targeting ones are spread over every lane by construction.
    assert meta["all_cells_elements_missing_a_batch_level"] == 2
    # The summary line is printed by an unchunked run too, not only a chunked one.
    assert "[perturbo] CRT propensity: resampling support restricted" in log
    assert f"miss at least one of {len(ALL_CELLS_LANES)} levels" in log
    # The warning says what it does, and it is what the run actually did.
    assert f"batch level(s) {UNSAMPLED_LANE} of 'lane'" in log
    assert "They are kept in the design" in log
