"""The per-pair low-information flag: what it counts, and that it only reports.

The flag exists because a CRT p-value can be extrapolated from almost nothing
and say nothing about it. The rule counts a pair's informative cells two ways -
how many of the element's cells detected the gene, and how many the fitted null
expected to - and flags the pair when neither reaches the threshold. Each count
alone is wrong in one direction, which is what the knockdown and induction cases
below pin down.

The load-bearing test is :func:`test_the_flag_changes_no_p_value`: everything
else describes the diagnostic, that one says it is a diagnostic.
"""

from __future__ import annotations

import json

import anndata as ad
import jax.numpy as jnp
import mudata as md
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from perturbo.cli import main as cli_main
from perturbo.core import ControlFit, PerTurboData
from perturbo.crt import (
    DEFAULT_CRT_MIN_INFORMATIVE_CELLS,
    CRTAccumulator,
    prepare_crt_baseline,
    run_crt_for_chunk,
)
from perturbo._internal.score_resampling import fit_batched_nb_null


# --- a screen with one gene of each kind ------------------------------------
#
# gene_0  well detected everywhere, no effect          -> informative
# gene_1  detected in ~1% of cells, no effect          -> the thin case
# gene_2  well detected, target 1 knocks it out        -> observed 0, expected high
# gene_3  essentially undetected, target 2 induces it  -> observed high, expected ~0

GENE_SPARSE = 1
GENE_KNOCKDOWN = 2
GENE_INDUCTION = 3

TARGET_SMALL = 0
TARGET_KNOCKDOWN = 1
TARGET_INDUCTION = 2


def _simulate_screen(*, seed: int = 11):
    """Controls and one perturbation chunk, split the way production splits them.

    Cell counts are chosen so the arithmetic of the rule is unambiguous rather
    than marginal: the small element has 6 cells against a gene detected in 1%
    of them, the knockdown element has 60, the induction element 40.
    """

    rng = np.random.default_rng(seed)
    num_control = 600
    cells_per_target = [6, 60, 40]
    num_genes = 4
    theta = np.asarray([8.0, 8.0, 8.0, 8.0])
    # gene_1 and gene_3 sit far below the detection floor; gene_0 and gene_2 are
    # ordinary. exp(-4.6) ~ 0.01 counts per cell.
    beta_0 = np.asarray([2.2, -4.6, 2.2, -6.5])

    codes = np.concatenate(
        [np.full(num_control, -1)]
        + [np.full(count, index) for index, count in enumerate(cells_per_target)]
    )
    total = codes.size
    offsets = np.zeros((total, 1))

    eta = offsets + beta_0[None, :]
    # A knockdown the null still expects to see: 60 cells, gene detected in
    # essentially all of them under the null, zero counts observed.
    eta[codes == TARGET_KNOCKDOWN, GENE_KNOCKDOWN] -= 40.0
    # An induction from an undetected baseline: the null expects ~0.3 cells over
    # the element's 40, and all 40 light up.
    eta[codes == TARGET_INDUCTION, GENE_INDUCTION] += 9.0

    counts = rng.negative_binomial(
        theta[None, :], theta[None, :] / (theta[None, :] + np.exp(eta))
    ).astype(np.float64)

    gene_names = [f"gene_{index}" for index in range(num_genes)]
    target_names = [f"target_{index}" for index in range(len(cells_per_target))]
    control_data = PerTurboData(
        counts=jnp.asarray(counts[:num_control], dtype=jnp.float32),
        pert_id=jnp.zeros((num_control,), dtype=jnp.int32),
        pert_names=["NTC"],
        gene_names=gene_names,
        size_factors=jnp.asarray(offsets[:num_control], dtype=jnp.float32),
        library_size_center_log_mean=0.0,
    )
    chunk_data = PerTurboData(
        counts=jnp.asarray(counts[num_control:], dtype=jnp.float32),
        pert_id=jnp.asarray(codes[num_control:], dtype=jnp.int32),
        pert_names=target_names,
        gene_names=gene_names,
        size_factors=jnp.asarray(offsets[num_control:], dtype=jnp.float32),
        library_size_center_log_mean=0.0,
    )
    return control_data, chunk_data, theta


def _control_fit(control_data: PerTurboData, theta: np.ndarray) -> ControlFit:
    counts = np.asarray(control_data.counts, dtype=np.float64)
    design = np.ones((counts.shape[0], 1))
    coefficients = fit_batched_nb_null(
        counts,
        nuisance_design=design,
        offsets=np.asarray(control_data.size_factors, dtype=np.float64),
        theta=theta,
    ).nuisance_mean
    return ControlFit(
        beta_0=jnp.asarray(coefficients[0], dtype=jnp.float32),
        theta=jnp.asarray(theta, dtype=jnp.float32),
        noise_scale=jnp.zeros(theta.size, dtype=jnp.float32),
        factor_loadings=None,
        factor_scores=None,
        factor_center=None,
        pca_loadings=None,
        size_factors=control_data.size_factors,
        losses=jnp.zeros((1,), dtype=jnp.float32),
        svi_result=None,
        covariate_coef=None,
    )


@pytest.fixture(scope="module")
def chunk_result():
    control_data, chunk_data, theta = _simulate_screen()
    baseline = prepare_crt_baseline(
        control_data, _control_fit(control_data, theta), strict=False
    )
    result = run_crt_for_chunk(
        baseline,
        chunk_data,
        control_data=control_data,
        num_resamples=99,
        seed=4,
        tail_families=(),
    )
    return result


def _counts(result, target: int, gene: int) -> tuple[float, float]:
    return (
        float(result.observed_nonzero[target, gene]),
        float(result.expected_nonzero[target, gene]),
    )


# --- (a) the thin case -------------------------------------------------------


def test_a_barely_detected_gene_on_a_small_element_is_flagged(chunk_result) -> None:
    observed, expected = _counts(chunk_result, TARGET_SMALL, GENE_SPARSE)

    assert observed < DEFAULT_CRT_MIN_INFORMATIVE_CELLS
    assert expected < DEFAULT_CRT_MIN_INFORMATIVE_CELLS
    assert max(observed, expected) < DEFAULT_CRT_MIN_INFORMATIVE_CELLS


def test_a_well_detected_gene_on_the_same_small_element_is_not_flagged(chunk_result) -> None:
    """The element is small either way; what changes is the gene.

    Six cells of a gene detected in nearly all of them is six informative
    cells, which is over the threshold. The rule is about information, not
    about element size, and this is the pair that separates the two.
    """

    observed, expected = _counts(chunk_result, TARGET_SMALL, 0)
    assert max(observed, expected) >= DEFAULT_CRT_MIN_INFORMATIVE_CELLS


# --- (b) and (c) the two asymmetric cases -----------------------------------


def test_a_strong_knockdown_is_not_flagged_though_nothing_was_observed(chunk_result) -> None:
    """Observed goes to zero precisely because the effect is real.

    SCEPTRE's low-MOI rule counts observed cells alone, and on this pair it
    would discard the strongest true positive in the screen. The expected count
    is what keeps it.
    """

    observed, expected = _counts(chunk_result, TARGET_KNOCKDOWN, GENE_KNOCKDOWN)

    assert observed == 0.0
    assert expected > 30.0
    assert max(observed, expected) >= DEFAULT_CRT_MIN_INFORMATIVE_CELLS


def test_an_induction_is_not_flagged_though_nothing_was_expected(chunk_result) -> None:
    """The mirror image: expected is ~0 because the baseline is undetected."""

    observed, expected = _counts(chunk_result, TARGET_INDUCTION, GENE_INDUCTION)

    assert observed > 15.0
    assert expected < 1.0
    assert max(observed, expected) >= DEFAULT_CRT_MIN_INFORMATIVE_CELLS


def test_the_expected_count_is_the_fitted_null_not_a_raw_gene_mean(chunk_result) -> None:
    """Expected detection is bounded by the element's cell count, and positive.

    A count that came from a gene-level rate rather than the per-cell fitted
    null would not respect the element's own cell count.
    """

    cells_per_target = np.asarray([6, 60, 40], dtype=float)
    expected = np.asarray(chunk_result.expected_nonzero)
    observed = np.asarray(chunk_result.observed_nonzero)

    assert np.all(expected >= 0.0)
    assert np.all(expected <= cells_per_target[:, None] + 1e-6)
    assert np.all(observed <= cells_per_target[:, None])
    # An ordinary gene is expected in essentially every cell of every element.
    np.testing.assert_allclose(expected[:, 0], cells_per_target, rtol=0.05)


# --- the flag is a flag ------------------------------------------------------


def test_the_flag_changes_no_p_value() -> None:
    """Counting cells must not touch the statistic, the tail, or the seed.

    Compared against a run with the counting switched off entirely, which is the
    only way to be sure the extra pass did not perturb the RNG or the kernel.
    """

    control_data, chunk_data, theta = _simulate_screen()
    baseline = prepare_crt_baseline(
        control_data, _control_fit(control_data, theta), strict=False
    )
    shared = dict(
        control_data=control_data, num_resamples=99, seed=4, tail_families=()
    )
    with_counts = run_crt_for_chunk(baseline, chunk_data, **shared)
    without_counts = run_crt_for_chunk(
        baseline, chunk_data, count_informative_cells=False, **shared
    )

    assert without_counts.observed_nonzero is None
    assert without_counts.expected_nonzero is None
    np.testing.assert_array_equal(with_counts.p_value, without_counts.p_value)
    np.testing.assert_array_equal(with_counts.observed_score, without_counts.observed_score)
    np.testing.assert_array_equal(with_counts.null_converged, without_counts.null_converged)


def test_the_counts_do_not_depend_on_the_gene_chunk_size() -> None:
    """Gene chunking is a decomposition of the same computation, here too."""

    control_data, chunk_data, theta = _simulate_screen()
    baseline = prepare_crt_baseline(
        control_data, _control_fit(control_data, theta), strict=False
    )
    shared = dict(
        control_data=control_data, num_resamples=49, seed=2, tail_families=()
    )
    whole = run_crt_for_chunk(baseline, chunk_data, **shared)
    blocked = run_crt_for_chunk(baseline, chunk_data, gene_chunk_size=1, **shared)

    np.testing.assert_array_equal(whole.observed_nonzero, blocked.observed_nonzero)
    np.testing.assert_allclose(whole.expected_nonzero, blocked.expected_nonzero, rtol=1e-6)


# --- accumulator and threshold ----------------------------------------------


def test_the_accumulator_emits_the_three_columns(chunk_result) -> None:
    accumulator = CRTAccumulator(
        element_names=chunk_result.target_names,
        gene_names=chunk_result.gene_names,
        tail_families=(),
    )
    accumulator.absorb(chunk_result)
    columns = accumulator.finalize()

    assert np.issubdtype(columns["crt_observed_nonzero"].dtype, np.integer)
    assert columns["crt_expected_nonzero"].dtype == np.float64
    assert columns["crt_low_information"].dtype == bool
    assert columns["crt_low_information"][TARGET_SMALL, GENE_SPARSE]
    assert not columns["crt_low_information"][TARGET_KNOCKDOWN, GENE_KNOCKDOWN]
    assert not columns["crt_low_information"][TARGET_INDUCTION, GENE_INDUCTION]


def test_a_threshold_of_zero_flags_nothing_and_keeps_the_columns(chunk_result) -> None:
    accumulator = CRTAccumulator(
        element_names=chunk_result.target_names,
        gene_names=chunk_result.gene_names,
        tail_families=(),
        min_informative_cells=0.0,
    )
    accumulator.absorb(chunk_result)
    columns = accumulator.finalize()

    assert "crt_observed_nonzero" in columns
    assert "crt_expected_nonzero" in columns
    assert not columns["crt_low_information"].any()


def test_a_run_without_the_counts_emits_no_columns() -> None:
    control_data, chunk_data, theta = _simulate_screen()
    baseline = prepare_crt_baseline(
        control_data, _control_fit(control_data, theta), strict=False
    )
    result = run_crt_for_chunk(
        baseline,
        chunk_data,
        control_data=control_data,
        num_resamples=49,
        seed=1,
        tail_families=(),
        count_informative_cells=False,
    )
    accumulator = CRTAccumulator(
        element_names=result.target_names, gene_names=result.gene_names, tail_families=()
    )
    accumulator.absorb(result)
    columns = accumulator.finalize()

    assert "crt_observed_nonzero" not in columns
    assert "crt_low_information" not in columns


# --- (d) and (e) the CLI -----------------------------------------------------


def _write_screen(path, *, seed: int = 0) -> None:
    """A CLI-shaped version of the same screen, with a barely-detected gene."""

    rng = np.random.default_rng(seed)
    targets = [f"t{index}" for index in range(3)]
    labels = np.asarray(["non-targeting"] * 400 + [t for t in targets for _ in range(40)])
    num_genes = 8
    theta = rng.uniform(4.0, 10.0, size=num_genes)
    baseline = np.full(num_genes, 1.6)
    baseline[GENE_SPARSE] = -4.6
    eta = np.broadcast_to(baseline, (labels.size, num_genes)).copy()
    eta[labels == "t0", 0] -= 1.4
    counts = rng.negative_binomial(theta[None, :], theta[None, :] / (theta[None, :] + np.exp(eta)))
    obs = pd.DataFrame(
        {"gene": pd.Categorical(labels), "UMI": counts.sum(axis=1).astype(float)},
        index=[f"cell_{index}" for index in range(labels.size)],
    )
    var = pd.DataFrame(index=[f"gene_{index}" for index in range(num_genes)])
    md.MuData(
        {"rna": ad.AnnData(X=sp.csr_matrix(counts.astype(np.float32)), obs=obs, var=var)}
    ).write(path)


def _run_cli(tmp_path, *extra: str):
    tmp_path.mkdir(parents=True, exist_ok=True)
    screen = tmp_path / "screen.h5mu"
    _write_screen(screen)
    out = tmp_path / "out"
    cli_main(
        [
            "--input", str(screen),
            "--out-dir", str(out),
            "--modality-key", "rna",
            "--perturbation-key", "gene",
            "--control-substring", "non-targeting",
            "--library-size-key", "UMI",
            "--size-factor-mode", "observed",
            "--likelihood", "nb",
            "--num-steps", "200",
            "--no-save-model-params",
            "--crt",
            "--crt-mechanism", "permutation",
            "--crt-tail-families", "skew_normal",
            "--crt-num-resamples", "99",
            "--crt-seed", "1",
            *extra,
        ]
    )
    return (
        pd.read_parquet(out / "element_effects.parquet"),
        json.loads((out / "crt_metadata.json").read_text()),
    )


def test_the_cli_writes_the_columns_with_their_own_dtypes(tmp_path) -> None:
    frame, metadata = _run_cli(tmp_path)

    assert np.issubdtype(frame["crt_observed_nonzero"].dtype, np.integer)
    assert np.issubdtype(frame["crt_expected_nonzero"].dtype, np.floating)
    assert frame["crt_low_information"].dtype == bool

    tested = frame[frame["element"] != "non-targeting"]
    sparse_gene = tested[tested["gene"] == f"gene_{GENE_SPARSE}"]
    ordinary = tested[tested["gene"] == "gene_3"]
    assert sparse_gene["crt_low_information"].all()
    assert not ordinary["crt_low_information"].any()
    # The p-values are untouched by the flag, so the flagged rows still carry them.
    assert sparse_gene["crt_p_value"].notna().all()

    assert metadata["min_informative_cells"] == DEFAULT_CRT_MIN_INFORMATIVE_CELLS
    assert metadata["low_information_pairs"] == int(frame["crt_low_information"].sum())
    assert metadata["low_information_total_pairs"] == len(frame)
    assert metadata["low_information_genes_entirely_flagged"] >= 1
    assert 0 <= metadata["low_information_pairs_significant"] <= metadata["low_information_pairs"]


def test_the_cli_threshold_is_honoured_and_zero_disables_it(tmp_path) -> None:
    lenient, metadata = _run_cli(tmp_path, "--crt-min-informative-cells", "0")

    assert "crt_observed_nonzero" in lenient.columns
    assert "crt_expected_nonzero" in lenient.columns
    assert not lenient["crt_low_information"].any()
    assert metadata["min_informative_cells"] == 0.0
    assert metadata["low_information_pairs"] == 0


def test_the_cli_threshold_only_moves_the_flag(tmp_path) -> None:
    """Two thresholds, same run: the counts and every p-value are identical."""

    default_frame, _ = _run_cli(tmp_path / "a")
    strict_frame, strict_metadata = _run_cli(tmp_path / "b", "--crt-min-informative-cells", "1000")

    keys = ["element", "gene"]
    merged = default_frame.merge(strict_frame, on=keys, suffixes=("_d", "_s"))
    np.testing.assert_array_equal(
        merged["crt_observed_nonzero_d"], merged["crt_observed_nonzero_s"]
    )
    np.testing.assert_allclose(
        merged["crt_p_value_d"].to_numpy(dtype=float),
        merged["crt_p_value_s"].to_numpy(dtype=float),
        equal_nan=True,
    )
    assert strict_frame["crt_low_information"].all()
    assert strict_metadata["low_information_pairs"] == len(strict_frame)


def test_the_cli_rejects_a_negative_threshold(tmp_path) -> None:
    with pytest.raises(ValueError, match="crt-min-informative-cells"):
        _run_cli(tmp_path, "--crt-min-informative-cells", "-1")


# --- the all-cells pool ------------------------------------------------------


def _write_high_moi_screen(path, *, n_cells=800, n_genes=8, n_elements=6, seed=0) -> None:
    """A high-MOI element design with one gene nobody detects.

    Cells carry several elements each, so membership is the guide-to-element COO
    the all-cells statistic segment-sums over rather than a one-per-cell label.
    """

    rng = np.random.default_rng(seed)
    guides_per_element = 2
    n_guides = n_elements * guides_per_element
    guide_to_element = np.zeros((n_guides, n_elements), dtype=np.float32)
    guide_to_element[
        np.arange(n_guides), np.repeat(np.arange(n_elements), guides_per_element)
    ] = 1.0
    assignment = (rng.random((n_cells, n_guides)) < 0.2).astype(np.float32)
    log_mu = np.full(n_genes, 1.5)
    log_mu[GENE_SPARSE] = -6.5
    log_mu = np.broadcast_to(log_mu, (n_cells, n_genes)).copy()
    theta = 4.0
    counts = rng.negative_binomial(theta, theta / (theta + np.exp(log_mu))).astype(np.float32)
    gene = ad.AnnData(
        X=sp.csr_matrix(counts),
        obs=pd.DataFrame(
            {"total_umis": counts.sum(axis=1).astype(np.int64) + 50},
            index=[f"cell{index}" for index in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"gene_{index}" for index in range(n_genes)]),
    )
    guide = ad.AnnData(
        X=sp.csr_matrix(assignment),
        obs=gene.obs.copy(),
        var=pd.DataFrame(index=[f"guide{index}" for index in range(n_guides)]),
    )
    guide.varm["guide_intended_target_pairs"] = sp.csr_matrix(guide_to_element)
    guide.uns["intended_targets"] = np.array(
        [f"elem_{index}" for index in range(n_elements - 1)] + ["non-targeting_a"],
        dtype=object,
    )
    md.MuData({"gene": gene, "guide": guide}).write_h5mu(path)


def test_the_all_cells_pool_counts_over_each_elements_own_cells(tmp_path) -> None:
    screen = tmp_path / "screen.h5mu"
    _write_high_moi_screen(screen)
    out = tmp_path / "out"
    cli_main([
        "--input", str(screen), "--out-dir", str(out),
        "--modality-key", "gene", "--perturbation-modality-key", "guide",
        "--perturbation-element-varm-key", "guide_intended_target_pairs",
        "--perturbation-element-names-uns-key", "intended_targets",
        "--library-size-key", "total_umis", "--size-factor-mode", "observed",
        "--likelihood", "nb", "--num-steps-control", "200", "--num-steps-betas", "50",
        "--no-save-model-params", "--crt", "--crt-mechanism", "propensity",
        "--crt-tail-families", "saddlepoint", "--crt-saddlepoint-only",
        "--crt-allow-unconverged-baseline", "--crt-only", "--crt-pool", "all-cells",
    ])
    frame = pd.read_parquet(out / "element_effects.parquet")
    metadata = json.loads((out / "crt_metadata.json").read_text())

    assert metadata["pool"] == "all-cells"
    assert np.issubdtype(frame["crt_observed_nonzero"].dtype, np.integer)
    assert frame["crt_low_information"].dtype == bool
    # Each element sits in roughly a third of 800 cells, so a well-detected gene
    # is informative and the barely-detected one is not, whatever the element.
    sparse_gene = frame[frame["gene"] == f"gene_{GENE_SPARSE}"]
    ordinary = frame[frame["gene"] == "gene_0"]
    assert sparse_gene["crt_low_information"].all()
    assert not ordinary["crt_low_information"].any()
    assert (ordinary["crt_observed_nonzero"] > 100).all()
    assert metadata["low_information_genes_entirely_flagged"] == 1
    assert sparse_gene["crt_saddlepoint_p_value"].notna().any()
