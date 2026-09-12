"""Phase-5 CRT surface: CLI wiring, accumulation, and the output contract."""

from __future__ import annotations

import anndata as ad
import jax.numpy as jnp
import mudata as md
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from perturbo.cli import main as cli_main
from perturbo.core import PerTurboData
from perturbo.crt import CRTAccumulator, ChunkCRTResult, exclude_targets


def _chunk_data(names: list[str], codes: np.ndarray, num_genes: int = 3) -> PerTurboData:
    cells = codes.shape[0]
    return PerTurboData(
        counts=jnp.ones((cells, num_genes), dtype=jnp.float32),
        pert_id=jnp.asarray(codes),
        pert_names=names,
        gene_names=[f"gene_{i}" for i in range(num_genes)],
        size_factors=jnp.zeros((cells, 1), dtype=jnp.float32),
    )


def test_excluding_targets_drops_their_cells_and_renumbers() -> None:
    data = _chunk_data(["NTC", "a", "b"], np.repeat([0, 1, 2], 4))
    kept = exclude_targets(data, ["NTC"])

    assert kept is not None
    assert kept.pert_names == ["a", "b"]
    assert int(np.asarray(kept.counts).shape[0]) == 8
    np.testing.assert_array_equal(np.asarray(kept.pert_id), np.repeat([0, 1], 4))


def test_excluding_every_target_returns_nothing() -> None:
    data = _chunk_data(["NTC"], np.zeros(6, dtype=np.int64))
    assert exclude_targets(data, ["NTC"]) is None


def test_excluding_nothing_is_a_passthrough() -> None:
    data = _chunk_data(["a", "b"], np.repeat([0, 1], 4))
    assert exclude_targets(data, ["NTC"]) is data


def test_excluding_targets_handles_an_assignment_matrix() -> None:
    matrix = np.zeros((12, 3), dtype=np.int8)
    matrix[np.arange(12), np.repeat([0, 1, 2], 4)] = 1
    data = _chunk_data(["NTC", "a", "b"], matrix)
    kept = exclude_targets(data, ["NTC"])

    assert kept is not None and kept.pert_names == ["a", "b"]
    assert np.asarray(kept.pert_id).shape == (8, 2)


def _chunk_result(targets: tuple[str, ...], genes: tuple[str, ...], value) -> ChunkCRTResult:
    shape = (len(targets), len(genes))
    p_value = np.broadcast_to(np.asarray(value, dtype=float), shape).copy()
    return ChunkCRTResult(
        observed_score=p_value.copy(),
        p_value=p_value,
        null_converged=np.ones(shape, dtype=bool),
        target_names=targets,
        gene_names=genes,
        num_resamples=99,
        parametric={},
        null_summaries={},
    )


def test_the_accumulator_places_chunks_by_name_not_position() -> None:
    genes = ("g0", "g1")
    accumulator = CRTAccumulator(element_names=("a", "b", "c"), gene_names=genes, tail_families=())
    # Absorbed out of order, and each chunk is position 0 in its own result.
    accumulator.absorb(_chunk_result(("c",), genes, 0.3))
    accumulator.absorb(_chunk_result(("a",), genes, 0.1))

    np.testing.assert_allclose(accumulator.p_value[0], 0.1)
    np.testing.assert_allclose(accumulator.p_value[2], 0.3)
    assert np.all(np.isnan(accumulator.p_value[1]))
    np.testing.assert_array_equal(accumulator.tested, [True, False, True])


def test_the_accumulator_rejects_an_unknown_element() -> None:
    accumulator = CRTAccumulator(element_names=("a",), gene_names=("g0",), tail_families=())
    with pytest.raises(ValueError, match="absent from the screen"):
        accumulator.absorb(_chunk_result(("z",), ("g0",), 0.5))


def test_the_accumulator_rejects_a_mismatched_gene_axis() -> None:
    accumulator = CRTAccumulator(element_names=("a",), gene_names=("g0",), tail_families=())
    with pytest.raises(ValueError, match="gene names do not match"):
        accumulator.absorb(_chunk_result(("a",), ("other",), 0.5))


def test_benjamini_hochberg_spans_every_chunk() -> None:
    """The correction must see the whole family, not one chunk at a time.

    Each chunk carries one small p-value against one large one. Corrected
    within a chunk the family size is 2; corrected across the screen it is 4,
    and the smallest q doubles. Getting this wrong understates FDR by exactly
    the number of chunks, which is a flag value rather than a property of the
    data - so it would not look wrong in the output.
    """

    genes = ("g0", "g1")
    accumulator = CRTAccumulator(element_names=("a", "b"), gene_names=genes, tail_families=())
    accumulator.absorb(_chunk_result(("a",), genes, [0.001, 0.9]))
    accumulator.absorb(_chunk_result(("b",), genes, [0.002, 0.9]))
    columns = accumulator.finalize()

    # Global: m=4, so 0.001 -> 4/1 * 0.001 = 0.004 (and 0.002 -> 4/2 * 0.002 too).
    # Per chunk it would have been m=2, giving 0.002.
    assert columns["crt_q_value"][0, 0] == pytest.approx(0.004)
    assert columns["crt_q_value"][1, 0] == pytest.approx(0.004)


def test_untested_elements_stay_missing_rather_than_becoming_significant() -> None:
    accumulator = CRTAccumulator(element_names=("a", "skipped"), gene_names=("g0",), tail_families=())
    accumulator.absorb(_chunk_result(("a",), ("g0",), 0.001))
    columns = accumulator.finalize()

    assert np.isfinite(columns["crt_q_value"][0, 0])
    assert np.isnan(columns["crt_q_value"][1, 0])


def _write_screen(path, *, num_genes: int = 24, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    targets = [f"t{i}" for i in range(4)]
    labels = np.asarray(["non-targeting"] * 400 + [t for t in targets for _ in range(60)])
    theta = rng.uniform(4.0, 10.0, size=num_genes)
    eta = np.full((labels.size, num_genes), 1.6)
    eta[labels == "t0", 1] -= 1.4
    counts = rng.negative_binomial(theta[None, :], theta[None, :] / (theta[None, :] + np.exp(eta)))
    obs = pd.DataFrame(
        {"gene": pd.Categorical(labels), "UMI": counts.sum(axis=1).astype(float)},
        index=[f"cell_{i}" for i in range(labels.size)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(num_genes)])
    md.MuData({"rna": ad.AnnData(X=sp.csr_matrix(counts.astype(np.float32)), obs=obs, var=var)}).write(path)


def _run_cli(tmp_path, *extra: str) -> pd.DataFrame:
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
            *extra,
        ]
    )
    return pd.read_parquet(out / "element_effects.parquet")


def test_the_crt_columns_are_absent_only_when_refused(tmp_path) -> None:
    """The test is on by default now; --no-crt is how a run opts out."""
    frame = _run_cli(tmp_path, "--no-crt")
    assert not [column for column in frame.columns if column.startswith("crt_")]


def test_the_crt_writes_its_columns_and_finds_the_planted_effect(tmp_path) -> None:
    frame = _run_cli(tmp_path, "--crt", "--crt-mechanism", "permutation", "--crt-tail-families", "skew_normal", "student_t", "--crt-num-resamples", "199", "--crt-seed", "1")

    for column in ("crt_z_value", "crt_p_value", "crt_q_value"):
        assert column in frame.columns

    # Controls are the pool, so they are not tested as targets.
    controls = frame[frame["element"] == "non-targeting"]
    assert controls["crt_p_value"].isna().all()

    planted = frame[(frame["element"] == "t0") & (frame["gene"] == "gene_1")]
    assert float(planted["crt_p_value"].iloc[0]) < 0.01

    tested = frame[frame["element"] != "non-targeting"]
    assert tested["crt_p_value"].notna().all()


def test_every_tail_family_is_reported_side_by_side(tmp_path) -> None:
    """All families share one resampling pass, so all of them are reported.

    They are alternative nulls for the same statistic, not a multiple-testing
    family among themselves, so each carries its own screen-wide q-value.
    """

    frame = _run_cli(tmp_path, "--crt", "--crt-mechanism", "permutation", "--crt-tail-families", "skew_normal", "student_t", "--crt-num-resamples", "199")

    for family in ("skew_normal", "student_t"):
        for suffix in ("p_value", "log_p_value", "q_value", "valid"):
            assert f"crt_{family}_{suffix}" in frame.columns
    for name in ("crt_null_mean", "crt_null_variance", "crt_null_skewness", "crt_null_excess_kurtosis"):
        assert name in frame.columns

    tested = frame[frame["element"] != "non-targeting"]
    # The point of the parametric tail: resolve below the empirical floor.
    floor = 1.0 / 200
    assert tested["crt_p_value"].min() >= floor * (1.0 - 1e-5)
    assert tested["crt_student_t_p_value"].min() < floor


def test_tail_families_can_be_switched_off(tmp_path) -> None:
    frame = _run_cli(tmp_path, "--crt", "--crt-num-resamples", "99", "--crt-tail-families")
    assert "crt_p_value" in frame.columns
    assert not [c for c in frame.columns if c.startswith("crt_student_t")]


def test_latent_size_factors_are_refused_before_any_fitting(tmp_path) -> None:
    """The guard must fire during argument handling, not after stage one."""

    screen = tmp_path / "screen.h5mu"
    _write_screen(screen)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="size_factor_mode"):
        cli_main(
            [
                "--input", str(screen),
                "--out-dir", str(out),
                "--modality-key", "rna",
                "--perturbation-key", "gene",
                "--control-substring", "non-targeting",
                "--library-size-key", "UMI",
                "--size-factor-mode", "infer",
                "--likelihood", "nb",
                "--num-steps", "200",
                "--crt",
            ]
        )
    # Nothing was fit, so nothing was written.
    assert not (out / "control_fit.npz").exists()


def test_the_saddlepoint_only_crt_needs_no_resamples(tmp_path) -> None:
    """The production flags for the exact-CGF saddlepoint under model-X draws.

    Nothing is drawn, so the empirical p-value is missing by construction and
    the saddlepoint column is what carries the test. The planted effect must
    still be found, and the screen flag must be reported.
    """

    frame = _run_cli(
        tmp_path,
        "--crt",
        "--crt-mechanism", "propensity",
        "--crt-tail-families", "saddlepoint",
        "--crt-saddlepoint-only",
        "--crt-screen-p-value", "0.2",
    )
    tested = frame[frame["element"] != "non-targeting"]
    assert tested["crt_p_value"].isna().all()
    assert tested["crt_z_value"].notna().all()
    for suffix in ("p_value", "log_p_value", "q_value", "valid", "used_screen"):
        assert f"crt_saddlepoint_{suffix}" in frame.columns
    assert tested["crt_saddlepoint_p_value"].notna().all()
    planted = tested[(tested["element"] == "t0") & (tested["gene"] == "gene_1")]
    assert float(planted["crt_saddlepoint_p_value"].iloc[0]) < 1e-3
    assert float(planted["crt_saddlepoint_used_screen"].iloc[0]) == 0.0
    # Pairs the screen did not promote keep the Pearson III value and say so.
    assert tested["crt_saddlepoint_used_screen"].gt(0).any()
    # Untested controls stay missing rather than becoming significant.
    assert frame.loc[frame["element"] == "non-targeting", "crt_saddlepoint_p_value"].isna().all()


def test_saddlepoint_only_is_refused_without_the_propensity_mechanism(tmp_path) -> None:
    with pytest.raises(ValueError, match="crt-mechanism propensity"):
        _run_cli(tmp_path, "--crt", "--crt-mechanism", "permutation", "--crt-tail-families", "saddlepoint", "--crt-saddlepoint-only")


def test_the_saddlepoint_runs_beside_the_moment_families(tmp_path) -> None:
    """With draws on, the saddlepoint is one more column on the same pairs."""

    frame = _run_cli(
        tmp_path,
        "--crt", "--crt-num-resamples", "99",
        "--crt-mechanism", "propensity",
        "--crt-tail-families", "skew_normal", "saddlepoint",
    )
    tested = frame[frame["element"] != "non-targeting"]
    assert tested["crt_p_value"].notna().all()
    assert tested["crt_skew_normal_p_value"].notna().any()
    assert tested["crt_saddlepoint_p_value"].notna().all()


def test_the_baseline_can_be_polished_from_the_cli(tmp_path) -> None:
    frame = _run_cli(tmp_path, "--crt", "--crt-mechanism", "permutation", "--crt-tail-families", "skew_normal", "student_t", "--crt-num-resamples", "99", "--crt-polish-baseline")
    tested = frame[frame["element"] != "non-targeting"]
    assert tested["crt_p_value"].notna().all()
    planted = tested[(tested["element"] == "t0") & (tested["gene"] == "gene_1")]
    assert float(planted["crt_p_value"].iloc[0]) < 0.05


def test_chunked_crt_only_tests_each_target_once(tmp_path, monkeypatch) -> None:
    """The chunked --crt-only path must not fall through into the unchunked branch and rerun the CRT."""
    import perturbo.cli as cli_module

    original = cli_module.run_crt_for_chunk
    tested: list[str] = []
    calls: list[int] = []

    def counting(*args, **kwargs):
        result = original(*args, **kwargs)
        tested.extend(str(name) for name in result.target_names)
        calls.append(1)
        return result

    monkeypatch.setattr(cli_module, "run_crt_for_chunk", counting)
    frame = _run_cli(
        tmp_path,
        "--crt", "--crt-mechanism", "propensity", "--crt-tail-families", "saddlepoint", "--crt-saddlepoint-only",
        "--crt-polish-baseline", "--crt-allow-unconverged-baseline", "--crt-only", "--max-chunk-size", "500",
    )
    assert len(calls) >= 2, "the fixture should have been split into several chunks"
    assert len(tested) == len(set(tested)) == 4, tested
    assert frame["crt_saddlepoint_p_value"].notna().any()
    assert frame["posterior_mean"].isna().all() if "posterior_mean" in frame else True


def _write_batched_screen(path, *, num_genes: int = 24, seed: int = 0) -> None:
    """Low-MOI screen with a three-level batch that shifts every gene's baseline."""
    rng = np.random.default_rng(seed)
    targets = [f"t{i}" for i in range(4)]
    labels = np.asarray(["non-targeting"] * 600 + [t for t in targets for _ in range(90)])
    batch = rng.integers(0, 3, size=labels.size)
    theta = rng.uniform(4.0, 10.0, size=num_genes)
    eta = np.full((labels.size, num_genes), 1.6) + np.array([0.0, -0.6, 0.5])[batch][:, None]
    eta[labels == "t0", 1] -= 1.4
    counts = rng.negative_binomial(theta[None, :], theta[None, :] / (theta[None, :] + np.exp(eta)))
    obs = pd.DataFrame(
        {"gene": pd.Categorical(labels), "UMI": counts.sum(axis=1).astype(float), "batch": pd.Categorical([f"b{b}" for b in batch])},
        index=[f"cell_{i}" for i in range(labels.size)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(num_genes)])
    md.MuData({"rna": ad.AnnData(X=sp.csr_matrix(counts.astype(np.float32)), obs=obs, var=var)}).write(path)


def test_categorical_batch_path_matches_the_dense_design(tmp_path, monkeypatch) -> None:
    """With a batch covariate the kernel's per-batch intercepts reproduce the dense one-hot design."""
    import perturbo.crt as crt_module

    def run(subdir: str) -> pd.DataFrame:
        screen = tmp_path / subdir / "screen.h5mu"
        screen.parent.mkdir()
        _write_batched_screen(screen)
        out = tmp_path / subdir / "out"
        cli_main([
            "--input", str(screen), "--out-dir", str(out), "--modality-key", "rna", "--perturbation-key", "gene",
            "--control-substring", "non-targeting", "--library-size-key", "UMI", "--size-factor-mode", "observed",
            "--batch-covariate", "batch", "--likelihood", "nb", "--num-steps", "300", "--no-save-model-params",
            "--crt", "--crt-only", "--crt-mechanism", "propensity", "--crt-tail-families", "saddlepoint", "--crt-saddlepoint-only",
            "--crt-polish-baseline", "--crt-allow-unconverged-baseline",
        ])
        return pd.read_parquet(out / "element_effects.parquet").sort_values(["element", "gene"]).reset_index(drop=True)

    seen: list[bool] = []
    original = crt_module._categorical_batch_applies

    def recording(design):
        value = original(design)
        seen.append(value)
        return value

    monkeypatch.setattr(crt_module, "_categorical_batch_applies", recording)
    categorical = run("categorical")
    assert seen and all(seen), "the batch-only design should take the categorical path"

    monkeypatch.setattr(crt_module, "_categorical_batch_applies", lambda design: False)
    dense = run("dense")

    assert list(categorical.element) == list(dense.element) and list(categorical.gene) == list(dense.gene)
    tested = (categorical.element != "non-targeting").to_numpy()  # the control element is skipped as a target
    categorical, dense = categorical[tested], dense[tested]
    lp_c = np.log10(np.clip(categorical.crt_saddlepoint_p_value.to_numpy(float), 1e-300, 1.0))
    lp_d = np.log10(np.clip(dense.crt_saddlepoint_p_value.to_numpy(float), 1e-300, 1.0))
    ok = np.isfinite(lp_c) & np.isfinite(lp_d)
    assert ok.all()
    # The categorical path stratifies the propensity null by batch (per-target,
    # per-batch selection intercepts); the dense path keeps one intercept per
    # target with the batch in the shared logits. On a screen whose targets are
    # spread across batches independently of batch the two selection models
    # coincide in expectation, and the statistics differ only through that
    # interaction and float32 kernel arithmetic.
    diff = np.abs(lp_c[ok] - lp_d[ok])
    print(f"categorical vs dense: median |dlog10p| {np.median(diff):.4f}, 99th pct {np.percentile(diff, 99):.4f}, Pearson {np.corrcoef(lp_c[ok], lp_d[ok])[0, 1]:.5f}")
    assert np.median(diff) < 0.02
    assert np.corrcoef(lp_c[ok], lp_d[ok])[0, 1] > 0.995
    planted = categorical[(categorical.element == "t0") & (categorical.gene == "gene_1")].crt_saddlepoint_p_value.iloc[0]
    assert planted < 1e-4


def test_the_crt_runs_by_default(tmp_path) -> None:
    """It is the point of the tool now, so a plain run carries it."""
    frame = _run_cli(tmp_path)            # no --crt anywhere
    assert [column for column in frame.columns if column.startswith("crt_")]


def test_an_unsupported_default_steps_aside_but_an_explicit_request_stops(tmp_path, capsys):
    """Latent factors are incompatible with the test. A run that never asked for it
    should still fit; a run that asked should hear why it cannot."""
    frame = _run_cli(tmp_path, "--num-factors", "2")
    out = capsys.readouterr().out
    assert "Skipping the conditional randomization test" in out and "latent factors" in out
    assert not [c for c in frame.columns if c.startswith("crt_")]
    explicit = tmp_path / "explicit"; explicit.mkdir()
    with pytest.raises(ValueError, match="latent factors"):
        _run_cli(explicit, "--num-factors", "2", "--crt")
