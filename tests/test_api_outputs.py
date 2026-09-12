"""Tests for standardized output table serialization."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
from perturbo.api import BetaFit, summarize_betas
from perturbo._statistics import empirical_pvals_from_tnull_fixed0, z_to_two_sided_pvalues
from perturbo.results import build_standard_element_effects_df, write_standard_element_effects_parquet


def _make_beta_fit() -> BetaFit:
    mean = jnp.array([[0.1, -0.2], [0.3, 0.0]])
    scale = jnp.array([[0.5, 0.7], [0.9, 1.1]])
    z_values = mean / scale
    return BetaFit(
        posterior_mean=mean,
        posterior_scale=scale,
        z_values=z_values,
        losses=jnp.array([]),
        svi_result=None,
    )


def test_summarize_betas_still_works_for_internal_wide_output() -> None:
    beta_fit = _make_beta_fit()
    summary = summarize_betas(
        beta_fit,
        pert_names=["pert_a", "pert_b"],
        gene_names=["gene_1", "gene_2"],
        single_frame=False,
    )
    assert set(summary) == {"posterior_mean", "posterior_scale", "posterior_prob"}


def test_element_effects_parquet_schema_and_float32(tmp_path) -> None:
    frame = build_standard_element_effects_df(
        method="perturbo",
        effect_loc=np.array([[0.1, -0.2]], dtype=float),
        effect_scale=np.array([[0.2, 0.3]], dtype=float),
        element_names=["pert_0"],
        gene_names=["g1", "g2"],
        null_z_values=np.array([0.0, 0.1, -0.2], dtype=float),
    )
    out_path = tmp_path / "element_effects.parquet"
    frame.to_parquet(out_path, index=False)
    roundtrip = pd.read_parquet(out_path)
    assert list(roundtrip.columns) == [
        "method",
        "element",
        "gene",
        "posterior_mean",
        "posterior_scale",
        "z_value",
        "posterior_prob",
        "empirical_p_value",
    ]
    assert "scenario_id" not in roundtrip.columns
    for col in ("posterior_mean", "posterior_scale", "z_value", "posterior_prob", "empirical_p_value"):
        assert str(roundtrip[col].dtype) == "float32"
    assert roundtrip["empirical_p_value"].notna().all()


def test_element_effects_empirical_p_value_nan_without_null() -> None:
    frame = build_standard_element_effects_df(
        method="perturbo",
        effect_loc=np.array([[0.1]], dtype=float),
        effect_scale=np.array([[0.2]], dtype=float),
        element_names=["pert_0"],
        gene_names=["g1"],
        null_z_values=None,
    )
    assert frame["empirical_p_value"].isna().all()


def test_blockwise_parquet_matches_the_in_memory_table(tmp_path) -> None:
    rng = np.random.default_rng(19)
    loc = rng.normal(size=(3, 5)).astype(np.float32)
    scale = rng.uniform(0.1, 1.0, size=(3, 5)).astype(np.float32)
    extra = {
        "crt_p_value": rng.uniform(size=(3, 5)),
        "crt_saddlepoint_p_value": np.geomspace(1e-90, 0.9, 15).reshape(3, 5),
    }
    kwargs = dict(
        method="perturbo", effect_loc=loc, effect_scale=scale,
        element_names=["a", "b", "c"], gene_names=[f"g{i}" for i in range(5)],
        null_z_values=np.asarray([-1.0, 0.0, 1.0]), extra_columns=extra,
    )
    z = loc.astype(float) / scale.astype(float)
    empirical = empirical_pvals_from_tnull_fixed0(kwargs["null_z_values"], z.reshape(-1))
    expected = pd.DataFrame({
        "method": "perturbo",
        "element": np.repeat(["a", "b", "c"], 5),
        "gene": np.tile([f"g{i}" for i in range(5)], 3),
        "posterior_mean": loc.reshape(-1),
        "posterior_scale": scale.reshape(-1),
        "z_value": z.reshape(-1).astype(np.float32),
        "posterior_prob": z_to_two_sided_pvalues(z.reshape(-1)).astype(np.float32),
        "empirical_p_value": empirical.astype(np.float32),
        "crt_p_value": extra["crt_p_value"].reshape(-1),
        "crt_saddlepoint_p_value": extra["crt_saddlepoint_p_value"].reshape(-1),
    })
    path = tmp_path / "blocked.parquet"
    write_standard_element_effects_parquet(path, row_block_size=4, **kwargs)
    actual = pd.read_parquet(path)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual["crt_saddlepoint_p_value"].iloc[0] == 1e-90


def test_blockwise_writer_fits_the_empirical_null_once(tmp_path, monkeypatch) -> None:
    import perturbo.results as results

    calls = 0
    original = results.empirical_pvals_from_tnull_fixed0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(results, "empirical_pvals_from_tnull_fixed0", counted)
    write_standard_element_effects_parquet(
        tmp_path / "effects.parquet", method="perturbo",
        effect_loc=np.zeros((3, 5)), effect_scale=np.ones((3, 5)),
        element_names=["a", "b", "c"], gene_names=[f"g{i}" for i in range(5)],
        null_z_values=np.asarray([-2.0, -0.5, 0.5, 2.0]), row_block_size=2,
    )
    assert calls == 1


def test_empty_grid_keeps_a_typed_parquet_schema(tmp_path) -> None:
    path = tmp_path / "empty.parquet"
    write_standard_element_effects_parquet(
        path, method="perturbo", effect_loc=np.empty((0, 2)), effect_scale=np.empty((0, 2)),
        element_names=[], gene_names=["g0", "g1"], extra_columns={"crt_p_value": np.empty((0, 2))},
    )
    frame = pd.read_parquet(path)
    assert frame.empty
    assert frame["posterior_mean"].dtype == np.float32
    assert frame["crt_p_value"].dtype == np.float64


def test_blockwise_writer_collects_and_recorrects_requested_pairs(tmp_path) -> None:
    loc = np.zeros((3, 4), dtype=np.float32)
    scale = np.ones_like(loc)
    p = np.asarray([[0.01, 0.8, 0.7, 0.6], [0.5, 0.02, 0.9, 0.4], [0.3, 0.2, 0.1, 0.05]])
    requested = pd.DataFrame({"element": ["a", "b", "missing"], "gene": ["g0", "g1", "g0"]})
    restricted = write_standard_element_effects_parquet(
        tmp_path / "effects.parquet", method="perturbo", effect_loc=loc, effect_scale=scale,
        element_names=["a", "b", "c"], gene_names=["g0", "g1", "g2", "g3"],
        extra_columns={"crt_p_value": p, "crt_q_value": p},
        requested_pairs=requested, row_block_size=3,
    )
    assert restricted is not None
    assert list(zip(restricted.element, restricted.gene)) == [("a", "g0"), ("b", "g1")]
    np.testing.assert_allclose(restricted["crt_q_value"], [0.02, 0.02])
