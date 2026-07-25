"""Tests for standardized output table serialization."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd

from perturbo.api import BetaFit, summarize_betas
from perturbo.results import build_standard_element_effects_df


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
