import json

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

from perturbo.fitted_export import export


def bundle(path, strategy):
    path.mkdir()
    genes, guides, elems = ["g2", "g1"], ["a2", "b", "a1"], ["A", "B"]
    cells = ["z", "x", "y"]
    rna = ad.AnnData(
        np.array([[2, 1], [4, 3], [6, 5]], dtype=np.uint16),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=genes),
    )
    grna = ad.AnnData(np.eye(3, dtype=np.uint8), obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=guides))
    md.MuData({"rna": rna, "grna": grna}).write_h5mu(path / "mdata.h5mu")
    meta = dict(
        gene_names=genes,
        guide_names=guides,
        perturbation_names=elems,
        guide_effect_strategy=strategy,
        cli_args={"none": None, "seed": 2},
    )
    (path / "metadata.json").write_text(json.dumps(meta))
    df = pd.DataFrame(
        {
            "element": ["A", "B", "A", "B"],
            "gene": ["g1", "g2", "g2", "g1"],
            "posterior_mean": np.array([1, 2, 3, 4], dtype="f4"),
            "posterior_scale": np.array([0.1, 0.2, 0.3, 0.4], dtype="f4"),
            "method": "perturbo",
            "crt_saddlepoint_p_value": [1e-200, 0.12, 0.31, np.nan],
            "crt_saddlepoint_q_value": [3e-200, 0.18, 0.31, np.nan],
            "crt_p_value": [0.4] * 4,
        }
    )
    df.to_parquet(path / "element_effects.parquet")
    cis = df.iloc[[2, 0]].copy()
    cis["crt_saddlepoint_q_value"] = [0.31, 2e-200]
    cis.to_parquet(path / "element_effects_requested_pairs.parquet")
    np.savez(path / "control_fit.npz", beta_0=np.ones(2, dtype="f4"), size_factors=np.array([4.0, 5.0]))
    np.save(path / "baseline_cell_indices.npy", [2, 0])
    np.save(path / "cell_keep_indices.npy", [0, 2])
    np.save(path / "size_factors.npy", [5.0, 4.0])
    cov = pd.DataFrame({"batch": [0.0, 1.0]}, index=["z", "y"])
    cov.to_parquet(path / "cell_covariates.parquet")
    if strategy == "relative":
        values = np.arange(6, dtype="f4").reshape(3, 2) / 10
        g = pd.DataFrame(
            {
                "guide": np.repeat(guides, 2),
                "gene": np.tile(genes, 3),
                "element": np.repeat(["A", "B", "A"], 2),
                "posterior_mean": values.ravel(),
                "method": "perturbo",
            }
        )
        g.iloc[::-1].to_parquet(path / "guide_efficiency.parquet", index=False)
        np.savez(
            path / "guide_posteriors.npz",
            guide_relative_efficiency_mean=values,
            guide_relative_efficiency_scale=values + 1,
            guide_effect_mean=values + 2,
            guide_effect_scale=values + 3,
        )
    return path


@pytest.mark.parametrize("strategy", ["relative", "shared"])
def test_native_export(tmp_path, strategy):
    root = bundle(tmp_path / "bundle", strategy)
    dest = export(root, tmp_path / "new", primary="saddlepoint", cis=True)
    m = md.read_h5mu(dest / "fitted.h5mu")
    pt = m.uns["_perturbo"]
    assert pt["metadata"]["method"] == "perturbo"
    np.testing.assert_array_equal(pt["baseline_cell_indices"], [2, 0])
    np.testing.assert_array_equal(pt["cell_keep_indices"], [0, 2])
    np.testing.assert_array_equal(m["rna"].X, [[2, 1], [4, 3], [6, 5]])
    np.testing.assert_array_equal(m["grna"].X, np.eye(3))
    fx = pt["effects"]
    np.testing.assert_equal(fx.q_cis.to_numpy(), [2e-200, np.nan, 0.31, np.nan])
    assert fx.q_global.dtype == np.float64
    assert pt["crt_metadata"]["result_fields"]["q_global"]["finite_p"] == 3
    if strategy == "relative":
        assert np.isclose(pt["efficiency"].iloc[0]["guide_effect_mean"], 2.5)
    else:
        assert "efficiency" not in pt
    assert (dest / "guide_efficiency.parquet").exists() == (strategy == "relative")


def test_unverified_legacy_baseline_and_undeclared_family(tmp_path):
    root = bundle(tmp_path / "old", "shared")
    (root / "baseline_cell_indices.npy").unlink()
    with pytest.raises(ValueError, match="primary"):
        export(root, tmp_path / "export")
    dest = export(root, tmp_path / "export", primary="saddlepoint")
    pt = md.read_h5mu(dest / "fitted.h5mu").uns["_perturbo"]
    assert "q_cis" not in pt["effects"]
    assert pt["metadata"]["baseline_cell_mapping"] == "unknown"


def test_mismatch_rejects_export(tmp_path):
    root = bundle(tmp_path / "old", "relative")
    g = pd.read_parquet(root / "guide_efficiency.parquet")
    g.loc[0, "posterior_mean"] = 100
    g.to_parquet(root / "guide_efficiency.parquet")
    with pytest.raises(ValueError, match="Values differ"):
        export(root, tmp_path / "new", primary="saddlepoint")
    assert not (tmp_path / "new").exists()
