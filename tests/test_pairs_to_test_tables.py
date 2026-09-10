"""One run, two tables: the restricted table differs only in its q-values."""
import numpy as np
import pandas as pd
import pytest

from perturbo.results import build_standard_element_effects_df, load_pairs_to_test, restrict_effects_to_pairs


def _effects(n_elements=4, n_genes=5, seed=0):
    rng = np.random.default_rng(seed)
    loc = rng.normal(size=(n_elements, n_genes))
    scale = np.full((n_elements, n_genes), 0.4)
    crt_p = rng.uniform(size=(n_elements, n_genes))
    return build_standard_element_effects_df(
        method="perturbo",
        effect_loc=loc,
        effect_scale=scale,
        element_names=[f"e{i}" for i in range(n_elements)],
        gene_names=[f"g{j}" for j in range(n_genes)],
        extra_columns={
            "crt_p_value": crt_p,
            "crt_q_value": np.full((n_elements, n_genes), np.nan),
        },
    )


def test_loader_requires_the_two_columns(tmp_path):
    path = tmp_path / "pairs.csv"
    pd.DataFrame({"element": ["e0"], "not_gene": ["g0"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="element' and 'gene'"):
        load_pairs_to_test(path)


def test_loader_reads_csv_tsv_and_parquet(tmp_path):
    pairs = pd.DataFrame({"element": ["e0", "e1", "e0"], "gene": ["g0", "g1", "g0"]})
    for name, writer in (("p.csv", lambda f: pairs.to_csv(f, index=False)),
                         ("p.tsv", lambda f: pairs.to_csv(f, sep="\t", index=False)),
                         ("p.parquet", lambda f: pairs.to_parquet(f, index=False))):
        writer(tmp_path / name)
        loaded = load_pairs_to_test(tmp_path / name)
        assert list(loaded.columns) == ["element", "gene"]
        assert len(loaded) == 2, "duplicates are dropped"


def test_restriction_keeps_the_estimates_and_recorrects_the_q_values():
    effects = _effects()
    pairs = pd.DataFrame({"element": ["e0", "e0", "e1"], "gene": ["g0", "g1", "g3"]})
    restricted = restrict_effects_to_pairs(effects, pairs)
    assert len(restricted) == 3
    merged = restricted.merge(effects, on=["element", "gene"], suffixes=("_r", "_full"))
    # The fit and the test are the same run: estimates and p-values are identical.
    for column in ("posterior_mean", "posterior_scale", "z_value", "crt_p_value"):
        np.testing.assert_allclose(merged[f"{column}_r"], merged[f"{column}_full"], rtol=0, atol=0)
    # The family is smaller, so the q-values are not.
    expected = restricted["crt_p_value"].to_numpy() * len(restricted) / restricted["crt_p_value"].rank().to_numpy()
    assert np.all(restricted["crt_q_value"].to_numpy() <= np.maximum(expected, 0) + 1e-9)
    assert restricted["crt_q_value"].notna().all()


def test_pairs_absent_from_the_grid_are_dropped_not_invented():
    effects = _effects()
    pairs = pd.DataFrame({"element": ["e0", "nonexistent"], "gene": ["g0", "g0"]})
    restricted = restrict_effects_to_pairs(effects, pairs)
    assert list(restricted["element"]) == ["e0"]


def test_correction_matches_the_pipelines_sceptre_module():
    """The cis table must be corrected exactly as the SCEPTRE module corrects its own.

    ``bin/merge_sceptre_chunk_results.py`` in the IGVF CRISPR pipeline applies
    ``scipy.stats.false_discovery_control(method="bh")`` to the finite p-values of
    the merged table and leaves the rest NaN. A cis table corrected any other way
    would not be comparable with SCEPTRE's, which is the only reason it exists.
    """
    from scipy.stats import false_discovery_control

    from perturbo._statistics import benjamini_hochberg_over_finite
    from perturbo._internal.score_resampling import _benjamini_hochberg

    rng = np.random.default_rng(7)
    p = rng.uniform(size=500)
    p[rng.choice(500, 40, replace=False)] = np.nan
    finite = np.isfinite(p)
    expected = np.full(p.shape, np.nan)
    expected[finite] = false_discovery_control(np.clip(p[finite], 0.0, 1.0), method="bh")

    np.testing.assert_allclose(benjamini_hochberg_over_finite(p), expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(_benjamini_hochberg(p), expected, rtol=1e-12, atol=0)


def test_the_two_tables_differ_only_where_the_family_does():
    effects = _effects(n_elements=6, n_genes=8, seed=3)
    pairs = pd.DataFrame(
        {"element": ["e0", "e1", "e2", "e3"], "gene": ["g0", "g1", "g2", "g3"]}
    )
    restricted = restrict_effects_to_pairs(effects, pairs)
    full_rows = effects.merge(pairs, on=["element", "gene"])
    # Same p-values in both tables ...
    np.testing.assert_allclose(
        restricted.sort_values(["element", "gene"])["crt_p_value"].to_numpy(),
        full_rows.sort_values(["element", "gene"])["crt_p_value"].to_numpy(),
    )
    # ... and the restricted q-values are the BH of exactly those p-values.
    from scipy.stats import false_discovery_control

    expected = false_discovery_control(restricted["crt_p_value"].to_numpy(dtype=float), method="bh")
    np.testing.assert_allclose(restricted["crt_q_value"].to_numpy(), expected, rtol=1e-12, atol=0)
