import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control

from perturbo.fdr import FdrFamilies
from perturbo.results import load_pairs_to_test, write_standard_element_effects_parquet


def bh(values):
    values = np.asarray(values)
    out = np.full(values.shape, np.nan)
    mask = np.isfinite(values)
    if mask.any():
        out[mask] = false_discovery_control(values[mask])
    return out


@pytest.mark.parametrize("size", [1, 5, 250_000])
@pytest.mark.parametrize("primary", ["saddlepoint", "resampling"])
def test_streamed_families_and_restricted_output(tmp_path, size, primary):
    # Nonalphabetical axes, tied p-values, wholly untested control and tiny tail.
    elements, genes = ["B", "NTC", "A"], ["z", "a", "t", "b"]
    tails = np.array([[0.08, 0.01, np.nan, 1e-210], [np.nan] * 4, [0.02, 0.02, 1, 0.95]])
    empirical = np.where(np.isfinite(tails), (tails + 1) / 2, np.nan)
    empirical[0, 1] = np.nan  # No switching tests per row.
    columns = {
        "crt_saddlepoint_p_value": tails,
        "crt_saddlepoint_q_value": bh(tails),
        "crt_p_value": empirical,
        "crt_q_value": bh(empirical),
    }
    p = tails if primary == "saddlepoint" else empirical
    pairs = pd.DataFrame(
        [("A", "a"), ("B", "b"), ("B", "a"), ("A", "a"), ("unknown", "z"), ("NTC", "t")], columns=["element", "gene"]
    )
    fdr = FdrFamilies(
        columns, primary=primary, elements=elements, genes=genes, cis_pairs=pairs, cis_source="candidates.tsv"
    )
    path = tmp_path / "effects.parquet"
    restricted = write_standard_element_effects_parquet(
        path,
        method="perturbo",
        element_names=elements,
        gene_names=genes,
        effect_loc=np.zeros_like(p),
        effect_scale=np.ones_like(p),
        extra_columns=columns,
        requested_pairs=pairs.iloc[[1]],
        fdr=fdr,
        row_block_size=size,
    )
    frame = pd.read_parquet(path)
    np.testing.assert_array_equal(frame.p_value, p.ravel())
    np.testing.assert_array_equal(frame.q_global, bh(p).ravel())
    np.testing.assert_allclose(frame.q_element, np.stack([bh(row) for row in p]).ravel(), atol=0)
    included = pd.MultiIndex.from_frame(frame[["element", "gene"]]).isin(pd.MultiIndex.from_frame(pairs))
    expected = np.full(p.size, np.nan)
    expected[included] = bh(p.ravel()[included])
    np.testing.assert_allclose(frame.q_cis, expected, atol=0)
    assert (frame[["p_value", "q_global", "q_element", "q_cis"]].dtypes == "float64").all()
    assert fdr.metadata["cis_family"] == dict(
        source="candidates.tsv", declared_pairs=5, matched_pairs=4, finite_p=int(np.isfinite(expected).sum())
    )
    assert fdr.metadata["result_fields"]["q_element"]["finite_p"] == np.isfinite(p).sum(axis=1).tolist()
    assert fdr.metadata["result_fields"]["q_global"]["finite_p"] == np.isfinite(p).sum()
    for name in ("p_value", "q_global", "q_element", "q_cis"):
        assert restricted[name].iat[0] == frame.loc[3, name]
    assert restricted.crt_saddlepoint_q_value.iat[0] == tails[0, 3]


@pytest.mark.parametrize("cis", [None, [("NTC", "g")], [("other", "g")]])
def test_absent_or_unsupported_cis_omits_column(cis):
    p = np.array([[0.25], [np.nan]])
    pairs = pd.DataFrame(cis, columns=["element", "gene"]) if cis is not None else None
    fdr = FdrFamilies(
        {"crt_p_value": p, "crt_q_value": p},
        primary="resampling",
        elements=["E", "NTC"],
        genes=["g"],
        cis_pairs=pairs,
        cis_source="cis.tsv",
    )
    assert "q_cis" not in fdr.metadata["result_fields"]
    assert all("q_cis" not in block for block in fdr.blocks(1))
    assert ("cis_family" in fdr.metadata) == (pairs is not None)


def test_preserve_literal_identifiers(tmp_path):
    path = tmp_path / "cis.tsv"
    path.write_text("element\tgene\n001\tNA\n010\t42\n")
    assert load_pairs_to_test(path).to_numpy().tolist() == [["001", "NA"], ["010", "42"]]


def test_cis_with_disabled_testing_fails_before_loading_inputs(tmp_path):
    from perturbo.cli import main

    with pytest.raises(ValueError, match="cis-pairs requires significance testing"):
        main(["--input", "absent.h5ad", "--out-dir", str(tmp_path), "--cis-pairs", "missing.csv", "--no-crt"])
