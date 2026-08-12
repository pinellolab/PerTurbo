"""Tests for exact element/gene pair restriction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from perturbo.cli import _load_pairs_to_test
from perturbo.core import build_effect_indices


def test_load_pairs_to_test_accepts_tsv_and_deduplicates(tmp_path) -> None:
    path = tmp_path / "cis_pairs.tsv"
    pd.DataFrame(
        {
            "element": ["e1", "e1", "e2"],
            "gene": ["g2", "g2", "g1"],
            "distance": [10, 10, 20],
        }
    ).to_csv(path, sep="\t", index=False)

    pairs = _load_pairs_to_test(path)

    assert pairs.to_dict("records") == [
        {"element": "e1", "gene": "g2"},
        {"element": "e2", "gene": "g1"},
    ]


def test_build_effect_indices_preserves_exact_pairs_not_cartesian_product() -> None:
    pairs = pd.DataFrame({"element": ["e1", "e2"], "gene": ["g2", "g1"]})

    indices = build_effect_indices(
        pairs,
        pert_names=["e1", "e2"],
        gene_names=["g1", "g2"],
    )

    np.testing.assert_array_equal(indices, np.array([[0, 1], [1, 0]], dtype=np.int32))
    assert indices.shape[0] == 2


def test_build_effect_indices_reports_missing_names() -> None:
    pairs = pd.DataFrame({"element": ["missing"], "gene": ["g1"]})

    with pytest.raises(KeyError, match="elements=.*missing"):
        build_effect_indices(pairs, pert_names=["e1"], gene_names=["g1"])
