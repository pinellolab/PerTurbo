"""Tests for the gene-chunking scaffolding shared across drivers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from perturbo._internal.chunked_runner import (
    apply_global_benjamini_hochberg,
    iter_gene_chunks,
    replace_gene_axis,
)


def test_iter_gene_chunks_covers_every_gene_without_overlap():
    chunks = iter_gene_chunks(10, 4)

    assert chunks == [slice(0, 4), slice(4, 8), slice(8, 10)]
    covered = np.zeros(10, dtype=bool)
    for chunk in chunks:
        covered[chunk] = True
    assert covered.all()


def test_iter_gene_chunks_handles_a_chunk_size_larger_than_the_panel():
    assert iter_gene_chunks(5, 100) == [slice(0, 5)]


def test_iter_gene_chunks_rejects_bad_arguments():
    with pytest.raises(ValueError):
        iter_gene_chunks(-1, 10)
    with pytest.raises(ValueError):
        iter_gene_chunks(10, 0)


@dataclass(frozen=True)
class _FakeDesign:
    """Stands in for LowMOIDesign/HighMOIDesign: same three gene-axis names."""

    counts: np.ndarray
    dispersion: np.ndarray
    gene_names: tuple[str, ...]
    target_codes: np.ndarray


def test_replace_gene_axis_swaps_only_the_gene_indexed_fields():
    original = _FakeDesign(
        counts=np.ones((5, 3)),
        dispersion=np.array([1.0, 2.0, 3.0]),
        gene_names=("a", "b", "c"),
        target_codes=np.arange(5),
    )

    updated = replace_gene_axis(
        original, counts=np.zeros((5, 2)), dispersion=np.array([9.0, 8.0]), gene_names=("x", "y")
    )

    assert updated.gene_names == ("x", "y")
    np.testing.assert_array_equal(updated.dispersion, [9.0, 8.0])
    np.testing.assert_array_equal(updated.counts, np.zeros((5, 2)))
    # Cell-indexed fields must carry over unchanged.
    np.testing.assert_array_equal(updated.target_codes, original.target_codes)


def test_apply_global_benjamini_hochberg_matches_the_manual_correction():
    p_values = np.array([0.001, 0.008, 0.039, 0.041, 0.042])
    frame = pd.DataFrame({"p": p_values})

    result = apply_global_benjamini_hochberg(frame, {"p": "q"})

    # Hand-computed BH: p * n / rank, then a running minimum from the largest
    # rank down so q is monotone in the sorted p-values.
    n = p_values.size
    raw = p_values * n / np.arange(1, n + 1)
    expected = np.minimum.accumulate(raw[::-1])[::-1]
    np.testing.assert_allclose(result["q"].to_numpy(), expected)


def test_apply_global_benjamini_hochberg_is_more_conservative_with_more_tests():
    """This is why a per-chunk correction is wrong: q shrinks as the false-BH
    denominator drops. A chunk sees only its own rows, so correcting each
    chunk separately understates how many hypotheses are actually in play.
    """

    small_frame = pd.DataFrame({"p": [0.01, 0.5, 0.9]})
    rng = np.random.default_rng(0)
    large_frame = pd.DataFrame({"p": np.concatenate([[0.01, 0.5, 0.9], rng.uniform(0.2, 1.0, size=997)])})

    small_q = apply_global_benjamini_hochberg(small_frame, {"p": "q"})["q"].iloc[0]
    large_q = apply_global_benjamini_hochberg(large_frame, {"p": "q"})["q"].iloc[0]

    assert large_q > small_q


def test_apply_global_benjamini_hochberg_skips_absent_columns():
    frame = pd.DataFrame({"p": [0.01, 0.5]})

    result = apply_global_benjamini_hochberg(frame, {"p": "q", "parametric_p": "parametric_q"})

    assert "q" in result.columns
    assert "parametric_q" not in result.columns
