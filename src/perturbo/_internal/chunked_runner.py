"""Scaffolding shared by gene-chunked drivers, low-MOI and high-MOI alike.

A transcriptome-wide count matrix does not fit densely in memory, but every
inference layer here - dispersion, effect fit, and the CRT null - is
mathematically independent across genes, so chunking the gene axis changes no
formula: it is a decomposition, not an approximation. It is not, however,
bit-reproducible across different chunk sizes - JAX's batched reductions
(segment_sum, einsum) are not guaranteed bit-identical when the batch width
changes, so the same gene fit at chunk width 97 versus 400 differs at the
float32 ULP level. Measured on the Gasperini pilot this shows up as effect
estimates agreeing to ~1e-6 relative and, rarely, a CRT p-value landing on the
other side of a resampled-score tie (about 0.07% of pairs at 49 resamples).
Caching the whole panel in memory and slicing it *is* bit-exact against
slicing on the fly at a fixed chunk size, since both feed the fit functions
the identical float32 array - only cross-chunk-size comparisons see drift.
Two design dataclasses drive this today,
:class:`~perturbo._internal.low_moi.LowMOIDesign` and
:class:`~perturbo._internal.high_moi.HighMOIDesign`, and both happen to
carry exactly three gene-indexed fields under the same names - ``counts``,
``dispersion``, ``gene_names`` - with every other field (cell, target, or
element layout) invariant across which genes are in the chunk. That symmetry
is what lets :func:`replace_gene_axis` be genuinely generic rather than
adapted per backend.

What is deliberately *not* here is a shared interface for fitting effects or
running the CRT. Those calls differ in real ways between backends - different
keyword arguments, a segment solver versus a matrix-free Newton-CG solve,
different score-permutation signatures - and forcing them behind one
interface would trade an honest difference for an abstraction that has to be
worked around on both sides. Drivers keep their own calls inline and use this
module only for the parts that are identical in shape: chunk iteration, the
gene-axis replace, and the global Benjamini-Hochberg step that has to happen
once over the concatenated table rather than once per chunk.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from perturbo._internal.score_resampling import _benjamini_hochberg

__all__ = ["iter_gene_chunks", "replace_gene_axis", "apply_global_benjamini_hochberg"]


def iter_gene_chunks(num_genes: int, chunk_size: int) -> list[slice]:
    """Gene-axis slices of width ``chunk_size`` covering ``0:num_genes``."""

    if num_genes < 0:
        raise ValueError("num_genes must be non-negative.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    return [slice(start, min(start + chunk_size, num_genes)) for start in range(0, num_genes, chunk_size)]


def replace_gene_axis(design, *, counts, dispersion, gene_names):
    """Swap the gene-indexed fields of a chunked design, leaving cells alone.

    Works for any design dataclass exposing exactly ``counts``, ``dispersion``,
    and ``gene_names`` as its gene axis - both :class:`LowMOIDesign` and
    :class:`HighMOIDesign` qualify today. Everything else (target codes,
    element pairs, nuisance design, offsets, control masks) is gene-invariant
    and is carried over unchanged, which is what makes chunking exact: a
    design built from chunk two differs from one built from chunk one only in
    which genes it can see.
    """

    return dataclasses.replace(design, counts=counts, dispersion=dispersion, gene_names=gene_names)


def apply_global_benjamini_hochberg(frame: pd.DataFrame, p_to_q: dict[str, str]) -> pd.DataFrame:
    """Overwrite q-value columns with BH computed once over the whole frame.

    Each chunk's own q-values (if a per-chunk fit produced any) treat that
    chunk as if it were the entire family of tests, which is wrong whenever
    there is more than one chunk - the correction has to see every hypothesis
    at once. This recomputes ``p_to_q.values()`` from ``p_to_q.keys()`` in
    place over the full concatenated table; columns absent from ``frame`` are
    skipped rather than raising, since not every driver produces a parametric
    p-value column.
    """

    for p_column, q_column in p_to_q.items():
        if p_column not in frame.columns:
            continue
        frame[q_column] = _benjamini_hochberg(frame[p_column].to_numpy())
    return frame
