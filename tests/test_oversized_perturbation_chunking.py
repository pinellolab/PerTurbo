"""A perturbation larger than the chunk cap gets its own chunk instead of aborting.

Screens exist with tens of thousands of cells behind a single perturbation. The cap
is a target for the size of a chunk, not a statement about which screens the tool
accepts, and one perturbation's cells cannot be split across chunks anyway: they are
all needed to estimate its effect.
"""
import numpy as np

from perturbo.core import _construct_perturbation_chunks


def _membership(sizes):
    out, start = [], 0
    for size in sizes:
        out.append(np.arange(start, start + size, dtype=np.int64))
        start += size
    return out


def test_a_single_oversized_perturbation_gets_its_own_chunk(capsys):
    names = ["small_a", "huge", "small_b"]
    membership = _membership([100, 6_000, 100])
    chunks = _construct_perturbation_chunks(
        names, membership, max_chunk_size=1_000, max_perturbations_per_chunk=None
    )
    by_name = {name: chunk for chunk in chunks for name in chunk.pert_names}
    assert set(by_name) == set(names), "every perturbation is still tested"
    huge = by_name["huge"]
    assert huge.pert_names == ["huge"], "the oversized perturbation is alone in its chunk"
    assert huge.cell_indices.size == 6_000, "and keeps all of its cells"
    message = capsys.readouterr().out
    assert "exceed --max-chunk-size" in message and "huge (6,000 cells)" in message


def test_ordinary_perturbations_still_respect_the_cap():
    names = [f"p{i}" for i in range(6)]
    chunks = _construct_perturbation_chunks(
        names, _membership([300] * 6), max_chunk_size=1_000, max_perturbations_per_chunk=None
    )
    assert all(chunk.cell_indices.size <= 1_000 for chunk in chunks)
    assert sum(len(chunk.pert_names) for chunk in chunks) == 6


def test_several_oversized_perturbations_are_each_isolated_and_reported(capsys):
    names = ["a", "big1", "b", "big2"]
    chunks = _construct_perturbation_chunks(
        names, _membership([50, 4_000, 50, 5_000]), max_chunk_size=500, max_perturbations_per_chunk=None
    )
    isolated = {chunk.pert_names[0] for chunk in chunks if len(chunk.pert_names) == 1}
    assert {"big1", "big2"} <= isolated
    out = capsys.readouterr().out
    assert "2 perturbation(s) exceed" in out
    assert "5,000 cells" in out, "the report names the largest, which sets peak memory"
