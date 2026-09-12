"""CRT adapters keep sparse perturbation assignments indexed."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from perturbo.core import PerTurboData
from perturbo.crt import _element_membership, exclude_targets
from perturbo.sparse_design import (
    IndexedDesignMatrix,
    indexed_design_from_matrix,
    indexed_design_to_dense,
)


def test_all_cells_membership_reads_indexed_guides_without_densifying() -> None:
    guides = np.asarray(
        [[1, 0, 1], [0, 1, 0], [0, 0, 0], [1, 1, 0]], dtype=np.float32
    )
    guide_to_element = jnp.asarray([[1, 0], [0, 1], [1, 0]], dtype=jnp.float32)

    def data(guide_matrix):
        return PerTurboData(
            counts=jnp.ones((4, 1), dtype=jnp.float32),
            pert_id=indexed_design_from_matrix(guides @ np.asarray(guide_to_element)),
            pert_names=["a", "b"],
            gene_names=["g"],
            guide_matrix=guide_matrix,
            guide_names=["u", "v", "w"],
            guide_to_element=guide_to_element,
        )

    dense = _element_membership(data(jnp.asarray(guides)))
    indexed = _element_membership(data(indexed_design_from_matrix(guides)))
    for actual, expected in zip(indexed, dense, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_exclude_targets_remaps_an_indexed_design_in_compact_form() -> None:
    matrix = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 1]], dtype=np.float32
    )
    data = PerTurboData(
        counts=jnp.ones((5, 1), dtype=jnp.float32),
        pert_id=indexed_design_from_matrix(matrix),
        pert_names=["NTC", "a", "b"],
        gene_names=["g"],
        size_factors=jnp.zeros((5, 1), dtype=jnp.float32),
    )

    kept = exclude_targets(data, ["NTC"])
    assert kept is not None and isinstance(kept.pert_id, IndexedDesignMatrix)
    assert kept.pert_names == ["a", "b"]
    np.testing.assert_array_equal(
        np.asarray(indexed_design_to_dense(kept.pert_id)),
        np.asarray([[1, 0], [0, 1], [1, 1], [0, 1]], dtype=np.float32),
    )
