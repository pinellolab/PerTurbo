"""The compact low-MOI propensity path forms own-cell logits in row chunks.

``fit_low_moi_propensity_saddlepoint`` used to build the own-cell logit vector as
``sum(basis[own_rows] * coefficients[own_codes], axis=1)``. That asks XLA for two
``(own cells, basis width)`` float64 operands and their product on the way to a
``(own cells,)`` result: at X-Atlas/Orion scale, 3,243,392 own cells over a
109-column basis is 2.63 GiB apiece to produce 26 MB, and it drove that screen
into GPU OOM. The chunked form accumulates the same dot products on the row
budget the pool projection already uses.

These tests pin the property that matters: the answer must not depend on how many
chunks the work is cut into. The chunk count is derived from
``_PROJECTION_ROWS_PER_CHUNK``, so forcing it down to a handful of rows is what
exercises the loop - the production constant leaves these fixtures in a single
chunk, which is why the whole path was previously covered only in its one-chunk
form.
"""

from __future__ import annotations

import numpy as np
import pytest

from perturbo._internal import saddlepoint as saddlepoint_module
from perturbo._internal.saddlepoint import fit_low_moi_propensity_saddlepoint


def _compact_problem(seed: int = 11, *, num_targets: int = 6, num_genes: int = 9):
    """A control-anchored screen described by a shared basis and per-target coefficients."""

    rng = np.random.default_rng(seed)
    num_controls = 400
    own_counts = rng.integers(8, 40, size=num_targets)
    codes = np.concatenate(
        [np.full(num_controls, -1), np.repeat(np.arange(num_targets), own_counts)]
    )
    num_cells = codes.size
    control = codes < 0
    contribution = rng.normal(size=(num_cells, num_genes))
    for target in range(num_targets):
        contribution[codes == target, target % num_genes] += 1.2
    basis = np.column_stack(
        [
            np.ones(num_cells),
            rng.normal(size=num_cells),
            rng.normal(scale=0.4, size=num_cells),
        ]
    )
    coefficients = np.column_stack(
        [
            np.log(own_counts / num_controls),
            rng.normal(scale=0.3, size=num_targets),
            rng.normal(scale=0.3, size=num_targets),
        ]
    )
    return dict(
        contribution=contribution,
        target_codes=codes,
        control_mask=control,
        propensity_coefficients=coefficients,
        propensity_basis=basis,
        num_targets=num_targets,
        screen_p_value=1.0,
    )


def _fit_with_row_budget(problem, budget, monkeypatch):
    monkeypatch.setattr(saddlepoint_module, "_PROJECTION_ROWS_PER_CHUNK", budget)
    return fit_low_moi_propensity_saddlepoint(**problem)


@pytest.mark.parametrize("budget", [3, 17, 64])
def test_own_logit_chunking_does_not_change_the_fit(budget, monkeypatch):
    problem = _compact_problem()
    own_cells = int(np.count_nonzero(problem["target_codes"] >= 0))
    basis_width = problem["propensity_basis"].shape[1]

    single = _fit_with_row_budget(problem, 1 << 40, monkeypatch)
    chunked = _fit_with_row_budget(problem, budget, monkeypatch)

    rows_per_chunk = max(1, budget // basis_width)
    assert rows_per_chunk < own_cells, "the budget must force more than one chunk"

    assert single.valid.any(), "the fixture must produce a usable fit to compare"
    np.testing.assert_array_equal(single.valid, chunked.valid)
    finite = np.isfinite(single.log_p_value) & np.isfinite(chunked.log_p_value)
    assert finite.any(), "saddlepoint-only runs report the fitted tail, not an empirical one"
    np.testing.assert_allclose(
        single.log_p_value[finite], chunked.log_p_value[finite], rtol=1e-12, atol=0.0
    )
    for field in ("null_mean", "null_variance"):
        left = np.asarray(getattr(single, field))
        right = np.asarray(getattr(chunked, field))
        mask = np.isfinite(left) & np.isfinite(right)
        np.testing.assert_allclose(left[mask], right[mask], rtol=1e-12, atol=0.0)


def test_a_single_own_cell_per_chunk_is_still_exact(monkeypatch):
    """One row per chunk is the degenerate end of the loop, and the slowest path."""

    problem = _compact_problem(seed=13, num_targets=4, num_genes=5)
    single = _fit_with_row_budget(problem, 1 << 40, monkeypatch)
    one_row = _fit_with_row_budget(problem, 1, monkeypatch)

    np.testing.assert_array_equal(single.valid, one_row.valid)
    finite = np.isfinite(single.log_p_value) & np.isfinite(one_row.log_p_value)
    assert finite.any()
    np.testing.assert_allclose(
        single.log_p_value[finite], one_row.log_p_value[finite], rtol=1e-12, atol=0.0
    )


def test_the_production_budget_leaves_this_fixture_in_one_chunk():
    """Why the chunked branch needs its own test rather than riding the suite."""

    problem = _compact_problem()
    own_cells = int(np.count_nonzero(problem["target_codes"] >= 0))
    basis_width = problem["propensity_basis"].shape[1]
    rows_per_chunk = max(
        1, int(saddlepoint_module._PROJECTION_ROWS_PER_CHUNK // max(basis_width, 1))
    )
    assert rows_per_chunk >= own_cells
