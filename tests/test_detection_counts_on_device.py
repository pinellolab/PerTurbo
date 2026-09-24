"""The device detection counts reproduce the host NumPy reference."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from perturbo import crt as crt_module
from perturbo.crt import _informative_cell_counts, _membership_matrix


def _reference(counts, nuisance_design, offsets, coefficients, dispersion, membership):
    design = np.asarray(nuisance_design, dtype=np.float32)
    offset_matrix = np.asarray(offsets, dtype=np.float32)
    if offset_matrix.ndim == 1:
        offset_matrix = offset_matrix[:, None]
    theta_row = np.asarray(dispersion, dtype=np.float32).reshape(1, -1)
    detected = design @ np.asarray(coefficients, dtype=np.float32)
    detected += offset_matrix
    np.clip(detected, -crt_module._INFORMATIVE_ETA_CLIP, crt_module._INFORMATIVE_ETA_CLIP, out=detected)
    np.exp(detected, out=detected); detected /= theta_row
    np.log1p(detected, out=detected); detected *= -theta_row
    np.expm1(detected, out=detected); np.negative(detected, out=detected)
    expected = membership @ detected
    observed = membership @ (np.asarray(counts) > 0).astype(np.float32)
    return np.asarray(observed, dtype=np.float64), np.asarray(expected, dtype=np.float64)


def _problem(seed, *, cells=900, genes=37, q=12, elements=25, memberships_per_cell):
    rng = np.random.default_rng(seed)
    cell_index, element_index = [], []
    for cell in range(cells):
        k = memberships_per_cell(rng)
        for e in rng.choice(elements, size=k, replace=False):
            cell_index.append(cell); element_index.append(e)
    member_cells, membership = _membership_matrix(np.asarray(cell_index), np.asarray(element_index), elements)
    design = np.column_stack([np.ones(member_cells.size), rng.normal(size=(member_cells.size, q - 1))])
    coef = rng.normal(scale=0.5, size=(q, genes)); coef[0] += 1.0
    offsets = rng.normal(scale=0.3, size=(member_cells.size, 1))
    theta = np.exp(rng.normal(1.0, 1.5, size=genes))  # spans small to large dispersion
    mu = np.exp(design @ coef + offsets)
    counts = rng.negative_binomial(theta, theta / (theta + mu)).astype(np.float32)
    return dict(counts=counts, nuisance_design=design, offsets=offsets, coefficients=coef, dispersion=theta, membership=membership)


def _check(kw):
    obs_ref, exp_ref = _reference(**kw)
    obs, exp = _informative_cell_counts(**kw)
    np.testing.assert_array_equal(obs, obs_ref)                       # integer counts: exact
    np.testing.assert_allclose(exp, exp_ref, rtol=2e-5, atol=1e-4)    # float32 elementwise chain


def test_low_moi_partition_matches_host_reference():
    _check(_problem(1, memberships_per_cell=lambda rng: 1))


def test_high_moi_multi_membership_matches_host_reference():
    _check(_problem(2, memberships_per_cell=lambda rng: int(rng.integers(1, 6))))


def test_row_chunking_does_not_change_the_answer(monkeypatch):
    kw = _problem(3, memberships_per_cell=lambda rng: int(rng.integers(1, 4)))
    whole = _informative_cell_counts(**kw)
    monkeypatch.setattr(crt_module, "_DETECTION_ROWS_PER_CHUNK", 5 * kw["counts"].shape[1])  # ~5 memberships per chunk
    crt_module._detection_counts_on_device.clear_cache() if hasattr(crt_module._detection_counts_on_device, "clear_cache") else None
    chunked = _informative_cell_counts(**kw)
    np.testing.assert_array_equal(whole[0], chunked[0])
    np.testing.assert_allclose(whole[1], chunked[1], rtol=1e-6, atol=1e-6)


def test_padded_chunks_and_empty_elements_match_host_reference(monkeypatch):
    rng = np.random.default_rng(4)
    genes, q, elements = 5, 3, 7
    # Thirteen memberships require three five-row chunks. Elements 1, 4, and 6
    # are intentionally empty, while repeated cells exercise high-MOI membership.
    cell_index = np.array([0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10])
    element_index = np.array([0, 2, 2, 3, 0, 5, 3, 5, 2, 0, 3, 5, 0])
    member_cells, membership = _membership_matrix(cell_index, element_index, elements)
    design = np.column_stack([np.ones(member_cells.size), rng.normal(size=(member_cells.size, q - 1))])
    coefficients = rng.normal(scale=0.4, size=(q, genes))
    offsets = rng.normal(scale=0.2, size=(member_cells.size, 1))
    dispersion = np.exp(rng.normal(size=genes))
    mean = np.exp(design @ coefficients + offsets)
    counts = rng.negative_binomial(dispersion, dispersion / (dispersion + mean)).astype(np.float32)
    problem = dict(
        counts=counts,
        nuisance_design=design,
        offsets=offsets,
        coefficients=coefficients,
        dispersion=dispersion,
        membership=membership,
    )
    monkeypatch.setattr(crt_module, "_DETECTION_ROWS_PER_CHUNK", 5 * genes)
    if hasattr(crt_module._detection_counts_on_device, "clear_cache"):
        crt_module._detection_counts_on_device.clear_cache()
    _check(problem)
