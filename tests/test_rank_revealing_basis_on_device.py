"""The device basis is the host basis: same rank, same column space, orthonormal."""

from __future__ import annotations

import numpy as np

from perturbo._internal.high_moi.resampling import propensity_basis


def _host_reference(design):
    raw = np.asarray(design)
    eps_src = np.finfo(raw.dtype).eps if np.issubdtype(raw.dtype, np.floating) else np.finfo(np.float64).eps
    m = np.asarray(raw, dtype=np.float64)
    norms = np.linalg.norm(m, axis=0); nz = norms > 0
    if not nz.any():
        return np.zeros((m.shape[0], 0), dtype=np.float32)
    scaled = m[:, nz] / norms[nz]
    left, s, _ = np.linalg.svd(scaled, full_matrices=False)
    tol = max(max(scaled.shape) * np.finfo(np.float64).eps, 8.0 * np.sqrt(scaled.shape[1]) * eps_src) * float(s[0])
    return left[:, : int((s > tol).sum())].astype(np.float32)


def _same_space(a, b, atol=1e-4):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.shape[1] == 0:
        return
    np.testing.assert_allclose(a.T @ a, np.eye(a.shape[1]), atol=atol)   # orthonormal
    np.testing.assert_allclose(b - a @ (a.T @ b), 0.0, atol=atol)         # b in span(a)
    np.testing.assert_allclose(a - b @ (b.T @ a), 0.0, atol=atol)         # a in span(b)


def _designs():
    rng = np.random.default_rng(0)
    n = 3000
    full = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    batch = rng.integers(0, 12, n)
    onehot_plus_intercept = np.column_stack([np.ones(n), rng.normal(size=n), np.eye(12)[batch]])  # rank-deficient by 1
    dependent = np.column_stack([full, full[:, 1] - 2 * full[:, 2]])
    with_zero = np.column_stack([full, np.zeros(n)])
    scaled_units = np.column_stack([np.ones(n), 1e6 * rng.normal(size=n), 1e-6 * rng.normal(size=n)])
    return {"full": full, "onehot+intercept": onehot_plus_intercept, "dependent": dependent,
            "zero column": with_zero, "mixed units": scaled_units}


def test_device_basis_matches_host_basis_on_every_design_class():
    for name, d in _designs().items():
        for dtype in (np.float32, np.float64):
            ref = _host_reference(d.astype(dtype))
            got = np.asarray(propensity_basis(d.astype(dtype)))
            assert got.shape[1] == ref.shape[1], (name, dtype, got.shape, ref.shape)
            _same_space(got, ref)


def test_one_hot_beside_an_intercept_loses_exactly_one_direction():
    d = _designs()["onehot+intercept"]
    assert np.asarray(propensity_basis(d)).shape[1] == d.shape[1] - 1


def test_all_zero_design_gives_an_empty_basis():
    assert np.asarray(propensity_basis(np.zeros((50, 3)))).shape == (50, 0)


def test_many_small_blocks_give_the_same_basis_as_one_block(monkeypatch):
    from perturbo._internal.high_moi import resampling as module
    d = _designs()["onehot+intercept"]
    whole = np.asarray(propensity_basis(d))
    monkeypatch.setattr(module, "_BASIS_ROWS_PER_BLOCK", 97)   # ~31 blocks, ragged last one
    blocked = np.asarray(propensity_basis(d))
    _same_space(blocked, whole)
    _same_space(blocked, _host_reference(d))
