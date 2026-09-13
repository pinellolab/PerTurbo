"""Structured high-MOI nuisance corrections retain the dense SPA calculation."""

from __future__ import annotations

import numpy as np
import pytest

from perturbo import crt
from perturbo.core import PerTurboData
from perturbo._internal import bordered
from perturbo._internal.saddlepoint import (
    fit_high_moi_propensity_saddlepoint,
    fit_high_moi_stratified_saddlepoint,
)


def _problem(*, jitter: float = 0.25, batch: bool = True, no_reference: bool = False):
    rng = np.random.default_rng(1309)
    n, g, k, e = 180, 4, 6, 3
    codes = np.arange(n) % k
    if no_reference:
        codes = 1 + np.arange(n) % (k - 1)
    continuous = rng.normal(size=(n, 2))
    dummy = np.eye(k)[codes, 1:]
    # Interleaving columns checks that coefficients stay in original coordinates.
    design = np.column_stack([np.ones(n), continuous[:, 0], dummy, continuous[:, 1]])
    if not batch:
        design = np.column_stack([np.ones(n), continuous])
    beta = rng.normal(scale=0.2, size=(design.shape[1], g))
    beta[0] += 1.0
    offsets = rng.normal(scale=0.25, size=(n, 1))
    theta = np.arange(g, dtype=float) + 3.0
    mean = np.exp(offsets + design @ beta)
    counts = rng.negative_binomial(theta, theta / (theta + mean)).astype(np.float32)
    membership = rng.random((n, e)) < 0.2
    cells, elements = np.nonzero(membership)
    genes = tuple(f"g{i}" for i in range(g))
    names = tuple(f"e{i}" for i in range(e))
    nuisance = crt.ControlNuisance(
        counts=counts, nuisance_design=design, coefficients=beta,
        offsets=offsets, dispersion=theta,
        nuisance_names=tuple(f"z{i}" for i in range(design.shape[1])), gene_names=genes,
    )
    baseline = crt.CRTBaseline(
        nuisance=nuisance,
        null_check=crt.BaselineNullCheck(np.zeros(g), np.zeros(g, dtype=bool), 1e-6),
        curvature_jitter=jitter,
    )
    data = PerTurboData(
        counts=counts, pert_id=membership.astype(np.float32),
        pert_names=list(names), gene_names=list(genes),
    )
    propensity = crt.AllCellsPropensityFit(
        cell_index=cells, element_index=elements, element_names=names,
        testable=np.ones(e, dtype=bool),
        coefficients=np.column_stack([np.full(e, -1.4), [0.1, -0.2, 0.3]]),
        basis=np.column_stack([np.ones(n), continuous[:, 0]]), num_cells=n,
    )
    return baseline, data, propensity, codes


@pytest.mark.parametrize("jitter", [1e-8, 0.25])
def test_structured_control_block_preserves_original_ridge(jitter):
    baseline, _, _, _ = _problem(jitter=jitter)
    design = bordered.detect_bordered_design(baseline.nuisance.nuisance_design)
    assert design is not None
    dense = crt.control_block_for_genes(baseline, slice(1, 4))
    structured = crt.control_block_for_genes(baseline, slice(1, 4), bordered_design=design)
    assert dense.information is not None and dense.nuisance_direction is None
    assert structured.information is None
    expected = np.linalg.solve(dense.information, dense.nuisance_score.T[..., None])[..., 0].T
    np.testing.assert_allclose(structured.nuisance_direction, expected, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(structured.nuisance_score, dense.nuisance_score, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("family", ["propensity", "stratified"])
def test_high_moi_spa_direct_direction_matches_dense_inverse(family):
    baseline, _, propensity, codes = _problem()
    dense = crt.control_block_for_genes(baseline)
    design = bordered.detect_bordered_design(baseline.nuisance.nuisance_design)
    structured = crt.control_block_for_genes(baseline, bordered_design=design)
    kwargs = dict(
        score_residual=dense.score_residual,
        observation_weight=dense.observation_weight,
        nuisance_design=baseline.nuisance.nuisance_design,
        cell_index=propensity.cell_index, element_index=propensity.element_index,
        num_elements=len(propensity.element_names), screen_p_value=1.0, gene_block_size=4,
    )
    if family == "propensity":
        fit = fit_high_moi_propensity_saddlepoint
        kwargs.update(propensity_coefficients=propensity.coefficients, propensity_basis=propensity.basis)
    else:
        fit = fit_high_moi_stratified_saddlepoint
        kwargs.update(strata=codes, finite_population=False)
    expected = fit(
        **kwargs, nuisance_information_inverse=np.linalg.inv(dense.information),
        nuisance_score=dense.nuisance_score,
    )
    actual = fit(**kwargs, nuisance_direction=structured.nuisance_direction)
    assert actual.valid.all()
    np.testing.assert_array_equal(actual.valid, expected.valid)
    np.testing.assert_array_equal(actual.used_fallback, expected.used_fallback)
    np.testing.assert_allclose(actual.observed_sum, expected.observed_sum, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(actual.log_p_value, expected.log_p_value, rtol=2e-6, atol=2e-7)


def test_all_cells_routes_bordered_design_without_dense_information(monkeypatch):
    baseline, data, propensity, _ = _problem()
    kwargs = dict(propensity_fit=propensity, screen_p_value=1.0, gene_chunk_size=2, gene_block_size=4)
    with monkeypatch.context() as dense_patch:
        dense_patch.setattr(bordered, "detect_bordered_design", lambda _: None)
        expected = crt.run_crt_all_cells(baseline, data, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("The bordered high-MOI route allocated dense nuisance information.")

    monkeypatch.setattr(crt, "_nuisance_information", forbidden)
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    actual = crt.run_crt_all_cells(baseline, data, **kwargs)
    np.testing.assert_allclose(actual.observed_score, expected.observed_score, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        actual.parametric["saddlepoint"]["log_p_value"],
        expected.parametric["saddlepoint"]["log_p_value"], rtol=2e-6, atol=2e-7,
    )


def test_all_cells_retains_dense_fallback_for_continuous_only_design(monkeypatch):
    baseline, data, propensity, _ = _problem(batch=False)
    assert bordered.detect_bordered_design(baseline.nuisance.nuisance_design) is None
    original = crt._nuisance_information
    calls = []

    def recording(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(crt, "_nuisance_information", recording)
    result = crt.run_crt_all_cells(
        baseline, data, propensity_fit=propensity, screen_p_value=1.0, gene_block_size=4,
    )
    assert calls
    assert np.isfinite(result.parametric["saddlepoint"]["p_value"]).all()


def test_absent_reference_retains_dense_control_block_and_high_moi_route(monkeypatch):
    baseline, data, propensity, _ = _problem(no_reference=True)
    design = bordered.detect_bordered_design(baseline.nuisance.nuisance_design)
    assert design is not None and bordered.has_reference_dependency(design)
    block = crt.control_block_for_genes(baseline, bordered_design=design)
    assert block.information is not None and block.nuisance_direction is None
    original = crt._nuisance_information
    calls = []

    def recording(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(crt, "_nuisance_information", recording)
    result = crt.run_crt_all_cells(
        baseline, data, propensity_fit=propensity, screen_p_value=1.0, gene_block_size=4,
    )
    assert calls
    assert np.isfinite(result.parametric["saddlepoint"]["p_value"]).all()


@pytest.mark.parametrize("family", ["propensity", "stratified"])
def test_high_moi_rejects_ambiguous_or_misaligned_directions(family):
    baseline, _, propensity, codes = _problem()
    dense = crt.control_block_for_genes(baseline)
    kwargs = dict(
        score_residual=dense.score_residual, observation_weight=dense.observation_weight,
        nuisance_design=baseline.nuisance.nuisance_design,
        cell_index=propensity.cell_index, element_index=propensity.element_index,
        num_elements=len(propensity.element_names),
    )
    if family == "propensity":
        fit = fit_high_moi_propensity_saddlepoint
        kwargs.update(propensity_coefficients=propensity.coefficients, propensity_basis=propensity.basis)
    else:
        fit = fit_high_moi_stratified_saddlepoint
        kwargs.update(strata=codes)
    with pytest.raises(ValueError, match="not both"):
        fit(**kwargs, nuisance_direction=dense.nuisance_score,
            nuisance_information_inverse=np.linalg.inv(dense.information))
    with pytest.raises(ValueError, match="shape"):
        fit(**kwargs, nuisance_direction=dense.nuisance_score.T)
