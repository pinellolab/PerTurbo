"""Guide-level summaries must not materialize the whole posterior.

The earlier implementation drew 64 samples of the full (elements, genes) effect
matrix and contracted them against the guide map. For a high-MOI screen with
~13,000 guides and ~13,000 genes that intermediate is tens of terabytes, so
guide-level output was unavailable at exactly the scale that asks for it. Two of
the three strategies are linear in independent Normals and need no draws at all;
the nonlinear one is sampled in bounded blocks.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from perturbo import core


def _inputs(num_elements=3, num_guides=5, num_genes=4, seed=0):
    rng = np.random.default_rng(seed)
    guide_to_element = np.zeros((num_guides, num_elements), dtype=np.float32)
    for guide in range(num_guides):
        guide_to_element[guide, guide % num_elements] = 1.0
    params = {
        "beta_auto_loc": rng.normal(size=(num_elements, num_genes)).astype(np.float32),
        "beta_auto_scale": rng.uniform(0.1, 0.5, size=(num_elements, num_genes)).astype(np.float32),
    }
    data = SimpleNamespace(guide_to_element=guide_to_element)
    return params, data, guide_to_element


def test_shared_strategy_returns_no_derived_summary():
    """Under ``shared`` the guide effect is the element effect, so there is nothing to derive.

    The model sets ``guide_effect = guide_to_element @ beta``. Materializing that as a
    (guides, genes) array duplicates the element table and, at screen scale, costs tens
    of gigabytes to say nothing. Consumers join instead; see
    ``PerTurboModel._guide_effect_payload``.
    """
    params, data, _ = _inputs()
    assert core._summarize_stage2_guide_posteriors(params, data=data, guide_effect_strategy="shared") == {}


def test_offset_strategy_is_closed_form_and_matches_sampling():
    """``offset`` does differ from its element, and adds an independent Normal."""
    params, data, guide_to_element = _inputs()
    num_guides, num_genes = guide_to_element.shape[0], params["beta_auto_loc"].shape[1]
    rng = np.random.default_rng(2)
    params["guide_offset_auto_loc"] = rng.normal(size=(num_guides, num_genes)).astype(np.float32)
    params["guide_offset_auto_scale"] = rng.uniform(0.1, 0.4, size=(num_guides, num_genes)).astype(np.float32)

    summary = core._summarize_stage2_guide_posteriors(params, data=data, guide_effect_strategy="offset")
    draws = params["beta_auto_loc"] + params["beta_auto_scale"] * rng.normal(
        size=(200_000, *params["beta_auto_loc"].shape)
    )
    offsets = params["guide_offset_auto_loc"] + params["guide_offset_auto_scale"] * rng.normal(
        size=(200_000, num_guides, num_genes)
    )
    sampled = np.einsum("qe,seg->sqg", guide_to_element, draws) + offsets
    np.testing.assert_allclose(summary["guide_effect_mean"], sampled.mean(axis=0), atol=5e-3)
    np.testing.assert_allclose(summary["guide_effect_scale"], sampled.std(axis=0), rtol=2e-2)


def test_relative_summary_samples_in_bounded_blocks(monkeypatch):
    params, data, guide_to_element = _inputs(num_elements=40, num_guides=60, num_genes=8)
    params["guide_relative_efficiency_auto_loc"] = np.zeros((60, 8), dtype=np.float32)
    params["guide_relative_efficiency_auto_scale"] = np.full((60, 8), 0.3, dtype=np.float32)

    seen: list[tuple[int, int]] = []
    original = core._summarize_relative_guide_block

    def _record(beta_loc, beta_scale, relative_loc, *args, **kwargs):
        seen.append((int(beta_loc.shape[0]), int(relative_loc.shape[0])))
        return original(beta_loc, beta_scale, relative_loc, *args, **kwargs)

    monkeypatch.setattr(core, "_summarize_relative_guide_block", _record)
    summary = core._summarize_stage2_guide_posteriors(
        params, data=data, guide_effect_strategy="relative", element_block_size=10, guide_block_size=4
    )
    assert seen, "the relative path is expected to sample"
    assert max(elements for elements, _ in seen) <= 10
    assert max(guides for _, guides in seen) <= 4
    assert summary["guide_effect_mean"].shape == (60, 8)
    assert np.isfinite(summary["guide_relative_efficiency_mean"]).all()


def test_unknown_strategy_is_refused():
    params, data, _ = _inputs()
    with pytest.raises(ValueError, match="Unknown guide_effect_strategy"):
        core._summarize_stage2_guide_posteriors(params, data=data, guide_effect_strategy="nonsense")


def test_no_guide_map_means_no_guide_summary():
    params, _, _ = _inputs()
    assert core._summarize_stage2_guide_posteriors(
        params, data=SimpleNamespace(guide_to_element=None), guide_effect_strategy="shared"
    ) == {}
