"""A batch-confined element must not get a degenerate all-cells null.

With a categorical batch in the design, an element that appears in only some
levels is separated by the level indicators: the unpenalized logistic MLE does
not exist, the IRLS walks the coefficients outward every iteration, and the fit
can collapse onto a zero/one assignment whose Bernoulli null carries no
variance. That turns an unperturbed element into 1e-300 p-values.

These tests build a screen with no perturbation effect at all, so every pair is
a true null, and check that the confined elements' far-gene p-values are
uniform while the elements spread over every level are left alone.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from perturbo import crt
from perturbo.core import PerTurboData


def _screen(*, seed: int = 11, num_cells: int = 6000, num_levels: int = 8, num_genes: int = 24,
            num_spread: int = 12, num_confined: int = 6, element_cells: int = 120):
    """An all-cells screen with no real effects, half its elements lane-confined."""

    rng = np.random.default_rng(seed)
    weights = np.full(num_levels, 1.0)
    weights[0] = 0.6          # a small level, as a shallow sequencing lane is
    weights /= weights.sum()
    codes = rng.choice(num_levels, size=num_cells, p=weights)
    continuous = rng.normal(size=num_cells)
    # Reference coding: the last level is the dropped one.
    design = np.column_stack([np.ones(num_cells), continuous, np.eye(num_levels)[codes][:, :-1]])

    beta = np.zeros((design.shape[1], num_genes))
    beta[0] = rng.uniform(1.0, 3.0, size=num_genes)
    beta[1] = rng.normal(scale=0.2, size=num_genes)
    beta[2:] = rng.normal(scale=0.4, size=(num_levels - 1, num_genes))   # real batch structure
    offsets = rng.normal(scale=0.3, size=(num_cells, 1))
    theta = rng.uniform(2.0, 10.0, size=num_genes)
    mean = np.exp(offsets + design @ beta)
    counts = rng.negative_binomial(theta, theta / (theta + mean)).astype(np.float32)

    num_elements = num_spread + num_confined
    membership = np.zeros((num_cells, num_elements), dtype=bool)
    confined_level = np.full(num_elements, -1)
    for element in range(num_spread):
        membership[rng.choice(num_cells, size=element_cells, replace=False), element] = True
    for offset in range(num_confined):
        element = num_spread + offset
        level = offset % num_levels
        confined_level[element] = level
        pool = np.flatnonzero(codes == level)
        membership[rng.choice(pool, size=min(element_cells, pool.size), replace=False), element] = True

    gene_names = [f"g{index}" for index in range(num_genes)]
    pert_names = [f"e{index}" for index in range(num_elements)]
    nuisance = crt.polish_baseline_to_null_mode(
        crt.ControlNuisance(
            counts=counts, nuisance_design=design, coefficients=beta, offsets=offsets,
            dispersion=theta,
            nuisance_names=tuple(f"z{index}" for index in range(design.shape[1])),
            gene_names=tuple(gene_names),
        ),
        curvature_jitter=1e-8,
    )
    check = crt.check_baseline_is_null_mode(nuisance, step_tolerance=5e-2, curvature_jitter=1e-8)
    assert check.ok, "the fixture's own baseline must sit on the null mode"
    baseline = crt.CRTBaseline(nuisance=nuisance, null_check=check, curvature_jitter=1e-8)

    # Four guides per element, so a cell's guide count carries element identity
    # the way it does in a real screen.
    guides_per_element = 4
    data = PerTurboData(
        counts=counts, pert_id=membership.astype(np.float32), pert_names=pert_names,
        gene_names=gene_names,
        covariates=np.column_stack([continuous, np.eye(num_levels)[codes][:, :-1]]),
        covariate_names=["continuous"] + [f"level{index}" for index in range(num_levels - 1)],
        guide_matrix=np.repeat(membership, guides_per_element, axis=1).astype(np.float32),
        guide_names=[f"gd{index}" for index in range(guides_per_element * num_elements)],
        guide_to_element=sp.csr_matrix(
            np.repeat(np.eye(num_elements, dtype=np.float32), guides_per_element, axis=0)
        ),
        size_factors=offsets,
    )
    return baseline, data, codes, confined_level


def _p_values(baseline, data, *, confine: bool):
    result = crt.run_crt_all_cells(
        baseline, data, screen_p_value=1.0, gene_chunk_size=None, gene_block_size=128,
        confine_to_observed_batches=confine,
    )
    return result.parametric[crt.CRT_SADDLEPOINT_FAMILY]["p_value"]


@pytest.fixture(scope="module")
def screen():
    return _screen()


def test_batch_confined_elements_keep_a_uniform_null(screen):
    baseline, data, _, confined_level = screen
    p_value = _p_values(baseline, data, confine=True)
    confined = p_value[confined_level >= 0].ravel()
    confined = confined[np.isfinite(confined)]
    assert confined.size > 100

    # Uniformity: the rate at each threshold, and the whole distribution.
    assert np.mean(confined < 0.05) < 0.12
    assert np.mean(confined < 0.01) < 0.04
    assert np.mean(confined < 0.001) < 0.01
    from scipy import stats

    assert stats.kstest(confined, "uniform").pvalue > 0.001
    # The collapse this guards against produces p-values no null can reach.
    assert confined.min() > 1e-6


def test_confinement_leaves_elements_present_in_every_level_alone(screen):
    baseline, data, _, _ = screen
    fit = crt.prepare_all_cells_propensity(baseline, data)
    full = fit.element_support.all(axis=1)
    assert full.sum() >= 5, "the fixture needs elements that occupy every level"
    with_support = _p_values(baseline, data, confine=True)
    without = _p_values(baseline, data, confine=False)
    # An element present in every level has every cell in its support, so the
    # restricted fit is the same likelihood. It is not bit-identical: a batch of
    # elements whose supports differ leaves the structured Newton solve for the
    # masked one, and the two agree only to float32.
    np.testing.assert_allclose(with_support[full], without[full], rtol=2e-3, atol=1e-6)


def test_support_is_the_elements_own_batch_levels(screen):
    baseline, data, codes, confined_level = screen
    fit = crt.prepare_all_cells_propensity(baseline, data)
    assert fit.element_support is not None and fit.batch_codes is not None
    np.testing.assert_array_equal(fit.batch_codes, codes)
    for element, level in enumerate(confined_level):
        support = fit.element_support[element]
        if level < 0:
            assert support.all()
        else:
            assert support.sum() == 1 and support[level]

    # Every element occupies every level: nothing to restrict, and the fit is
    # the unrestricted one.
    _, spread_only, _, _ = _screen(num_confined=0)
    unrestricted = crt.prepare_all_cells_propensity(baseline, spread_only)
    assert unrestricted.element_support is None and unrestricted.batch_codes is None


def test_a_two_level_batch_is_still_given_a_support():
    """Two lanes are coded by one indicator column, and still separate.

    The support used to be read off the bordered factorization, which needs two
    indicator columns before the split is worth making. A two-level batch has
    one, so the levels went unseen and ``--crt-all-cells-batch-support`` was a
    silent no-op for exactly the design that needs the least work to protect.
    """

    from perturbo._internal.high_moi.resampling import propensity_logits_from_coefficients

    baseline, data, codes, confined_level = _screen(
        num_levels=2, num_spread=4, num_confined=1, num_cells=3000, num_genes=6
    )
    fit = crt.prepare_all_cells_propensity(baseline, data)
    assert fit.element_support is not None and fit.batch_codes is not None
    assert fit.element_support.shape == (5, 2)
    np.testing.assert_array_equal(fit.batch_codes, codes)

    confined = int(np.flatnonzero(confined_level >= 0)[0])
    assert fit.element_support[confined].tolist() == [True, False]
    assert fit.element_support[:confined].all(), "the spread elements need both lanes"

    # The mask reached the fit rather than only the metadata: the masked MLE's
    # own score equation is that the selection probability over the support sums
    # to the element's cell count, which the unmasked fit spreads over both lanes.
    probability = np.asarray(
        1.0 / (1.0 + np.exp(-propensity_logits_from_coefficients(fit.coefficients, fit.basis)))
    )
    inside = codes == confined_level[confined]
    membership = np.asarray(data.pert_id) > 0
    assert probability[confined][inside].sum() == pytest.approx(
        int(membership[:, confined].sum()), rel=1e-3
    )
    # --no-crt-all-cells-batch-support still opts out, two levels or fourteen.
    unrestricted = crt.prepare_all_cells_propensity(
        baseline, data, confine_to_observed_batches=False
    )
    assert unrestricted.element_support is None and unrestricted.batch_codes is None


def test_the_selection_model_is_fit_on_its_own_support(screen):
    """sum(pi) over the support is the element's own cell count, as the MLE requires."""

    from perturbo._internal.high_moi.resampling import propensity_logits_from_coefficients

    baseline, data, codes, confined_level = screen
    fit = crt.prepare_all_cells_propensity(baseline, data)
    probability = np.asarray(
        1.0 / (1.0 + np.exp(-propensity_logits_from_coefficients(fit.coefficients, fit.basis)))
    )
    membership = np.asarray(data.pert_id) > 0
    for element in np.flatnonzero(confined_level >= 0):
        inside = codes == confined_level[element]
        assert probability[element][inside].sum() == pytest.approx(
            int(membership[:, element].sum()), rel=1e-3
        )


def test_support_removes_other_levels_from_the_cumulants_outright():
    """The mask is pi = 0, not a small pi: the kernel is called both ways.

    Contributions outside the support are made large enough that any residual
    probability there would dominate the reported cumulants, so this separates
    "excluded" from "nearly excluded" without depending on float32 noise.
    """

    from perturbo._internal.saddlepoint import fit_high_moi_propensity_saddlepoint

    rng = np.random.default_rng(4)
    num_cells, num_genes, num_levels = 400, 3, 4
    codes = np.arange(num_cells) % num_levels
    inside = codes == 0
    residual = np.where(inside[:, None], rng.normal(size=(num_cells, num_genes)),
                        1000.0 * rng.normal(size=(num_cells, num_genes)))
    weight = np.full((num_cells, num_genes), 1e-12)   # keep contribution == residual
    design = np.column_stack([np.ones(num_cells), np.eye(num_levels)[codes][:, 1:]])
    basis = np.column_stack([np.ones(num_cells) / np.sqrt(num_cells), inside - inside.mean()])
    basis[:, 1] /= np.linalg.norm(basis[:, 1])
    cells = np.flatnonzero(inside)[:20]
    support = np.zeros((1, num_levels), dtype=bool)
    support[0, 0] = True
    shared = dict(
        score_residual=residual, observation_weight=weight, nuisance_design=design,
        nuisance_direction=np.zeros((design.shape[1], num_genes)),
        cell_index=cells, element_index=np.zeros_like(cells), num_elements=1,
        # A deliberately unconverged fit: real probability everywhere.
        propensity_coefficients=np.array([[-1.0, 2.0]]), propensity_basis=basis,
        screen_p_value=1.0, gene_block_size=4,
    )
    restricted = fit_high_moi_propensity_saddlepoint(
        **shared, batch_codes=codes, element_support=support
    )
    unrestricted = fit_high_moi_propensity_saddlepoint(**shared)

    selection = 1.0 / (1.0 + np.exp(-(basis @ np.array([-1.0, 2.0]))))
    np.testing.assert_allclose(
        restricted.null_mean[0], selection[inside] @ residual[inside], rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        restricted.null_variance[0],
        (selection * (1 - selection))[inside] @ np.square(residual[inside]),
        rtol=1e-5, atol=1e-8,
    )
    # The out-of-level cells were carrying a thousand times the signal.
    assert np.all(unrestricted.null_variance[0] > 100.0 * restricted.null_variance[0])
