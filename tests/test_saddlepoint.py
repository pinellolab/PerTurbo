from __future__ import annotations

from itertools import combinations, product

import numpy as np

from perturbo._internal.saddlepoint import (
    fit_high_moi_stratified_saddlepoint,
    fit_stratified_saddlepoint_from_components,
    stratified_saddlepoint_log_two_sided,
)


def test_stratified_spa_adds_independent_stratum_cumulants() -> None:
    pools = (
        np.asarray([[-2.0, -1.0], [-0.5, 0.5], [0.5, -0.5], [2.0, 1.0]]),
        np.asarray([[-1.5, -0.25], [-0.5, 0.25], [0.5, -0.25], [1.5, 0.25]]),
    )
    selected = np.asarray([1, 2])
    observed = np.asarray([1.0, 0.5])

    log_p, valid, variance, skewness = stratified_saddlepoint_log_two_sided(
        observed, pools, selected, finite_population=False
    )

    expected_variance = sum(
        count * np.mean(np.square(pool), axis=0)
        for pool, count in zip(pools, selected)
    )
    expected_third = sum(
        count * np.mean(np.power(pool, 3), axis=0)
        for pool, count in zip(pools, selected)
    )
    np.testing.assert_allclose(variance, expected_variance, rtol=1e-12)
    np.testing.assert_allclose(skewness, expected_third / expected_variance**1.5, atol=1e-12)
    assert np.all(valid)
    assert np.all(np.isfinite(log_p))
    assert np.all(log_p <= 0.0)


def test_power_cgf_correction_matches_srswor_variance() -> None:
    pool = np.arange(-10.0, 11.0)[:, None]
    selected = np.asarray([5])
    _, _, variance, _ = stratified_saddlepoint_log_two_sided(
        np.asarray([4.0]), [pool], selected, finite_population=True
    )

    population = pool.shape[0]
    expected = selected[0] * np.var(pool[:, 0]) * (population - selected[0]) / (population - 1)
    np.testing.assert_allclose(variance[0], expected, rtol=1e-12)


def test_two_sided_spa_is_invariant_to_flipping_all_contributions() -> None:
    pool = np.asarray([[-2.0], [-1.0], [-0.25], [0.5], [1.0], [1.75]])
    pool -= pool.mean(axis=0)
    first = stratified_saddlepoint_log_two_sided(
        np.asarray([1.25]), [pool], np.asarray([2]), finite_population=False
    )[0]
    second = stratified_saddlepoint_log_two_sided(
        np.asarray([-1.25]), [-pool], np.asarray([2]), finite_population=False
    )[0]
    np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-12)


def test_stratified_spa_agrees_with_exhaustive_srswor_in_distribution_bulk() -> None:
    """Small-pool enumeration checks the approximation against the actual CRT law.

    This deliberately probes the body rather than the most extreme attainable
    subset. The latter is where replacement can matter combinatorially even at
    a modest sampling fraction, and is the reason this prototype reports that
    fraction rather than presenting itself as the conditional SPA.
    """
    rng = np.random.default_rng(4)
    first = rng.normal(size=24)
    second = rng.normal(size=30)
    first -= first.mean()
    second -= second.mean()
    pools = [first[:, None], second[:, None]]
    selected = np.asarray([2, 3])
    observed = first[[0, 1]].sum() + second[[0, 1, 2]].sum()

    first_sums = np.asarray([first[list(index)].sum() for index in combinations(range(24), 2)])
    second_sums = np.asarray([second[list(index)].sum() for index in combinations(range(30), 3)])
    exact = np.mean(np.abs(first_sums[:, None] + second_sums[None, :]) >= abs(observed))
    log_p = stratified_saddlepoint_log_two_sided(
        np.asarray([observed]), pools, selected, finite_population=True
    )[0]

    np.testing.assert_allclose(np.exp(log_p[0]), exact, rtol=0.02)


def test_component_builder_uses_efficient_contributions_and_reports_fraction() -> None:
    residual = np.asarray(
        [
            [-1.2, 0.2],
            [0.4, -0.1],
            [0.8, -0.1],
            [-0.7, 0.3],
            [0.2, -0.2],
            [0.5, -0.1],
        ]
    )
    weight = np.asarray(
        [
            [0.8, 1.0],
            [1.2, 0.7],
            [0.9, 1.1],
            [1.1, 0.8],
            [0.7, 1.2],
            [1.3, 0.9],
        ]
    )
    strata = np.asarray([0, 0, 0, 1, 1, 1])
    control = np.asarray([True, True, False, True, True, False])
    target_cells = {0: np.asarray([2, 5])}

    fit = fit_stratified_saddlepoint_from_components(
        score_residual=residual,
        observation_weight=weight,
        strata=strata,
        control_mask=control,
        target_cells=target_cells,
        num_targets=1,
        finite_population=False,
    )

    expected = np.zeros(2)
    for stratum, target in product((0, 1), (0,)):
        pool_rows = np.flatnonzero((strata == stratum) & (control | np.isin(np.arange(6), target_cells[target])))
        selected_rows = target_cells[target][strata[target_cells[target]] == stratum]
        information = weight[pool_rows].sum(axis=0)
        nuisance_score = residual[pool_rows].sum(axis=0)
        contribution = residual[pool_rows] - weight[pool_rows] * (nuisance_score / information)
        np.testing.assert_allclose(contribution.sum(axis=0), 0.0, atol=1e-12)
        lookup = {row: index for index, row in enumerate(pool_rows)}
        expected += contribution[[lookup[row] for row in selected_rows]].sum(axis=0)

    np.testing.assert_allclose(fit.observed_sum[0], expected, atol=1e-12)
    np.testing.assert_allclose(fit.max_sampling_fraction[0], np.asarray([1 / 3, 1 / 3]))
    assert np.all(fit.valid)


def test_high_moi_builder_matches_arbitrary_nuisance_efficient_numerator() -> None:
    rng = np.random.default_rng(12)
    num_cells, num_genes, num_nuisance, num_elements = 30, 3, 2, 4
    residual = rng.normal(size=(num_cells, num_genes))
    weight = np.exp(rng.normal(scale=0.2, size=(num_cells, num_genes)))
    nuisance = np.column_stack([np.ones(num_cells), rng.normal(size=num_cells)])
    inverse = np.stack(
        [np.linalg.inv(nuisance.T @ (weight[:, gene, None] * nuisance) + np.eye(num_nuisance))
         for gene in range(num_genes)]
    )
    nuisance_score = nuisance.T @ residual
    cell_index = np.concatenate([rng.choice(num_cells, 5, replace=False) for _ in range(num_elements)])
    element_index = np.repeat(np.arange(num_elements), 5)
    strata = np.repeat(np.arange(3), 10)

    fit = fit_high_moi_stratified_saddlepoint(
        score_residual=residual,
        observation_weight=weight,
        nuisance_design=nuisance,
        nuisance_information_inverse=inverse,
        nuisance_score=nuisance_score,
        cell_index=cell_index,
        element_index=element_index,
        num_elements=num_elements,
        strata=strata,
        screen_p_value=1.0,
        gene_block_size=4,
        finite_population=False,
    )

    direction = np.einsum("gqr,rg->gq", inverse, nuisance_score)
    contribution = residual - np.einsum("ng,nq,gq->ng", weight, nuisance, direction)
    expected = np.zeros((num_elements, num_genes))
    np.add.at(expected, element_index, contribution[cell_index])
    np.testing.assert_allclose(fit.observed_sum, expected, rtol=1e-12, atol=1e-12)
    assert np.all(fit.valid)
    assert not np.any(fit.used_fallback)


def test_spa_keeps_range_and_stays_conservative_where_the_skew_normal_does_not() -> None:
    """The two reasons to prefer the SPA over a moment-matched skew-normal.

    The skew-normal's linear tail underflows past roughly z = 38 and reports
    ``-inf`` thereafter. Less obviously, it is already wrong long before that:
    a three-moment fit extrapolated deep into the tail overstates significance
    by tens of orders of magnitude, because those moments carry no information
    out there. The SPA controls *relative* error uniformly into the tail
    instead, and is evaluated in the exponent so it never underflows.
    """
    from scipy import stats

    from perturbo._internal.parametric_null import fit_skew_normal_from_moments
    from perturbo._internal.saddlepoint import saddlepoint_log_two_sided

    rng = np.random.default_rng(0)
    pool = rng.gamma(0.4, 2.0, size=4000)
    pool -= pool.mean()
    count = 200
    mean = count * pool.mean()
    variance = count * pool.var()
    sd = np.sqrt(variance)
    skewness = stats.skew(pool) / np.sqrt(count)

    scale = np.asarray([1e6])
    moments = {
        "count": scale,
        "sum_score": np.asarray([mean]) * scale,
        "sum_square": np.asarray([variance + mean**2]) * scale,
        "sum_cube": np.asarray([skewness * variance**1.5 + 3 * mean * variance + mean**3]) * scale,
    }

    def tails(z: float) -> tuple[float, float]:
        observed = mean + z * sd
        spa = float(saddlepoint_log_two_sided(observed, pool, float(count)))
        fit = fit_skew_normal_from_moments(np.asarray([observed]), **moments)
        return spa, float(fit.log_p_value[0])

    # Well inside the tail the two still agree to a few percent.
    near_spa, near_skew = tails(5.0)
    assert abs(near_spa - near_skew) / abs(near_spa) < 0.05

    # Far out the skew-normal claims far more significance than is there.
    far_spa, far_skew = tails(30.0)
    assert far_skew < far_spa - 100.0, "skew-normal should be wildly anti-conservative by z=30"

    # And past its underflow it reports nothing at all, while the SPA keeps going.
    beyond_spa, beyond_skew = tails(60.0)
    assert not np.isfinite(beyond_skew)
    assert np.isfinite(beyond_spa) and beyond_spa < -400.0
    # Monotone in z, as any tail must be.
    assert beyond_spa < far_spa < near_spa < 0.0
