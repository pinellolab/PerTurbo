from __future__ import annotations

import numpy as np
from scipy import stats

from perturbo._internal.parametric_null import (
    fit_skew_normal_from_moments,
    fit_skew_normal_from_samples,
    fit_student_t_from_samples,
)


def test_skew_normal_moment_fit_recovers_distribution_and_tail() -> None:
    rng = np.random.default_rng(8)
    samples = stats.skewnorm.rvs(4.0, loc=-0.3, scale=1.2, size=200_000, random_state=rng)
    observed = np.asarray([2.5])
    fit = fit_skew_normal_from_samples(observed, samples[None, :])
    expected = stats.skewnorm.cdf(-2.5, 4.0, loc=-0.3, scale=1.2) + stats.skewnorm.sf(2.5, 4.0, loc=-0.3, scale=1.2)

    assert fit.valid[0]
    np.testing.assert_allclose(fit.location, [-0.3], atol=0.03)
    np.testing.assert_allclose(fit.scale, [1.2], atol=0.03)
    np.testing.assert_allclose(fit.p_value, [expected], rtol=0.08)


def _infeasible_skewness_moments() -> dict:
    """Moments whose skewness exceeds what a skew-normal can represent."""
    return {
        "count": np.asarray([100.0]),
        "sum_score": np.asarray([0.0]),
        "sum_square": np.asarray([100.0]),
        "sum_cube": np.asarray([120.0]),
    }


def test_skew_normal_moment_fit_marks_infeasible_skewness_invalid() -> None:
    """``valid`` reports family membership regardless of whether a tail is produced."""
    fit = fit_skew_normal_from_moments(np.asarray([2.0]), **_infeasible_skewness_moments())
    assert not fit.valid[0]

    unclamped = fit_skew_normal_from_moments(
        np.asarray([2.0]), clamp_skewness=False, **_infeasible_skewness_moments()
    )
    assert not unclamped.valid[0]
    assert np.isnan(unclamped.p_value[0])


def test_skew_normal_clamps_infeasible_skewness_to_the_family_boundary() -> None:
    """Past the feasible range, fit the most skewed member instead of reporting nothing.

    A NaN is read downstream as a pair that was never tested rather than one
    the model could not describe, and it is the only failure mode this fit has
    - on the Replogle essential panel every unfitted pair was an out-of-range
    skewness, none from a degenerate variance or a non-finite moment.
    """
    fit = fit_skew_normal_from_moments(np.asarray([2.0]), **_infeasible_skewness_moments())

    assert not fit.valid[0], "still flagged as outside the family"
    assert np.isfinite(fit.p_value[0]), "but a tail is produced anyway"
    assert 0.0 < fit.p_value[0] <= 1.0
    assert np.isfinite(fit.shape[0]), "shape stays finite just inside the boundary"


def test_skew_normal_clamping_is_a_strict_extension() -> None:
    """Clamping must not perturb any fit that already succeeded."""
    rng = np.random.default_rng(4)
    n = 400
    count = np.full(n, 1000.0)
    mean = rng.normal(0.0, 0.05, size=n)
    variance = rng.uniform(0.8, 1.2, size=n)
    # Deliberately straddle the boundary so both branches are exercised.
    skewness = rng.uniform(-1.5, 1.5, size=n)
    sum_score = mean * count
    sum_square = (variance + mean**2) * count
    sum_cube = (skewness * variance**1.5 + 3 * mean * variance + mean**3) * count
    observed = rng.normal(0.0, 2.0, size=n)

    clamped = fit_skew_normal_from_moments(
        observed, count=count, sum_score=sum_score, sum_square=sum_square,
        sum_cube=sum_cube, clamp_skewness=True,
    )
    plain = fit_skew_normal_from_moments(
        observed, count=count, sum_score=sum_score, sum_square=sum_square,
        sum_cube=sum_cube, clamp_skewness=False,
    )

    both = np.isfinite(clamped.p_value) & np.isfinite(plain.p_value)
    assert both.sum() > 0 and (~np.isfinite(plain.p_value)).sum() > 0, "need both branches"
    np.testing.assert_allclose(clamped.p_value[both], plain.p_value[both], rtol=1e-12)
    assert np.isfinite(clamped.p_value).all(), "clamping leaves nothing unfitted"
    np.testing.assert_array_equal(clamped.valid, plain.valid)


def test_skew_normal_tail_matches_absolute_empirical_event() -> None:
    symmetric = np.random.default_rng(21).normal(size=200_000)
    fit = fit_skew_normal_from_samples(np.asarray([-2.0]), symmetric[None, :])
    empirical = np.mean(np.abs(symmetric) >= 2.0)

    assert fit.valid[0]
    np.testing.assert_allclose(fit.p_value, [empirical], atol=0.01)


def test_student_t_moment_fit_recovers_heavy_tail_and_absolute_event() -> None:
    rng = np.random.default_rng(34)
    samples = stats.t.rvs(7.0, loc=0.1, scale=0.8, size=300_000, random_state=rng)
    observed = np.asarray([3.0])
    fit = fit_student_t_from_samples(observed, samples[None, :])
    expected = stats.t.cdf(-3.0, 7.0, loc=0.1, scale=0.8) + stats.t.sf(3.0, 7.0, loc=0.1, scale=0.8)

    assert fit.valid[0]
    np.testing.assert_allclose(fit.degrees_of_freedom, [7.0], rtol=0.12)
    np.testing.assert_allclose(fit.scale, [0.8], rtol=0.04)
    np.testing.assert_allclose(fit.p_value, [expected], rtol=0.12)
