"""Moment-based parametric approximations to resampled score nulls."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats


_MAX_SKEW_NORMAL_SKEWNESS = float(((4.0 - np.pi) / 2.0) * (2.0 / np.pi) ** 1.5 / (1.0 - 2.0 / np.pi) ** 1.5)


@dataclass(frozen=True)
class SkewNormalMomentFit:
    """Per-hypothesis skew-normal fits reconstructed from null moments.

    ``log_p_value`` is the natural log of the *same* tail event as ``p_value``,
    carried alongside rather than instead of it. It is not redundant: a
    linear-scale probability cannot represent anything below ~1e-308 even in
    float64, and the strongest on-target hits sit at ``z`` of 14 and well past
    it, so every consumer that only ever wanted ``-log10(p)`` was paying a
    truncation for the round trip through a linear probability.

    A caveat specific to this family: SciPy's ``skewnorm`` has no native
    log-scale CDF, because a skew-normal CDF needs Owen's T. Its
    ``logcdf``/``logsf`` are the logs of the linear values, so this field
    reaches ``-inf`` wherever the linear tail underflows and buys no extra
    range at all - it is here for interface parity with the other families.
    :mod:`perturbo._internal.saddlepoint` is the family that does keep range,
    because it is evaluated in the exponent rather than logged afterwards.
    """

    p_value: np.ndarray
    log_p_value: np.ndarray
    location: np.ndarray
    scale: np.ndarray
    shape: np.ndarray
    null_mean: np.ndarray
    null_variance: np.ndarray
    null_skewness: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class StudentTMomentFit:
    """Per-hypothesis symmetric Student-t fits reconstructed from null moments.

    ``log_p_value`` is the natural log of the same tail event as ``p_value``;
    see :class:`SkewNormalMomentFit` for why it is carried separately. SciPy's
    ``t.logsf`` is likewise the log of the linear tail, but a Student-t tail
    decays polynomially, so underflow needs a threshold around ``10**(308/nu)``
    and is not a practical concern. The ``nu = inf`` branch evaluates through
    ``norm.logsf``/``logcdf`` and is exact in log space at any distance.
    """

    p_value: np.ndarray
    log_p_value: np.ndarray
    location: np.ndarray
    scale: np.ndarray
    degrees_of_freedom: np.ndarray
    null_mean: np.ndarray
    null_variance: np.ndarray
    null_skewness: np.ndarray
    null_excess_kurtosis: np.ndarray
    valid: np.ndarray


def fit_skew_normal_from_moments(
    observed_score: np.ndarray,
    *,
    count: np.ndarray,
    sum_score: np.ndarray,
    sum_square: np.ndarray,
    sum_cube: np.ndarray,
    clamp_skewness: bool = True,
) -> SkewNormalMomentFit:
    """Fit skew-normal nulls by moments and evaluate absolute two-sided tails.

    The tail event matches the empirical CRT exactly: ``P(|S| >= |S_obs|)``.

    A skew-normal can only represent ``|skewness| < 0.995272`` - the half-normal
    limit as its shape parameter runs to infinity - so a more skewed empirical
    null has no moment-matched member of the family. On the Replogle essential
    panel that is the cause of *every* unfitted pair: 285 of 412,885 with
    resampled moments and 681 with analytic ones, all of them exceeding the
    limit, none failing for any other reason.

    ``clamp_skewness`` (default) fits the most skewed member available instead
    of returning ``NaN``. This is a strict extension, not a change: on the
    412,600 pairs that already fit, clamped and unclamped tails agree to within
    8e-14 relative, because the clamp only binds where no fit existed. The
    alternative of reporting nothing is worse than reporting the boundary of
    the family, since a downstream BH pass reads a ``NaN`` as a pair that was
    never tested rather than one the model could not describe.

    Pass ``clamp_skewness=False`` for the previous behaviour, which marks those
    triples invalid. ``valid`` still records whether the skewness was inside
    the family either way, so a clamped fit remains identifiable.

    The tail is reported twice, on the linear and the log scale. The two agree
    to roundoff wherever the linear one is representable, and only the log one
    is meaningful once it is not.
    """

    observed = np.asarray(observed_score, dtype=np.float64)
    n = np.asarray(count, dtype=np.float64)
    first = np.asarray(sum_score, dtype=np.float64)
    second = np.asarray(sum_square, dtype=np.float64)
    third = np.asarray(sum_cube, dtype=np.float64)
    for value in (n, first, second, third):
        if value.shape != observed.shape:
            raise ValueError("Moment arrays must have the same shape as observed_score.")

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = first / n
        variance = second / n - np.square(mean)
        central_third = third / n - 3.0 * mean * (second / n) + 2.0 * np.power(mean, 3)
        skewness = central_third / np.power(variance, 1.5)

    # Everything the fit needs except a representable skewness. Separated so
    # ``valid`` can keep meaning "inside the family" while the clamp decides
    # only whether an out-of-range skewness still produces a tail.
    usable = (
        (n >= 3)
        & np.isfinite(observed)
        & np.isfinite(mean)
        & np.isfinite(variance)
        & (variance > 0.0)
        & np.isfinite(skewness)
    )
    representable = np.abs(skewness) < _MAX_SKEW_NORMAL_SKEWNESS
    valid = usable & representable
    fitted = usable if clamp_skewness else valid
    location = np.full(observed.shape, np.nan, dtype=np.float64)
    scale = np.full(observed.shape, np.nan, dtype=np.float64)
    shape = np.full(observed.shape, np.nan, dtype=np.float64)
    p_value = np.full(observed.shape, np.nan, dtype=np.float64)
    log_p_value = np.full(observed.shape, np.nan, dtype=np.float64)
    if np.any(fitted):
        # Just inside the boundary: at the limit itself delta is exactly 1 and
        # the shape parameter divides by zero.
        limit = _MAX_SKEW_NORMAL_SKEWNESS * (1.0 - 1e-9)
        gamma = np.clip(skewness[fitted], -limit, limit)
        transformed = np.power(2.0 * np.abs(gamma) / (4.0 - np.pi), 2.0 / 3.0)
        delta = np.sign(gamma) * np.sqrt((np.pi / 2.0) * transformed / (1.0 + transformed))
        fitted_scale = np.sqrt(variance[fitted] / (1.0 - 2.0 * np.square(delta) / np.pi))
        fitted_location = mean[fitted] - fitted_scale * delta * np.sqrt(2.0 / np.pi)
        fitted_shape = delta / np.sqrt(1.0 - np.square(delta))
        threshold = np.abs(observed[fitted])
        tail = stats.skewnorm.cdf(
            -threshold, fitted_shape, loc=fitted_location, scale=fitted_scale
        ) + stats.skewnorm.sf(threshold, fitted_shape, loc=fitted_location, scale=fitted_scale)
        # Taken from the linear tail, not from skewnorm.logcdf/logsf. Those
        # look like they should extend the range the way norm.logsf does, but
        # SciPy computes them as the log of the linear value: measured equal to
        # log(cdf) within 2.2e-16 wherever the linear value exists, and equally
        # non-finite once it underflows. So they buy neither accuracy nor range
        # here - only about 5 seconds per million hypotheses, because a
        # skew-normal CDF goes through Owen's T. This family structurally
        # cannot reach past the linear underflow, and paying for the pretence
        # is worse than documenting the limit; the saddlepoint path is the one
        # that keeps range, being evaluated in the exponent throughout.
        with np.errstate(divide="ignore"):
            log_tail = np.log(tail)
        location[fitted] = fitted_location
        scale[fitted] = fitted_scale
        shape[fitted] = fitted_shape
        p_value[fitted] = np.clip(tail, np.finfo(float).tiny, 1.0)
        # Only the upper half of that clip carries over. ``log(1) = 0`` is a
        # genuine bound on a probability, whereas the lower clip exists purely
        # to keep the linear value off zero - and that floor is the loss this
        # field is here to avoid, so imposing it again would defeat the point.
        log_p_value[fitted] = np.minimum(log_tail, 0.0)

    return SkewNormalMomentFit(
        p_value=p_value,
        log_p_value=log_p_value,
        location=location,
        scale=scale,
        shape=shape,
        null_mean=mean,
        null_variance=variance,
        null_skewness=skewness,
        valid=valid,
    )


def fit_skew_normal_from_samples(
    observed_score: np.ndarray,
    resampled_scores: np.ndarray,
) -> SkewNormalMomentFit:
    """Convenience wrapper for score arrays with resamples on the last axis."""

    samples = np.asarray(resampled_scores, dtype=np.float64)
    observed = np.asarray(observed_score, dtype=np.float64)
    if samples.shape[:-1] != observed.shape:
        raise ValueError("resampled_scores must end in a resample axis after observed_score shape.")
    finite = np.isfinite(samples)
    values = np.where(finite, samples, 0.0)
    return fit_skew_normal_from_moments(
        observed,
        count=finite.sum(axis=-1),
        sum_score=values.sum(axis=-1),
        sum_square=np.square(values).sum(axis=-1),
        sum_cube=np.power(values, 3).sum(axis=-1),
    )


def fit_student_t_from_moments(
    observed_score: np.ndarray,
    *,
    count: np.ndarray,
    sum_score: np.ndarray,
    sum_square: np.ndarray,
    sum_cube: np.ndarray,
    sum_fourth: np.ndarray,
) -> StudentTMomentFit:
    """Fit centered-location Student-t nulls by variance and kurtosis matching.

    A Student-t has excess kurtosis ``6 / (nu - 4)`` for ``nu > 4``. Negative
    empirical excess is represented by the Normal limiting case ``nu = inf``;
    that is preferable to forcing an artificial heavy tail. The reported tail is
    the CRT event ``P(|S| >= |S_obs|)`` rather than a tail around the fitted
    location.

    The tail is reported twice, on the linear and the log scale; the two agree
    to roundoff wherever the linear one is representable.
    """

    observed = np.asarray(observed_score, dtype=np.float64)
    n = np.asarray(count, dtype=np.float64)
    first = np.asarray(sum_score, dtype=np.float64)
    second = np.asarray(sum_square, dtype=np.float64)
    third = np.asarray(sum_cube, dtype=np.float64)
    fourth = np.asarray(sum_fourth, dtype=np.float64)
    for value in (n, first, second, third, fourth):
        if value.shape != observed.shape:
            raise ValueError("Moment arrays must have the same shape as observed_score.")

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = first / n
        raw_second = second / n
        raw_third = third / n
        raw_fourth = fourth / n
        variance = raw_second - np.square(mean)
        central_third = raw_third - 3.0 * mean * raw_second + 2.0 * np.power(mean, 3)
        central_fourth = (
            raw_fourth - 4.0 * mean * raw_third + 6.0 * np.square(mean) * raw_second - 3.0 * np.power(mean, 4)
        )
        skewness = central_third / np.power(variance, 1.5)
        excess_kurtosis = central_fourth / np.square(variance) - 3.0

    valid = (
        (n >= 4)
        & np.isfinite(observed)
        & np.isfinite(mean)
        & np.isfinite(variance)
        & (variance > 0.0)
        & np.isfinite(excess_kurtosis)
    )
    degrees_of_freedom = np.full(observed.shape, np.nan, dtype=np.float64)
    scale = np.full(observed.shape, np.nan, dtype=np.float64)
    p_value = np.full(observed.shape, np.nan, dtype=np.float64)
    log_p_value = np.full(observed.shape, np.nan, dtype=np.float64)
    if np.any(valid):
        fitted_excess = excess_kurtosis[valid]
        finite_df = fitted_excess > 0.0
        df = np.full(fitted_excess.shape, np.inf, dtype=np.float64)
        df[finite_df] = 4.0 + 6.0 / fitted_excess[finite_df]
        fitted_scale = np.sqrt(variance[valid])
        fitted_scale[finite_df] *= np.sqrt((df[finite_df] - 2.0) / df[finite_df])
        threshold = np.abs(observed[valid])
        tail = np.empty_like(threshold)
        # Both tails of the two-sided event are summed with logaddexp rather
        # than logged after the fact, so the sum itself never visits the linear
        # scale and cannot underflow there.
        log_tail = np.empty_like(threshold)
        if np.any(finite_df):
            tail[finite_df] = stats.t.cdf(
                -threshold[finite_df],
                df[finite_df],
                loc=mean[valid][finite_df],
                scale=fitted_scale[finite_df],
            ) + stats.t.sf(
                threshold[finite_df],
                df[finite_df],
                loc=mean[valid][finite_df],
                scale=fitted_scale[finite_df],
            )
            log_tail[finite_df] = np.logaddexp(
                stats.t.logcdf(
                    -threshold[finite_df],
                    df[finite_df],
                    loc=mean[valid][finite_df],
                    scale=fitted_scale[finite_df],
                ),
                stats.t.logsf(
                    threshold[finite_df],
                    df[finite_df],
                    loc=mean[valid][finite_df],
                    scale=fitted_scale[finite_df],
                ),
            )
        if np.any(~finite_df):
            tail[~finite_df] = stats.norm.cdf(
                -threshold[~finite_df],
                loc=mean[valid][~finite_df],
                scale=fitted_scale[~finite_df],
            ) + stats.norm.sf(
                threshold[~finite_df],
                loc=mean[valid][~finite_df],
                scale=fitted_scale[~finite_df],
            )
            log_tail[~finite_df] = np.logaddexp(
                stats.norm.logcdf(
                    -threshold[~finite_df],
                    loc=mean[valid][~finite_df],
                    scale=fitted_scale[~finite_df],
                ),
                stats.norm.logsf(
                    threshold[~finite_df],
                    loc=mean[valid][~finite_df],
                    scale=fitted_scale[~finite_df],
                ),
            )
        degrees_of_freedom[valid] = df
        scale[valid] = fitted_scale
        p_value[valid] = np.clip(tail, np.finfo(float).tiny, 1.0)
        # No lower clip on the log scale: see fit_skew_normal_from_moments.
        log_p_value[valid] = np.minimum(log_tail, 0.0)

    return StudentTMomentFit(
        p_value=p_value,
        log_p_value=log_p_value,
        location=mean,
        scale=scale,
        degrees_of_freedom=degrees_of_freedom,
        null_mean=mean,
        null_variance=variance,
        null_skewness=skewness,
        null_excess_kurtosis=excess_kurtosis,
        valid=valid,
    )


def fit_student_t_from_samples(observed_score: np.ndarray, resampled_scores: np.ndarray) -> StudentTMomentFit:
    """Convenience wrapper for score arrays with resamples on the last axis."""

    samples = np.asarray(resampled_scores, dtype=np.float64)
    observed = np.asarray(observed_score, dtype=np.float64)
    if samples.shape[:-1] != observed.shape:
        raise ValueError("resampled_scores must end in a resample axis after observed_score shape.")
    finite = np.isfinite(samples)
    values = np.where(finite, samples, 0.0)
    return fit_student_t_from_moments(
        observed,
        count=finite.sum(axis=-1),
        sum_score=values.sum(axis=-1),
        sum_square=np.square(values).sum(axis=-1),
        sum_cube=np.power(values, 3).sum(axis=-1),
        sum_fourth=np.power(values, 4).sum(axis=-1),
    )


__all__ = [
    "SkewNormalMomentFit",
    "StudentTMomentFit",
    "fit_skew_normal_from_moments",
    "fit_skew_normal_from_samples",
    "fit_student_t_from_moments",
    "fit_student_t_from_samples",
]
