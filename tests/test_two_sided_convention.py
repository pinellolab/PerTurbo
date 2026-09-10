"""The two conventions for the saddlepoint's two-sided p-value on a skewed Bernoulli-sum null."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

jax.config.update("jax_enable_x64", True)

from perturbo._internal.saddlepoint import pearson3_log_two_sided, propensity_saddlepoint_log_two_sided  # noqa: E402


def _skewed_pool(seed: int = 0, pool: int = 400):
    """Right-skewed contributions (count-residual-like) with a common selection probability."""
    rng = np.random.default_rng(seed)
    c = rng.gamma(shape=0.7, scale=1.0, size=pool) - 0.7  # mean zero, skewed right
    p = 0.15
    logits = np.full(pool, np.log(p / (1 - p)))
    return c, logits, p


def _monte_carlo_null(c, p, draws=400_000, seed=1):
    rng = np.random.default_rng(seed)
    selected = rng.random((draws, c.size)) < p
    return selected @ c


def test_both_conventions_match_monte_carlo_tails():
    c, logits, p = _skewed_pool()
    null = _monte_carlo_null(c, p)
    observed = np.array([-3.0, -1.5, 1.5, 3.0, 4.5])
    sym, sym_valid = propensity_saddlepoint_log_two_sided(
        jnp.asarray(observed), jnp.asarray(c)[:, None], jnp.asarray(logits), two_sided="symmetric"
    )
    eq, eq_valid = propensity_saddlepoint_log_two_sided(
        jnp.asarray(observed), jnp.asarray(c)[:, None], jnp.asarray(logits), two_sided="equal-tail"
    )
    assert bool(sym_valid.all()) and bool(eq_valid.all())
    mc_sym = np.array([np.mean(np.abs(null) >= abs(o)) for o in observed])
    mc_eq = np.array([2.0 * min(np.mean(null >= o), np.mean(null <= o)) for o in observed])
    np.testing.assert_allclose(np.exp(np.asarray(sym)), mc_sym, rtol=0.12, atol=2e-4)
    np.testing.assert_allclose(np.exp(np.asarray(eq)), np.minimum(mc_eq, 1.0), rtol=0.12, atol=2e-4)
    # Right-skewed null: the equal-tail p-value is larger than the symmetric one
    # on the long (right) side and smaller on the short (left) side.
    assert np.all(np.asarray(eq)[observed > 0] > np.asarray(sym)[observed > 0])
    assert np.all(np.asarray(eq)[observed < 0] < np.asarray(sym)[observed < 0])


def test_equal_tail_rejects_each_tail_equally_under_the_null():
    c, logits, p = _skewed_pool(seed=2)
    null = _monte_carlo_null(c, p, draws=60_000, seed=3)
    contribution = jnp.asarray(c)[:, None]
    take = null[:20_000]
    sym, _ = propensity_saddlepoint_log_two_sided(jnp.asarray(take), contribution, jnp.asarray(logits), two_sided="symmetric")
    eq, _ = propensity_saddlepoint_log_two_sided(jnp.asarray(take), contribution, jnp.asarray(logits), two_sided="equal-tail")
    p_sym, p_eq = np.exp(np.asarray(sym)), np.exp(np.asarray(eq))
    mean = float(np.mean(null))
    right_sym, left_sym = np.mean((p_sym < 0.05) & (take > mean)), np.mean((p_sym < 0.05) & (take < mean))
    right_eq, left_eq = np.mean((p_eq < 0.05) & (take > mean)), np.mean((p_eq < 0.05) & (take < mean))
    # Symmetric: the long right tail is over-rejected, the short left tail under-rejected.
    assert right_sym > 1.3 * left_sym
    # Equal-tail: both tails at 0.025 within Monte Carlo error (binomial sd about 0.0011).
    assert abs(right_eq - 0.025) < 0.005 and abs(left_eq - 0.025) < 0.005
    assert abs(np.mean(p_eq < 0.05) - 0.05) < 0.006


def test_pearson_screen_conventions_match_the_shifted_gamma():
    mean, variance, skew = 0.0, 4.0, 1.2
    sd = np.sqrt(variance)
    k, theta, shift = 4.0 / skew**2, sd * skew / 2.0, mean - 2.0 * sd / skew
    dist = stats.gamma(a=k, scale=theta, loc=shift)
    observed = np.array([-3.0, -1.0, 1.0, 3.0, 6.0])
    sym = np.exp(np.asarray(pearson3_log_two_sided(jnp.asarray(observed), jnp.asarray(mean), jnp.asarray(variance), jnp.asarray(skew), two_sided="symmetric")))
    eq = np.exp(np.asarray(pearson3_log_two_sided(jnp.asarray(observed), jnp.asarray(mean), jnp.asarray(variance), jnp.asarray(skew), two_sided="equal-tail")))
    np.testing.assert_allclose(sym, dist.sf(np.abs(observed)) + dist.cdf(-np.abs(observed)), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(eq, np.minimum(2.0 * np.minimum(dist.sf(observed), dist.cdf(observed)), 1.0), rtol=1e-6, atol=1e-12)
    # Reflection: a negative skew mirrors the answer.
    eq_neg = np.exp(np.asarray(pearson3_log_two_sided(jnp.asarray(-observed), jnp.asarray(mean), jnp.asarray(variance), jnp.asarray(-skew), two_sided="equal-tail")))
    np.testing.assert_allclose(eq_neg, eq, rtol=1e-6, atol=1e-12)


def test_unknown_convention_is_refused():
    with pytest.raises(ValueError):
        propensity_saddlepoint_log_two_sided(jnp.zeros(1), jnp.zeros((3, 1)), jnp.zeros(3), two_sided="both")
