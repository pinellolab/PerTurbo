"""Regression cases for the Xaira equal-tail LR/Chernoff policy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binom

from perturbo._internal import saddlepoint as sp


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64():
        yield


def test_linear_underflow_is_not_a_tail_failure():
    log_p, valid, diag = sp.propensity_saddlepoint_log_two_sided_diagnostics(
        jnp.array([1000.]), jnp.ones((2000, 1)), jnp.full((2000,), np.log(.01 / .99))
    )
    assert bool(valid[0])
    assert float(log_p[0]) < -3000
    assert np.exp(float(log_p[0])) == 0
    assert int(diag.failure_reason_code[0]) == 0
    assert not bool(diag.fallback_used[0])


def test_root_residual_only_failure_uses_a_nonoptimal_chernoff_bound(monkeypatch):
    # For Binomial(100, .2), t=1 is a finite positive tilt, but does not solve
    # K'(t)=35. It still bounds the upper tail by exp(K(t)-35t).
    def unfinished_root(target, contribution, logits, *, iterations):
        return jnp.ones_like(target), jnp.ones_like(target, dtype=bool)

    monkeypatch.setattr(sp, '_solve_propensity_saddlepoint', unfinished_root)
    log_p, valid, diag = sp.propensity_saddlepoint_log_two_sided_diagnostics.__wrapped__(
        jnp.array([35.]), jnp.ones((100, 1)), jnp.full((100,), np.log(.2 / .8))
    )
    expected = min(0., np.log(2) + 100 * np.log(.8 + .2 * np.e) - 35)
    np.testing.assert_allclose(log_p, [expected], atol=1e-12)
    assert bool(valid[0])
    assert int(diag.failure_reason_code[0]) == 16
    assert bool(diag.fallback_used[0]) and bool(diag.chernoff_usable[0])
    assert not bool(diag.fallback_conservative_one[0])
    assert float(log_p[0]) >= np.log(2 * binom.sf(34, 100, .2))


def test_multiple_failures_do_not_relax_the_bound_guards(monkeypatch):
    def wrong_sign_root(target, contribution, logits, *, iterations):
        return -jnp.ones_like(target), jnp.ones_like(target, dtype=bool)

    monkeypatch.setattr(sp, '_solve_propensity_saddlepoint', wrong_sign_root)
    log_p, valid, diag = sp.propensity_saddlepoint_log_two_sided_diagnostics.__wrapped__(
        jnp.array([35.]), jnp.ones((100, 1)), jnp.full((100,), np.log(.2 / .8))
    )
    assert bool(valid[0]) and float(log_p[0]) == 0.
    assert int(diag.failure_reason_code[0]) & 256
    assert bool(diag.fallback_conservative_one[0])
    assert not bool(diag.chernoff_usable[0])


def test_exact_support_and_near_mean_are_preserved():
    observed = jnp.array([4., 5., 0., 2.])
    log_p, valid, diag = sp.propensity_saddlepoint_log_two_sided_diagnostics(
        observed, jnp.ones((4, 4)), jnp.zeros(4)
    )
    np.testing.assert_allclose(log_p, [np.log(1 / 8), -np.inf, np.log(1 / 8), 0.])
    assert np.asarray(valid).all()
    assert not np.asarray(diag.fallback_used).any()
    assert not np.asarray(diag.failure_reason_code).any()


def test_frozen_xaira_policy_regressions():
    # Golden outputs cross-checked against the frozen successful Xaira wrapper
    # (full11194_newton_finite_bound_v4), finite-bound policy. Seeded mixed
    # Bernoulli contributions exercise LR failure and root-residual failure.
    rng = np.random.default_rng(20260917)
    values = rng.lognormal(0, 3, (32, 256)) * rng.choice([-1, 1], (32, 256))
    logits = rng.uniform(-10, 0, (32, 256))
    observed = (values * (rng.random((32, 256)) < .15)).sum(0)
    log_p, valid, diag = sp.propensity_saddlepoint_log_two_sided_diagnostics(
        jnp.asarray(observed), jnp.asarray(values), jnp.asarray(logits)
    )
    assert np.asarray(valid).all()
    assert int(diag.failure_reason_code[25]) == 32
    assert float(log_p[25]) == 0.
    assert bool(diag.fallback_used[25]) and bool(diag.chernoff_usable[25])
    assert int(diag.failure_reason_code[67]) == 16
    np.testing.assert_allclose(log_p[67], -0.4922154895294969, atol=1e-8)
    assert bool(diag.fallback_used[67]) and bool(diag.chernoff_usable[67])


def test_nonfinite_observation_is_not_made_valid_by_fallback():
    log_p, valid, _ = sp.propensity_saddlepoint_log_two_sided_diagnostics(
        jnp.array([jnp.nan, jnp.inf]), jnp.ones((4, 2)), jnp.zeros(4)
    )
    assert not np.asarray(valid).any()
    assert np.isnan(np.asarray(log_p)).all()
