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


def _frozen_policy_case():
    rng = np.random.default_rng(20260917)
    values = rng.lognormal(0, 3, (32, 256)) * rng.choice([-1, 1], (32, 256))
    logits = rng.uniform(-10, 0, (32, 256))
    observed = (values * (rng.random((32, 256)) < .15)).sum(0)
    return jnp.asarray(observed), jnp.asarray(values), jnp.asarray(logits)


def test_frozen_xaira_policy_regressions_after_root_polish(monkeypatch):
    # Golden outputs cross-checked against the frozen successful Xaira wrapper
    # (full11194_newton_finite_bound_v4), finite-bound policy, at that wrapper's
    # 1e-6 root-residual tolerance. Seeded mixed Bernoulli contributions
    # exercise an LR failure and a formerly incomplete root solve. The clipped
    # Newton polish now resolves the latter even at the old strict tolerance.
    monkeypatch.setattr(sp, "_PROPENSITY_ROOT_RESIDUAL_TOLERANCE", 1e-6)
    log_p, valid, diag = sp.propensity_saddlepoint_log_two_sided_diagnostics.__wrapped__(
        *_frozen_policy_case()
    )
    assert np.asarray(valid).all()
    assert int(diag.failure_reason_code[25]) == 32
    assert float(log_p[25]) == 0.
    assert bool(diag.fallback_used[25]) and bool(diag.chernoff_usable[25])
    assert int(diag.failure_reason_code[67]) == 0
    np.testing.assert_allclose(log_p[67], -2.7327796212938247, atol=1e-10)
    assert not bool(diag.fallback_used[67])


def test_default_tolerance_keeps_polished_roots_on_the_frozen_case():
    # Column 67 now finishes at the root rather than relying on the tolerance
    # to accept a nearly converged iterate. The LR-ratio failure on column 25
    # is unchanged.
    log_p, valid, diag = sp.propensity_saddlepoint_log_two_sided_diagnostics(
        *_frozen_policy_case()
    )
    assert np.asarray(valid).all()
    assert int(diag.failure_reason_code[25]) == 32 and float(log_p[25]) == 0.
    assert int(diag.failure_reason_code[67]) == 0
    assert not bool(diag.fallback_used[67])
    assert float(diag.root_residual_null_sd[67]) <= 1e-12
    np.testing.assert_allclose(log_p[67], -2.7327796212938247, atol=1e-10)
    assert float(log_p[67]) < -0.4922154895294969


def test_nonfinite_observation_is_not_made_valid_by_fallback():
    log_p, valid, _ = sp.propensity_saddlepoint_log_two_sided_diagnostics(
        jnp.array([jnp.nan, jnp.inf]), jnp.ones((4, 2)), jnp.zeros(4)
    )
    assert not np.asarray(valid).any()
    assert np.isnan(np.asarray(log_p)).all()


def test_a_nearly_converged_root_keeps_the_saddlepoint_value(monkeypatch):
    """A residual of ~6e-5 null sd is not a failure: log p moves about 0.27
    nats per null sd of residual here, so the value is within 2e-5 nats. At
    the earlier 1e-6 tolerance this pair went to the Chernoff bound instead."""
    observed = jnp.array([35.])
    contribution = jnp.ones((100, 1))
    logits = jnp.full((100,), np.log(.2 / .8))
    exact_log_p, _, exact_diag = sp.propensity_saddlepoint_log_two_sided_diagnostics(
        observed, contribution, logits
    )
    assert int(exact_diag.failure_reason_code[0]) == 0
    true_solver = sp._solve_propensity_saddlepoint

    def nearly_converged(target, values, logits, *, iterations):
        t_hat, bracketed = true_solver(target, values, logits, iterations=iterations)
        # Binomial(100, .2): a shift of 1e-5 in t moves K'(t) by about 5.7e-5
        # null sd (measured), squarely between the old and new tolerances.
        return t_hat + 1e-5, bracketed

    monkeypatch.setattr(sp, '_solve_propensity_saddlepoint', nearly_converged)
    log_p, valid, diag = sp.propensity_saddlepoint_log_two_sided_diagnostics.__wrapped__(
        observed, contribution, logits
    )
    residual = float(diag.root_residual_null_sd[0])
    assert 1e-6 < residual < 1e-3, residual
    assert bool(valid[0])
    assert int(diag.failure_reason_code[0]) == 0
    assert not bool(diag.fallback_used[0])
    np.testing.assert_allclose(float(log_p[0]), float(exact_log_p[0]), atol=1e-4)
