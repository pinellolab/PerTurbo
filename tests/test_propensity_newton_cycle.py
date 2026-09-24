"""The propensity root solve must not settle into a Newton two-cycle.

Bracketing alone does not make safeguarded Newton converge. When ``K'`` is flat
over most of the bracket and then rises like an exponential - a Bernoulli pool
with a few dominant contributions - the Newton step overshoots far past the root
from the flat side and undershoots far short of it from the steep side, and each
iterate lands strictly inside the bracket the other one just set. The bracket
ends collapse onto the two cycle points, bisection is never reached, and the
loop returns a tilt whose ``K'`` is nowhere near the target.

This was not hypothetical. On the Nadig 2025 Jurkat screen (control-anchored
pool, equal-tail propensity saddlepoint) the pairs flagged
``crt_tail_failure_reason == 144`` were all of this kind. Two of them, captured
from the guarded tail with their real contribution and logit vectors:

  pool 12529 cells, observed 68.533, null mean 0, null sd 22.569 (z = 3.04):
    the cycle sat at t = 0.00263 and t = 0.13216, the root is at t = 0.06029
  pool 13256 cells, observed 16.953, null mean 0, null sd 2.027 (z = 8.36):
    the cycle sat at t = 0.00233 and t = 4.08078, the root is at t = 0.46709

Both returned root residuals of 2.98 and 8.36 null sd (bit 16), a raw
Lugannani-Rice tail above one (bit 128), and a Chernoff bound evaluated at the
cycle point rather than at the root - exp(-0.178), i.e. a reported p of exactly
1 for pairs whose independent root-based LR approximations are 0.0139 and
0.00037.

The fix is the ``rtsafe`` sufficient-progress test - take the Newton step only
when its magnitude is at most half the preceding step size - plus a
final bracket-clipped Newton polish, so that a solve ending on a bisection is
not left only linearly accurate. Both real pairs then converge to a residual
below 1e-15 null sd and report p = 0.0139 and p = 3.7e-4. The synthetic pool
below reproduces the same stall on 256 cells.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from perturbo._internal import saddlepoint as sp


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64():
        yield


def _cycling_pool():
    """A 256-cell pool whose ``K'`` traps the unsafeguarded Newton iteration.

    Four dominant contributions falling off geometrically over a floor of small
    negative ones, all selected with the same probability. This is the shape the
    efficient score contributions take for a low-MOI target whose pool holds a
    handful of high-count cells.
    """

    contribution = np.full(256, -0.5)
    contribution[:4] = 100.0 * 0.45 ** np.arange(4)
    logit = -5.0
    probability = 1.0 / (1.0 + np.exp(-logit))
    mean = float(contribution.sum() * probability)
    sd = float(np.sqrt((contribution**2).sum() * probability * (1.0 - probability)))
    observed = mean + 8.25 * sd
    return (
        jnp.asarray(contribution[:, None]),
        jnp.full((contribution.size, 1), logit, dtype=jnp.float64),
        jnp.asarray([observed]),
        mean,
        sd,
    )


def _bracket_only_newton(target, contribution, logits, *, iterations=30):
    """The pre-fix refine loop: inside the bracket is the only requirement."""

    zero = jnp.zeros_like(target)
    mean, variance = sp._propensity_cgf_derivatives(zero, contribution, logits)
    step = (target - mean) / jnp.maximum(variance, jnp.finfo(jnp.float64).tiny)
    above = target >= mean
    start = jnp.where(
        jnp.isfinite(step) & (step != 0.0), 2.0 * step, jnp.where(above, 1.0, -1.0)
    )

    def widen(_, bound):
        first, _ = sp._propensity_cgf_derivatives(bound, contribution, logits)
        short = jnp.where(above, first < target, first > target)
        return jnp.where(short, bound * 2.0, bound)

    bound = jax.lax.fori_loop(0, 20, widen, start)
    at_bound, _ = sp._propensity_cgf_derivatives(bound, contribution, logits)
    bracketed = jnp.where(above, at_bound >= target, at_bound <= target)
    low = jnp.minimum(bound, zero)
    high = jnp.maximum(bound, zero)
    trajectory = []
    t = zero
    for _ in range(iterations):
        first, second = sp._propensity_cgf_derivatives(t, contribution, logits)
        low = jnp.where(first < target, t, low)
        high = jnp.where(first < target, high, t)
        newton = t + (target - first) / jnp.maximum(second, jnp.finfo(jnp.float64).tiny)
        inside = (newton > low) & (newton < high) & jnp.isfinite(newton)
        t = jnp.where(inside, newton, 0.5 * (low + high))
        trajectory.append(float(t[0]))
    return t, bracketed, trajectory


def _bisected_root(target, contribution, logits):
    low, high = 0.0, 1e3
    for _ in range(200):
        middle = 0.5 * (low + high)
        first, _ = sp._propensity_cgf_derivatives(
            jnp.asarray([middle]), contribution, logits
        )
        if float(first[0]) < float(target[0]):
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def test_bracket_only_newton_settles_into_a_two_cycle():
    """Document the defect the safeguard exists to prevent."""

    contribution, logits, observed, _mean, sd = _cycling_pool()
    t_hat, bracketed, trajectory = _bracket_only_newton(observed, contribution, logits)
    assert bool(bracketed[0]), "the root is bracketed; this is not a support failure"
    first, _ = sp._propensity_cgf_derivatives(t_hat, contribution, logits)
    residual = abs(float(first[0]) - float(observed[0])) / sd
    assert residual > 1.0, residual
    # The last ten iterates alternate between two attracting points instead of
    # converging on one, which is why bisection is never reached: each leg of
    # the cycle is orders of magnitude tighter than the gap between the legs.
    tail = np.array(trajectory[-10:])
    separation = abs(tail[0] - tail[1])
    assert separation > 1e-2, tail
    assert np.ptp(tail[::2]) < 1e-2 * separation, tail
    assert np.ptp(tail[1::2]) < 1e-2 * separation, tail


def test_safeguarded_solver_finds_the_root_on_the_cycling_pool():
    contribution, logits, observed, _mean, sd = _cycling_pool()
    t_hat, bracketed = sp._solve_propensity_saddlepoint(
        observed, contribution, logits, iterations=30
    )
    assert bool(bracketed[0])
    expected = _bisected_root(observed, contribution, logits)
    np.testing.assert_allclose(float(t_hat[0]), expected, rtol=1e-9)
    first, _ = sp._propensity_cgf_derivatives(t_hat, contribution, logits)
    residual = abs(float(first[0]) - float(observed[0])) / sd
    assert residual <= 1e-12, residual


def test_cycling_pair_is_no_longer_a_tail_failure():
    contribution, logits, observed, _mean, _sd = _cycling_pool()
    log_p, valid, diagnostics = sp.propensity_saddlepoint_log_two_sided_diagnostics(
        observed, contribution, logits
    )
    assert bool(valid[0])
    assert int(diagnostics.failure_reason_code[0]) == 0
    assert not bool(diagnostics.fallback_used[0])
    assert float(diagnostics.root_residual_null_sd[0]) <= 1e-12

    root = _bisected_root(observed, contribution, logits)
    cgf, _first, second = sp._propensity_cgf_terms(
        jnp.asarray([root]), contribution, logits
    )
    w = np.sqrt(2.0 * (root * float(observed[0]) - float(cgf[0])))
    u = root * np.sqrt(float(second[0]))
    tail = norm.sf(w) + norm.pdf(w) * (1.0 / u - 1.0 / w)
    np.testing.assert_allclose(float(np.exp(log_p[0])), min(1.0, 2.0 * tail), rtol=1e-8)
    # The LR approximation is now informative, unlike the old vacuous bound.
    assert 0.0 < float(np.exp(log_p[0])) < 0.05


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_ordinary_solver_reaches_the_bisected_root_for_skewed_both_signs(sign):
    pool = np.full(256, -0.5)
    pool[:4] = 100.0 * 0.45 ** np.arange(4)
    pool = sign * pool
    draws = jnp.asarray(12.0)
    mean = float(draws) * float(pool.mean())
    sd = np.sqrt(float(draws) * float(pool.var()))
    target = jnp.asarray(mean + sign * 7.0 * sd)

    fitted = sp._solve_saddlepoint(target, jnp.asarray(pool), draws, iterations=30)
    _cgf, first, _second = sp._cgf_terms(fitted, jnp.asarray(pool), draws)
    assert abs(float(first - target)) / sd <= 1e-12

    low, high = (-1e3, 0.0) if sign < 0 else (0.0, 1e3)
    for _ in range(200):
        middle = 0.5 * (low + high)
        _cgf, at_middle, _second = sp._cgf_terms(jnp.asarray(middle), jnp.asarray(pool), draws)
        if float(at_middle) < float(target):
            low = middle
        else:
            high = middle
    np.testing.assert_allclose(float(fitted), 0.5 * (low + high), rtol=1e-10)


def test_stratified_solver_reaches_bisected_roots_with_fpc_for_both_signs():
    base = np.full(256, -0.5)
    base[:4] = 100.0 * 0.45 ** np.arange(4)
    pool = jnp.asarray(np.stack([base, -base], axis=1)[None, :, :])
    mask = jnp.ones((1, base.size), dtype=bool)
    selected = jnp.asarray([12.0])
    fpc = jnp.asarray([0.7])
    _cgf, mean, second = sp._stratified_cgf_terms(
        jnp.zeros(2), pool, mask, selected, fpc
    )
    target = mean + jnp.asarray([7.0, -7.0]) * jnp.sqrt(second)
    fitted, bracketed = sp._solve_stratified_saddlepoint(
        target, pool, mask, selected, fpc, iterations=30
    )
    assert np.asarray(bracketed).all()
    _cgf, first, _second = sp._stratified_cgf_terms(fitted, pool, mask, selected, fpc)
    np.testing.assert_allclose(np.asarray(first), np.asarray(target), rtol=1e-12, atol=1e-12)

    # Reference K' directly. The power-CGF transform already applies the FPC
    # inside _stratified_cgf_terms; no additional correction belongs here.
    expected = []
    for gene, sign in enumerate((1.0, -1.0)):
        low, high = (-1e3, 0.0) if sign < 0 else (0.0, 1e3)
        for _ in range(200):
            middle = 0.5 * (low + high)
            _cgf, at_middle, _second = sp._stratified_cgf_terms(
                jnp.full(2, middle), pool, mask, selected, fpc
            )
            if float(at_middle[gene]) < float(target[gene]):
                low = middle
            else:
                high = middle
        expected.append(0.5 * (low + high))
    np.testing.assert_allclose(np.asarray(fitted), expected, rtol=1e-10)


@pytest.mark.parametrize("reason", [16, 128, 16 | 128])
def test_finite_bound_policy_relaxes_only_root_and_lr_bits(reason):
    bounded, usable = sp._apply_root_residual_policy(
        jnp.asarray([-9.0]), jnp.asarray([False]), jnp.asarray([reason], dtype=jnp.int32),
        jnp.asarray([1]), jnp.asarray([0.5]), jnp.asarray([-3.0]), root_policy="finite-bound",
    )
    assert bool(usable[0])
    np.testing.assert_allclose(float(bounded[0]), np.log(2.0) - 3.0)


@pytest.mark.parametrize("reason", [1, 32, 16 | 32, 128 | 256])
def test_finite_bound_policy_keeps_every_other_reason_fail_closed(reason):
    bounded, usable = sp._apply_root_residual_policy(
        jnp.asarray([-9.0]), jnp.asarray([False]), jnp.asarray([reason], dtype=jnp.int32),
        jnp.asarray([1]), jnp.asarray([0.5]), jnp.asarray([-3.0]), root_policy="finite-bound",
    )
    assert not bool(usable[0])
    assert float(bounded[0]) == -9.0
