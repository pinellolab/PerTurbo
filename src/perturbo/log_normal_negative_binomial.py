"""Log-normal Negative Binomial utilities for PerTurbo."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from numpy.polynomial.hermite_e import hermegauss
import numpyro.distributions as dist
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import promote_shapes, validate_sample


def _get_quad_rule(num_quad_points: int, dtype) -> tuple[jnp.ndarray, jnp.ndarray]:
    quad_rule = hermegauss(num_quad_points)
    points = jnp.asarray(quad_rule[0], dtype=dtype)
    # hermegauss returns quadrature *weights*; the mixture needs their logs.
    log_weights = jnp.log(jnp.asarray(quad_rule[1], dtype=dtype))
    log_weights = log_weights - logsumexp(log_weights)
    return points, log_weights


class LogNormalNegativeBinomial(Distribution):
    """Negative binomial with additive Normal noise on its log mean.

    Parameters
    ----------
    total_count
        Negative-binomial shape parameter, broadcast with the other parameters.
    logits
        NumPyro negative-binomial logits, conventionally
        ``log(mean / total_count)`` before the Normal perturbation.
    multiplicative_noise_scale
        Positive standard deviation of the Normal perturbation on ``logits``.
        Equivalently, expected counts receive a log-normal multiplicative factor.
    num_quad_points
        Number of Gauss-Hermite points used to approximate :meth:`log_prob`.
    validate_args
        Whether to validate parameter and sample constraints.

    Notes
    -----
    Parameters broadcast to a common ``batch_shape`` and the event shape is
    scalar. Log probabilities integrate over the continuous Normal mixing
    variable by quadrature. Sampling instead draws that variable directly and
    then draws from the resulting negative binomial.
    """

    arg_constraints = {
        "total_count": constraints.greater_than_eq(0),
        "logits": constraints.real,
        "multiplicative_noise_scale": constraints.positive,
    }
    support = constraints.nonnegative_integer
    pytree_data_fields = (
        "total_count",
        "logits",
        "multiplicative_noise_scale",
        "log_weights",
    )

    def __init__(
        self,
        total_count,
        logits,
        multiplicative_noise_scale,
        *,
        num_quad_points: int = 5,
        validate_args=None,
    ):
        """Initialize a broadcast log-normal negative-binomial mixture.

        Raises
        ------
        ValueError
            If ``num_quad_points`` is less than one.
        """
        if num_quad_points < 1:
            raise ValueError("num_quad_points must be at least 1")

        total_count, logits, multiplicative_noise_scale = promote_shapes(
            total_count, logits, multiplicative_noise_scale
        )
        batch_shape = jnp.broadcast_shapes(
            jnp.shape(total_count), jnp.shape(logits), jnp.shape(multiplicative_noise_scale)
        )
        total_count = jnp.broadcast_to(total_count, batch_shape)
        logits = jnp.broadcast_to(logits, batch_shape)
        multiplicative_noise_scale = jnp.broadcast_to(multiplicative_noise_scale, batch_shape)

        quad_points, log_weights = _get_quad_rule(num_quad_points, logits.dtype)
        quad_logits = logits[..., None] + multiplicative_noise_scale[..., None] * quad_points
        self.nb_dist = dist.NegativeBinomialLogits(
            total_count=total_count[..., None],
            logits=quad_logits,
        )

        self.multiplicative_noise_scale = multiplicative_noise_scale
        self.total_count = total_count
        self.logits = logits
        self.num_quad_points = num_quad_points
        self.log_weights = log_weights

        quad_batch_shape = self.nb_dist.batch_shape[:-1]
        batch_shape = np.broadcast_shapes(
            multiplicative_noise_scale.shape,
            quad_batch_shape,
        )
        event_shape = ()
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @validate_sample
    def log_prob(self, value):
        """Approximate the marginal log probability by Gauss-Hermite quadrature.

        Parameters
        ----------
        value
            Nonnegative integer observations broadcastable with ``batch_shape``.

        Returns
        -------
        jax.Array
            Marginal log probabilities with the quadrature axis reduced.
        """
        value = jnp.asarray(value)
        nb_log_prob = self.nb_dist.log_prob(value[..., None])
        return logsumexp(self.log_weights + nb_log_prob, axis=-1)

    def sample(self, key, sample_shape=()):
        """Draw from the continuous log-normal mixture.

        Parameters
        ----------
        key
            JAX PRNG key.
        sample_shape
            Leading sample dimensions.

        Returns
        -------
        jax.Array
            Integer draws with shape ``sample_shape + batch_shape``.

        Notes
        -----
        Sampling is not restricted to the quadrature support: it draws an exact
        Normal mixing value, then conditionally samples a negative binomial.
        """
        normals_key, nb_key = jax.random.split(key)
        normals = dist.Normal(0.0, self.multiplicative_noise_scale).sample(
            normals_key,
            sample_shape=sample_shape,
        )
        nb_dist = dist.NegativeBinomialLogits(
            total_count=self.total_count,
            logits=self.logits + normals,
        )
        # The normal draw already expanded the NB batch by sample_shape.
        return nb_dist.sample(nb_key)

    def expand(self, batch_shape):
        """Return the distribution broadcast to ``batch_shape``."""
        batch_shape = tuple(batch_shape)
        total_count = jnp.broadcast_to(self.total_count, batch_shape)
        logits = jnp.broadcast_to(self.logits, batch_shape)
        multiplicative_noise_scale = jnp.broadcast_to(self.multiplicative_noise_scale, batch_shape)
        return LogNormalNegativeBinomial(
            total_count,
            logits,
            multiplicative_noise_scale,
            num_quad_points=self.num_quad_points,
            validate_args=self._validate_args,
        )

    @property
    def mean(self):
        """Analytic marginal mean after integrating over Normal log noise."""
        return jnp.exp(self.logits + jnp.log(self.total_count) + 0.5 * self.multiplicative_noise_scale**2)

    @property
    def variance(self):
        """Analytic marginal variance of the continuous mixture."""
        kappa = jnp.exp(self.multiplicative_noise_scale**2) * (1 + 1 / self.total_count) - 1
        return self.mean + kappa * self.mean**2
