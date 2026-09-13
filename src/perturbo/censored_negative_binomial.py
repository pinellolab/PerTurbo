"""Right-censored Negative Binomial utilities for PerTurbo."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import promote_shapes, validate_sample

from perturbo.preprocessing.counts import compute_gene_clip_thresholds

DEFAULT_COUNT_CENSORING_PERCENTILE = 99.5


def require_count_censoring_percentile(percentile: float | None) -> float:
    """Validate the per-gene censoring percentile for censored likelihoods."""
    if percentile is None:
        raise ValueError(
            "count_censoring_percentile must be specified when using the censored_nb likelihood."
        )
    resolved = float(percentile)
    if not np.isfinite(resolved) or not 0.0 < resolved < 100.0:
        raise ValueError(
            "count_censoring_percentile must be in the interval (0, 100) when using the censored_nb likelihood."
        )
    return resolved


def compute_gene_count_censoring_thresholds(
    counts,
    *,
    percentile: float = DEFAULT_COUNT_CENSORING_PERCENTILE,
    threshold_floor: int = 2,
) -> np.ndarray:
    """Compute full-dataset per-gene censoring cutoffs."""
    return compute_gene_clip_thresholds(
        counts,
        percentile=percentile,
        threshold_floor=threshold_floor,
    )


class CensoredNegativeBinomial(Distribution):
    """Negative binomial likelihood with right censoring above a cutoff.

    Parameters
    ----------
    total_count
        Negative-binomial shape parameter, broadcast with the other parameters.
    logits
        NumPyro negative-binomial logits, conventionally
        ``log(mean / total_count)``.
    censoring_threshold
        Inclusive uncensored cutoff. Values greater than this threshold receive
        the upper-tail probability instead of their point probability.
    validate_args
        Whether to validate parameter and sample constraints.

    Notes
    -----
    Parameters broadcast to a common ``batch_shape`` and the event shape is
    scalar. Sampling and moment properties describe the underlying uncensored
    negative binomial; censoring is applied only by :meth:`log_prob`.
    """

    arg_constraints = {
        "total_count": constraints.greater_than_eq(0),
        "logits": constraints.real,
        "censoring_threshold": constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer
    pytree_data_fields = ("total_count", "logits", "censoring_threshold")

    def __init__(
        self,
        total_count,
        logits,
        censoring_threshold,
        *,
        validate_args=None,
    ):
        """Initialize a broadcast right-censored negative binomial."""
        total_count, logits, censoring_threshold = promote_shapes(
            total_count,
            logits,
            censoring_threshold,
        )
        batch_shape = np.broadcast_shapes(
            np.shape(total_count),
            np.shape(logits),
            np.shape(censoring_threshold),
        )
        self.total_count = jnp.broadcast_to(total_count, batch_shape)
        self.logits = jnp.broadcast_to(logits, batch_shape)
        self.censoring_threshold = jnp.broadcast_to(censoring_threshold, batch_shape)
        self.nb_dist = dist.NegativeBinomialLogits(
            total_count=self.total_count,
            logits=self.logits,
        )
        super().__init__(batch_shape=batch_shape, event_shape=(), validate_args=validate_args)

    @validate_sample
    def log_prob(self, value):
        """Evaluate point or right-tail log probability.

        Parameters
        ----------
        value
            Nonnegative integer observations broadcastable with ``batch_shape``.

        Returns
        -------
        jax.Array
            Elementwise log probability. Observations at or below the threshold
            use the ordinary negative-binomial mass; larger observations use
            ``log(1 - CDF(threshold))``.

        Notes
        -----
        JAX does not provide the required incomplete-beta derivative with
        respect to ``total_count``. The censored upper-tail branch therefore
        stops that derivative while retaining the exact likelihood value and
        the supported derivative with respect to ``logits``. The uncensored
        point-mass branch keeps the ordinary derivatives.
        """
        value = jnp.asarray(value)
        uncensored_log_prob = self.nb_dist.log_prob(value)
        # JAX currently does not support gradients of betainc with respect to the
        # concentration argument used by NegativeBinomialLogits.cdf. We keep the
        # exact censored likelihood value while stopping that unsupported path.
        cdf_dist = dist.NegativeBinomialLogits(
            total_count=jax.lax.stop_gradient(self.total_count),
            logits=self.logits,
        )
        cdf_at_threshold = cdf_dist.cdf(self.censoring_threshold)
        eps = jnp.finfo(cdf_at_threshold.dtype).eps
        safe_cdf = jnp.clip(cdf_at_threshold, min=0.0, max=1.0 - eps)
        censored_log_prob = jnp.log1p(-safe_cdf)
        return jnp.where(value > self.censoring_threshold, censored_log_prob, uncensored_log_prob)

    def sample(self, key, sample_shape=()):
        """Draw from the underlying uncensored negative binomial.

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
        """
        return self.nb_dist.sample(key, sample_shape=sample_shape)

    def expand(self, batch_shape):
        """Return the distribution broadcast to ``batch_shape``."""
        batch_shape = tuple(batch_shape)
        return CensoredNegativeBinomial(
            total_count=jnp.broadcast_to(self.total_count, batch_shape),
            logits=jnp.broadcast_to(self.logits, batch_shape),
            censoring_threshold=jnp.broadcast_to(self.censoring_threshold, batch_shape),
            validate_args=self._validate_args,
        )

    @property
    def mean(self):
        """Mean of the underlying uncensored negative binomial."""
        return self.nb_dist.mean

    @property
    def variance(self):
        """Variance of the underlying uncensored negative binomial."""
        return self.nb_dist.variance
