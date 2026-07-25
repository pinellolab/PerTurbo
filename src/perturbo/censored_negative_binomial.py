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
    """Negative binomial likelihood with right censoring above a fixed cutoff."""

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
        return self.nb_dist.sample(key, sample_shape=sample_shape)

    def expand(self, batch_shape):
        batch_shape = tuple(batch_shape)
        return CensoredNegativeBinomial(
            total_count=jnp.broadcast_to(self.total_count, batch_shape),
            logits=jnp.broadcast_to(self.logits, batch_shape),
            censoring_threshold=jnp.broadcast_to(self.censoring_threshold, batch_shape),
            validate_args=self._validate_args,
        )

    @property
    def mean(self):
        return self.nb_dist.mean

    @property
    def variance(self):
        return self.nb_dist.variance
