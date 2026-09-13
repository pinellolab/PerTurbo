# Copyright Contributors to the Pyro project.
# Original NumPyro portions are Apache-2.0 licensed; see NOTICE and LICENSES/Apache-2.0.txt.
"""Local copy of NumPyro's ``plate`` with support for explicit subsample indices.

Adapted from NumPyro 0.20.1 so we can prototype a Pyro-style ``subsample=``
argument locally before proposing it upstream.
"""

from __future__ import annotations

from typing import Optional
import warnings

import jax
from jax import lax, random
import jax.numpy as jnp
from jax.typing import ArrayLike

import numpyro
from numpyro._typing import Message
from numpyro.primitives import CondIndepStackFrame, Messenger, apply_stack
from numpyro.util import find_stack_level


def _subsample_fn(
    size: int, subsample_size: int, rng_key: Optional[ArrayLike] = None
) -> ArrayLike:
    if rng_key is None:
        raise ValueError(
            "Missing random key to generate subsample indices."
            " Algorithms like HMC/NUTS do not support subsampling."
            " You might want to use SVI or HMCECS instead."
        )
    if jax.default_backend() == "cpu":
        rng_keys = random.split(rng_key, subsample_size)

        def body_fn(val, idx):
            i_p1 = size - idx
            i = i_p1 - 1
            j = random.randint(rng_keys[idx], (), 0, i_p1)
            val = val.at[jnp.array([i, j])].set(val[jnp.array([j, i])])
            return val, None

        val, _ = lax.scan(body_fn, jnp.arange(size), jnp.arange(subsample_size))
        return val[-subsample_size:]
    return random.choice(rng_key, size, (subsample_size,), replace=False)


class plate(Messenger):
    """NumPyro plate with optional explicit subsample indices."""

    def __init__(
        self,
        name: str,
        size: int,
        subsample_size: int | None = None,
        dim: int | None = None,
        subsample: ArrayLike | None = None,
    ) -> None:
        self.name = name
        assert size > 0, "size of plate should be positive"
        self.size = size
        if dim is not None and dim >= 0:
            raise ValueError("dim arg must be negative.")
        self.dim, self._indices = self._subsample(
            self.name,
            self.size,
            subsample_size,
            subsample,
            dim,
        )
        self.subsample_size = self._indices.shape[0]
        super().__init__()

    @staticmethod
    def _subsample(
        name: str,
        size: int,
        subsample_size: int | None,
        subsample: ArrayLike | None,
        dim: int | None,
    ) -> tuple[int, jnp.ndarray]:
        if subsample is not None:
            subsample = jnp.asarray(subsample)
            if subsample.ndim != 1:
                raise ValueError("subsample must be a 1D array of indices.")
            if subsample.shape[0] == 0:
                raise ValueError("subsample must contain at least one index.")
            if not jnp.issubdtype(subsample.dtype, jnp.integer):
                subsample = subsample.astype(jnp.int32)
            if subsample_size is None:
                subsample_size = int(subsample.shape[0])
            msg_value = subsample
        else:
            msg_value = None if (subsample_size is not None and size != subsample_size) else jnp.arange(size)

        msg: Message = {
            "type": "plate",
            "fn": _subsample_fn,
            "name": name,
            "args": (size, subsample_size),
            "kwargs": {"rng_key": None},
            "value": msg_value,
            "scale": 1.0,
            "cond_indep_stack": [],
        }
        apply_stack(msg)
        resolved_subsample = msg["value"]
        assert isinstance(resolved_subsample, jnp.ndarray)
        resolved_subsample_size = msg["args"][1]
        if (
            resolved_subsample_size is not None
            and resolved_subsample_size != resolved_subsample.shape[0]
        ):
            warnings.warn(
                "subsample_size does not match len(subsample), {} vs {}.".format(
                    resolved_subsample_size, len(resolved_subsample)
                )
                + " Did you accidentally use different subsample_size in the model and guide?",
                stacklevel=find_stack_level(),
            )
        cond_indep_stack: list[CondIndepStackFrame] = msg["cond_indep_stack"]
        occupied_dims = {f.dim for f in cond_indep_stack}
        if dim is None:
            new_dim = -1
            while new_dim in occupied_dims:
                new_dim -= 1
            dim = new_dim
        else:
            assert dim not in occupied_dims
        return dim, resolved_subsample

    def __enter__(self):
        super().__enter__()
        return self._indices

    @staticmethod
    def _get_batch_shape(
        cond_indep_stack: list[CondIndepStackFrame],
    ) -> tuple[int, ...]:
        n_dims = max(-f.dim for f in cond_indep_stack)
        batch_shape = [1] * n_dims
        for f in cond_indep_stack:
            batch_shape[f.dim] = f.size
        return tuple(batch_shape)

    def process_message(self, msg: Message) -> None:
        if msg["type"] not in ("param", "sample", "plate", "deterministic"):
            if msg["type"] == "control_flow":
                raise NotImplementedError(
                    "Cannot use control flow primitive under a `plate` primitive."
                    " Please move those `plate` statements into the control flow"
                    " body function. See `scan` documentation for more information."
                )
            return

        if (
            "block_plates" in msg.get("infer", {})
            and self.name in msg["infer"]["block_plates"]
        ):
            return

        cond_indep_stack: list[CondIndepStackFrame] = msg["cond_indep_stack"]
        frame = CondIndepStackFrame(self.name, self.dim, self.subsample_size)
        cond_indep_stack.append(frame)
        if msg["type"] == "deterministic":
            return
        if msg["type"] == "sample":
            expected_shape = self._get_batch_shape(cond_indep_stack)
            dist_batch_shape = msg["fn"].batch_shape
            if "sample_shape" in msg["kwargs"]:
                dist_batch_shape = msg["kwargs"]["sample_shape"] + dist_batch_shape
                msg["kwargs"]["sample_shape"] = ()
            overlap_idx = max(len(expected_shape) - len(dist_batch_shape), 0)
            trailing_shape = expected_shape[overlap_idx:]
            broadcast_shape = lax.broadcast_shapes(
                trailing_shape, tuple(dist_batch_shape)
            )
            batch_shape = expected_shape[:overlap_idx] + broadcast_shape
            msg["fn"] = msg["fn"].expand(batch_shape)
        if self.size != self.subsample_size:
            scale = 1.0 if msg["scale"] is None else msg["scale"]
            msg["scale"] = scale * (
                self.size / self.subsample_size if self.subsample_size else 1
            )

    def postprocess_message(self, msg: Message) -> None:
        if msg["type"] in ("subsample", "param") and self.dim is not None:
            event_dim = msg["kwargs"].get("event_dim")
            if event_dim is not None:
                assert event_dim >= 0
                dim = self.dim - event_dim
                shape = jnp.shape(msg["value"])
                if len(shape) >= -dim and shape[dim] != 1:
                    if shape[dim] != self.size:
                        if msg["type"] == "param":
                            statement = "numpyro.param({}, ..., event_dim={})".format(
                                msg["name"], event_dim
                            )
                        else:
                            statement = "numpyro.subsample(..., event_dim={})".format(
                                event_dim
                            )
                        raise ValueError(
                            "Inside numpyro.plate({}, {}, dim={}) invalid shape of {}: {}".format(
                                self.name, self.size, self.dim, statement, shape
                            )
                        )
                    if self.subsample_size < self.size:
                        value = msg["value"]
                        msg["value"] = jnp.take(value, self._indices, dim)


# Keep local autoguides compatible with ``isinstance(p, numpyro.plate)`` checks.
numpyro.plate = plate
