"""Deprecated PyTorch/Pyro PerTurbo implementation.

Install ``perturbo[legacy]`` before importing this namespace.  New analyses
should use :class:`perturbo.PERTURBO`, the NumPyro/JAX implementation.
"""

from __future__ import annotations

import warnings
from importlib.metadata import PackageNotFoundError, version

warnings.warn(
    "perturbo.legacy is deprecated and will be removed in a future major release; "
    "migrate to the NumPyro/JAX perturbo API.",
    DeprecationWarning,
    stacklevel=2,
)

from . import models, simulation
from .models import PERTURBO
from .simulation import Learn_Data, Simulate_Data

try:
    __version__ = version("perturbo")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["Learn_Data", "PERTURBO", "Simulate_Data", "__version__", "models", "simulation"]
