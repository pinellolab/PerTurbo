"""PerTurbo's NumPyro/JAX public API.

The PyTorch implementation remains available as :mod:`perturbo.legacy` for a
limited transition period. It is deliberately not imported here, so a normal
PerTurbo installation has no Torch, Pyro or scvi-tools dependency.
"""

import jax as _jax

# Enable float64 before anything else touches JAX. This is a process-global
# setting, so importing perturbo changes the dtype promotion of any other JAX
# code in the same process - an accepted trade, because deployment is via CLI
# or subprocess rather than in-process library use.
#
# It is permissive, not coercive: arrays built with an explicit
# ``dtype=jnp.float32`` stay float32, so the score kernels keep their precision
# and memory profile (verified by test_jax_score_kernels_stay_float32). What it
# buys is float64 where float64 is meant - most importantly the parametric
# tails, whose p-values reach 1e-300 and were previously destroyed by float32's
# ~1e-45 floor, and the analytic null's near-cancelling control-pool sums.
_jax.config.update("jax_enable_x64", True)

from importlib.metadata import PackageNotFoundError, version

from . import core
from .inference import PerTurboModel, PERTURBO
from .io import MuDataSetup, get_mudata_setup, load_fit_bundle, save_fit_bundle, setup_mudata
from .results import PosteriorMedians, build_guide_effects_df
from .simulation import save_simulated_mudata, simulate_data_from_trained_model
from .censored_negative_binomial import CensoredNegativeBinomial
from .log_normal_negative_binomial import LogNormalNegativeBinomial
from .model import CensoredNegativeBinomialModel, LogNormalNegativeBinomialModel, MixtureNegativeBinomialModel, NegBinModel
from .api import fit_from_path

try:
    __version__ = version("perturbo")
except PackageNotFoundError:  # editable checkout without metadata
    __version__ = "0+unknown"

BaselinePosteriorSummary = core.BaselinePosteriorSummary
BetaFit = core.BetaFit
ControlFit = core.ControlFit
PerTurboData = core.PerTurboData
SVIConfig = core.SVIConfig

fit_control = core.fit_control
fit_perturbation_effects = core.fit_perturbation_effects
load_analysis_cells = core.load_analysis_cells
load_controls = core.load_controls
summarize_betas = core.summarize_betas

__all__ = [
    "BaselinePosteriorSummary",
    "BetaFit",
    "ControlFit",
    "PerTurboData",
    "PerTurboModel",
    "CensoredNegativeBinomial",
    "CensoredNegativeBinomialModel",
    "LogNormalNegativeBinomial",
    "LogNormalNegativeBinomialModel",
    "MixtureNegativeBinomialModel",
    "MuDataSetup",
    "NegBinModel",
    "PERTURBO",
    "PosteriorMedians",
    "build_guide_effects_df",
    "SVIConfig",
    "fit_control",
    "fit_from_path",
    "fit_perturbation_effects",
    "get_mudata_setup",
    "load_analysis_cells",
    "load_controls",
    "load_fit_bundle",
    "save_fit_bundle",
    "save_simulated_mudata",
    "setup_mudata",
    "simulate_data_from_trained_model",
    "summarize_betas",
]
