"""PerTurbo's NumPyro/JAX public API.

The PyTorch implementation remains available as :mod:`perturbo.legacy` for a
limited transition period.  It is deliberately not imported here so a normal
PerTurbo installation has no Torch, Pyro, or scvi-tools dependency.
"""

from importlib.metadata import PackageNotFoundError, version

from . import core
from .api import fit_from_path
from .inference import PERTURBO, PerTurboModel
from .io import MuDataSetup, get_mudata_setup, load_fit_bundle, save_fit_bundle, setup_mudata
from .results import PosteriorMedians, build_guide_effects_df
from .simulation import save_simulated_mudata, simulate_data_from_trained_model
from .censored_negative_binomial import CensoredNegativeBinomial
from .log_normal_negative_binomial import LogNormalNegativeBinomial
from .model import CensoredNegativeBinomialModel, LogNormalNegativeBinomialModel, MixtureNegativeBinomialModel, NegBinModel

try:
    __version__ = version("perturbo")
except PackageNotFoundError:
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
