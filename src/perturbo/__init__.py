from importlib.metadata import version

from ._jax_module import perturbseq_guide_autonormal, perturbseq_model, run_mcmc, run_svi
from ._model import PERTURBO

__version__ = version("perturbo")
