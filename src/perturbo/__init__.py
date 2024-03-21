from importlib.metadata import version

from ._jax_module import perturbseq_model, perturbseq_guide_autonormal
from ._model import PERTURBO

__version__ = version("perturbo")
