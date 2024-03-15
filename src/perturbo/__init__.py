from importlib.metadata import version

from ._model import PERTURBO
from ._jax_module import create_plates, perturbseq_model

__version__ = version("perturbo")
