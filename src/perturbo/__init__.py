from importlib.metadata import version

from . import models, simulation
from .models import PERTURBO
from .simulation import Learn_Data, Simulate_Data, Support_Functions

__version__ = version("perturbo")
