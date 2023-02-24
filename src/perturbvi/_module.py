from typing import Optional

import pyro
import pyro.distributions as dist
import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import PyroBaseModuleClass


class PerturbVIPyroModule(PyroBaseModuleClass):
    def __init__(self, **module_kwargs) -> None:
        super().__init__()

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        x = tensor_dict[REGISTRY_KEYS.X_KEY]
        return x

    @property
    def model(self, x):
        pass

    @property
    def guide(self, x):
        pass
