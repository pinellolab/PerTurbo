import pyro
import pyro.distributions as dist
import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import PyroBaseModuleClass


PERTURBATION_REGISTRY_KEY = "perturbations"


class PerturbVIPyroModule(PyroBaseModuleClass):
    def __init__(self, **module_kwargs) -> None:
        super().__init__()

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        return (
            tensor_dict[REGISTRY_KEYS.X_KEY],
            tensor_dict[PERTURBATION_REGISTRY_KEY],
            tensor_dict[REGISTRY_KEYS.OBSERVED_LIB_SIZE],
        ), {}

    def model(self, x, perturbations, library_size):
        n_cells, n_vars = x.shape
        with pyro.plate("vars", n_vars):
            log_var_mean = pyro.sample("log_var_mean", dist.Normal(0, 4))
            log_var_dispersion = pyro.sample("log_var_dispersion", dist.Normal(2, 1))
            with pyro.plate("cells", n_cells):
                pyro.sample(
                    "obs",
                    dist.NegativeBinomial(
                        total_count=log_var_dispersion.exp(),
                        logits=log_var_mean + library_size.log(),
                    ),
                    obs=x,
                )

    def guide(self, x, perturbations, library_size):
        _, n_vars = x.shape
        log_var_mean_mu = pyro.param("log_var_mean.mu", lambda: torch.full((n_vars,), 0.0))
        log_var_disp_mu = pyro.param("log_var_disp.mu", lambda: torch.full((n_vars,), 0.0))
        with pyro.plate("vars", n_vars):
            pyro.sample("log_var_mean", dist.Delta(log_var_mean_mu))
            pyro.sample("log_var_dispersion", dist.Delta(log_var_disp_mu))
