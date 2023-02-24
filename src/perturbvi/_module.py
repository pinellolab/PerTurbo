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
        return (tensor_dict[REGISTRY_KEYS.X_KEY],), {}

    def create_plates(self, x):
        n_cells, n_vars = x.shape
        cell_plate = pyro.plate("cells", n_cells, dim=-2)
        vars_plate = pyro.plate("vars", n_vars, dim=-1)
        return (cell_plate, vars_plate)

    def model(self, x):
        cell_plate, vars_plate = self.create_plates(x)
        with vars_plate:
            log_var_mean = pyro.sample("log_var_mean", dist.Normal(2, 4))
            log_var_dispersion = pyro.sample("log_var_dispersion", dist.Normal(2, 1))
            with cell_plate:
                pyro.sample(
                    "obs",
                    dist.NegativeBinomial(total_count=log_var_dispersion.exp(), logits=log_var_mean),
                    obs=x,
                )

    def guide(self, x):
        _, vars_plate = self.create_plates(x)
        _, n_vars = x.shape
        log_var_mean_mu = pyro.param("log_var_mean.mu", lambda: torch.full((n_vars,), 0.))
        log_var_disp_mu = pyro.param("log_var_disp.mu", lambda: torch.full((n_vars,), 0.))
        with vars_plate:
            pyro.sample("log_var_mean", dist.Delta(log_var_mean_mu))
            pyro.sample("log_var_dispersion", dist.Delta(log_var_disp_mu))
