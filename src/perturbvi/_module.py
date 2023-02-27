import pyro
import pyro.distributions as dist
import torch
from scvi.module.base import PyroBaseModuleClass

from ._constants import REGISTRY_KEYS


class PerturbVIPyroModule(PyroBaseModuleClass):
    def __init__(self, summary_stats, **module_kwargs) -> None:
        super().__init__()
        self.n_cells = summary_stats.n_cells
        self.n_vars = summary_stats.n_vars
        self.n_perturbations = summary_stats.n_perturbations

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        return (tensor_dict[REGISTRY_KEYS.INDICES_KEY],), tensor_dict

    def create_plates(self, idx, **tensor_dict):
        return (
            pyro.plate("cells", self.n_cells, dim=-2, subsample=idx),
            pyro.plate("perturbations", self.n_perturbations, dim=-3),
            pyro.plate("vars", self.n_vars, dim=-1),
        )

    def model(self, idx, **tensor_dict):
        cell_plate, _, var_plate = self.create_plates(idx)

        with var_plate:
            log_var_mean = pyro.sample("log_var_mean", dist.Normal(0, 4))
            log_var_dispersion = pyro.sample("log_var_dispersion", dist.Normal(2, 1))
            with cell_plate:
                pyro.sample(
                    "obs",
                    dist.NegativeBinomial(
                        total_count=log_var_dispersion.exp(),
                        logits=log_var_mean + tensor_dict[REGISTRY_KEYS.OBSERVED_LIB_SIZE].log(),
                    ),
                    obs=tensor_dict[REGISTRY_KEYS.X_KEY],
                )

    def guide(self, idx, **tensor_dict):
        _, _, var_plate = self.create_plates(idx)
        log_var_mean_mu = pyro.param("log_var_mean.mu", lambda: torch.full((self.n_vars,), 0.0))
        log_var_disp_mu = pyro.param("log_var_disp.mu", lambda: torch.full((self.n_vars,), 0.0))
        with var_plate:
            pyro.sample("log_var_mean", dist.Delta(log_var_mean_mu))
            pyro.sample("log_var_dispersion", dist.Delta(log_var_disp_mu))
