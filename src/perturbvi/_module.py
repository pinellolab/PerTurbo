import pyro
import pyro.distributions as dist
import torch
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.infer import config_enumerate
from scvi.distributions import NegativeBinomial as SCVINegativeBinomial
from scvi.distributions import NegativeBinomialMixture as SCVINegativeBinomialMixture
from scvi.module.base import PyroBaseModuleClass
from torch.distributions.utils import broadcast_all

from ._constants import REGISTRY_KEYS


# Wraps scvi NegativeBinomial implementation for use with Pyro
class NegativeBinomial(SCVINegativeBinomial, TorchDistribution):
    pass


# Wraps scvi NegativeBinomialMixture implementation for Pyro
class NegativeBinomialMixture(SCVINegativeBinomialMixture, TorchDistribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # fixes broadcasting error when theta2 is different from theta1
        self.mu2, self.theta2 = broadcast_all(kwargs["mu2"], kwargs["theta2"])


class PerturbVIPyroModule(PyroBaseModuleClass):
    def __init__(
        self,
        summary_stats,
        **module_kwargs,
    ) -> None:
        super().__init__()
        self.n_cells = summary_stats.n_cells
        self.n_vars = summary_stats.n_vars
        self.n_perturbations = summary_stats.n_perturbations
        self.n_covariates = 1  # include (inferred) size factor by default
        if "n_extra_continuous_covs" in summary_stats:
            self.n_covariates += summary_stats.n_extra_continuous_covs
        self.n_batches = summary_stats.n_batch

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        # tack on size factor after the other continuous covariates
        size_factor = tensor_dict[REGISTRY_KEYS.SIZE_FACTOR_KEY]
        if REGISTRY_KEYS.CONT_COVS_KEY in tensor_dict:
            tensor_dict[REGISTRY_KEYS.CONT_COVS_KEY] = torch.cat(
                (tensor_dict[REGISTRY_KEYS.CONT_COVS_KEY], size_factor), dim=-1
            )
        else:
            tensor_dict[REGISTRY_KEYS.CONT_COVS_KEY] = size_factor

        # return indices and then the rest of the tensors
        return (tensor_dict[REGISTRY_KEYS.INDICES_KEY],), tensor_dict

    def create_plates(self, idx, subsample_size=None, **tensor_dict):
        return (
            pyro.plate("cells", self.n_cells, dim=-2, subsample=idx),
            pyro.plate("perturbations", self.n_perturbations, dim=-2),
            pyro.plate("batches", self.n_batches, dim=-2),
            pyro.plate("vars", self.n_vars, dim=-1),
            pyro.plate("covariates", self.n_covariates, dim=-2),
        )

    @config_enumerate
    def model(self, idx, **tensor_dict):
        pyro.module("perturbvi", self)
        (
            cell_plate,
            perturbation_plate,
            batch_plate,
            var_plate,
            covariate_plate,
        ) = self.create_plates(idx)
        batch = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        size_factor = tensor_dict[REGISTRY_KEYS.SIZE_FACTOR_KEY]
        perturbations = tensor_dict[REGISTRY_KEYS.PERTURBATION_KEY]
        covariates = tensor_dict[REGISTRY_KEYS.CONT_COVS_KEY]
        # log_var_mean_global = pyro.sample("log_var_mean_global", dist.Normal(0.0, 4.0))

        with var_plate:
            with batch_plate:
                # n_batches x n_vars
                batch_effect_size = pyro.sample("batch_effect", dist.Normal(0.0, 1.0))
                batch_effects = batch_effect_size[batch.squeeze(), ...]
            with covariate_plate:
                # n_covariates x n_vars
                covariate_effect_size = pyro.sample(
                    "covariate_effect", dist.Normal(0.0, 1.0)
                )
            with perturbation_plate:
                prior_inclusion_prob = 1e-3
                perturb_mean_mask = pyro.sample(
                    "perturb_include", dist.Bernoulli(prior_inclusion_prob)
                )
                epsilon = 1e-8
                perturb_mean_lfc = pyro.sample(
                    "perturb_mean_lfc",
                    dist.Normal(0, 1.0 * perturb_mean_mask + epsilon),
                )
                perturb_disp_lfc = pyro.sample("perturb_disp_lfc", dist.Normal(0, 0.1))

            log_var_mean = pyro.sample("log_var_mean", dist.Normal(0.0, 3.0))
            log_var_dispersion = pyro.sample(
                "log_var_dispersion", dist.Normal(0.0, 2.0)
            )

            nb_log_dispersion = log_var_dispersion + perturbations @ perturb_disp_lfc
            nb_log_mean = (
                log_var_mean
                + perturbations @ perturb_mean_lfc
                + size_factor
                + batch_effects
                + covariates @ covariate_effect_size
            )

            with cell_plate:
                return pyro.sample(
                    "obs",
                    dist.NegativeBinomial(
                        total_count=nb_log_dispersion.exp(),
                        logits=nb_log_mean - nb_log_dispersion,
                    ),
                    obs=tensor_dict[REGISTRY_KEYS.X_KEY],
                )

    # def guide(self, idx, **tensor_dict):
    # return self._guide(idx, **tensor_dict)

    def guide(self, idx, init_scale=0.2, **tensor_dict):
        pyro.module("perturbvi", self)
        # scale_factor = pyro.param("scale_factor", torch.tensor(init_scale).log()).exp()
        (
            cell_plate,
            perturbation_plate,
            batch_plate,
            var_plate,
            covariate_plate,
        ) = self.create_plates(idx)

        # shared global mean parameters
        # log_var_mean_global_mu = pyro.param(
        #     "log_var_mean_global.mu", lambda: torch.tensor((0.0,))
        # )
        # pyro.sample("log_var_mean_global", dist.Delta(log_var_mean_global_mu))

        log_var_mean_mu = pyro.param(
            "log_var_mean.mu", lambda: torch.zeros((self.n_vars,))
        )
        log_var_disp_mu = pyro.param(
            "log_var_disp.mu", lambda: torch.zeros((self.n_vars,))
        )

        batch_effect_mu = pyro.param(
            "batch_effect.mu", lambda: torch.zeros((self.n_batches, self.n_vars))
        )
        batch_effect_sigma = pyro.param(
            "batch_effect.sigma",
            lambda: torch.full((self.n_batches, self.n_vars), init_scale),
            constraint=dist.constraints.positive,
        )

        covariate_effect_mu = pyro.param(
            "covariate_effect.mu", lambda: torch.zeros((self.n_covariates, self.n_vars))
        )
        covariate_effect_sigma = pyro.param(
            "covariate_effect.sigma",
            lambda: torch.full((self.n_covariates, self.n_vars), init_scale),
            constraint=dist.constraints.positive,
        )

        log_var_mean_sigma = pyro.param(
            "log_var_mean.sigma",
            lambda: torch.full((self.n_vars,), init_scale),
            constraint=dist.constraints.positive,
        )
        log_var_disp_sigma = pyro.param(
            "log_var_disp.sigma",
            lambda: torch.full((self.n_vars,), init_scale),
            constraint=dist.constraints.positive,
        )

        perturb_mean_lfc_mu = pyro.param(
            "perturb_mean_lfc.mu",
            lambda: torch.zeros((self.n_perturbations, self.n_vars)),
        )
        perturb_disp_lfc_mu = pyro.param(
            "perturb_disp_lfc.mu",
            lambda: torch.zeros((self.n_perturbations, self.n_vars)),
        )
        perturb_lfc_mu = torch.stack((perturb_mean_lfc_mu, perturb_disp_lfc_mu), dim=-1)

        perturb_lfc_scale_tril = pyro.param(
            "perturb_lfc.scale_tril",
            lambda: torch.eye(2).repeat((self.n_perturbations, self.n_vars, 1, 1))
            * init_scale,
            constraint=dist.constraints.lower_cholesky,
        )

        with var_plate:
            pyro.sample(
                "log_var_mean", dist.Normal(log_var_mean_mu, log_var_mean_sigma)
            )
            pyro.sample(
                "log_var_dispersion", dist.Normal(log_var_disp_mu, log_var_disp_sigma)
            )
            with batch_plate:
                pyro.sample(
                    "batch_effect", dist.Normal(batch_effect_mu, batch_effect_sigma)
                )
            with covariate_plate:
                pyro.sample(
                    "covariate_effect",
                    dist.Normal(covariate_effect_mu, covariate_effect_sigma),
                )

            with perturbation_plate:
                perturb_lfc = pyro.sample(
                    "perturb_lfc",
                    dist.MultivariateNormal(
                        perturb_lfc_mu,
                        scale_tril=perturb_lfc_scale_tril,
                    ),
                    infer={"is_auxiliary": True},
                )

                pyro.sample("perturb_mean_lfc", dist.Delta(perturb_lfc[..., 0]))
                pyro.sample("perturb_disp_lfc", dist.Delta(perturb_lfc[..., 1]))

    @staticmethod
    def get_perturbation_effects():
        """Return the perturbation effects on each variable's mean and variance."""
        store = pyro.get_param_store()
        return (
            store["perturb_mean_lfc.mu"].detach().cpu().numpy(),
            store["perturb_disp_lfc.mu"].detach().cpu().numpy(),
        )
