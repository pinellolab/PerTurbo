from typing import Iterable, Optional

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.infer.autoguide import AutoNormal, init_to_mean
from pyro.nn import PyroModule, PyroParam, PyroSample
from scvi.distributions import NegativeBinomial as SCVINegativeBinomial
from scvi.distributions import NegativeBinomialMixture as SCVINegativeBinomialMixture
from scvi.module.base import PyroBaseModuleClass
from torch.distributions.utils import broadcast_all

from ._constants import REGISTRY_KEYS


class LogNormalNegativeBinomial(dist.LogNormalNegativeBinomial):
    def sample(self, sample_shape=torch.Size()):
        normals = (
            dist.Normal(0, self.multiplicative_noise_scale)
            .expand(self.batch_shape)
            .sample(sample_shape=sample_shape)
        )
        return dist.NegativeBinomial(
            total_count=self.total_count, logits=self.logits + normals
        ).sample()


# Wraps scvi NegativeBinomial implementation for use with Pyro
class NegativeBinomial(SCVINegativeBinomial, TorchDistribution):
    pass


# Wraps scvi NegativeBinomialMixture implementation for Pyro
class NegativeBinomialMixture(SCVINegativeBinomialMixture, TorchDistribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # fixes broadcasting error when theta2 is different from theta1
        self.mu2, self.theta2 = broadcast_all(kwargs["mu2"], kwargs["theta2"])


class PerTurboPyroModule(PyroBaseModuleClass):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self._model = PerTurboPyroModel(*args, **kwargs)
        self._guide = AutoNormal(
            self.model, init_loc_fn=init_to_mean, create_plates=self.model.create_plates
        )
        self._get_fn_args_from_batch = self._model._get_fn_args_from_batch

    @property
    def model(self):
        return self._model

    @property
    def guide(self):
        return self._guide

    @property
    def list_obs_plate_vars(self):
        raise NotImplementedError("Not Implemented")
        return self.model.list_obs_plate_vars()

    def get_element_effects(self):
        """Return the perturbation effects on each variable's mean and variance."""
        store = pyro.get_param_store()

        element_mu = store["element_mean_lfc.mu"].detach().cpu()
        element_sigma = store["element_mean_lfc.sigma"].detach().cpu()

        return (element_mu.numpy(), element_sigma.numpy())

    def get_perturbation_effects(self):
        """Return the perturbation effects on each variable's mean and variance."""
        store = pyro.get_param_store()
        guide_by_element = self.guide_by_element.detach().cpu()

        element_mu = guide_by_element @ store["element_mean_lfc.mu"].detach().cpu()
        element_sigma = (
            guide_by_element @ store["element_mean_lfc.sigma"].detach().cpu()
        )

        q_mu = element_mu + store["perturb_mean_lfc.mu"].detach().cpu()
        q_sigma = torch.sqrt(
            element_sigma**2 + store["perturb_mean_lfc.sigma"].detach().cpu() ** 2
        )

        return (q_mu.numpy(), q_sigma.numpy())


class PerTurboPyroModel(PyroModule):
    def __init__(
        self,
        summary_stats,
        guide_by_element: torch.Tensor,
        likelihood="lnnb",
        n_cats_per_cov: Optional[Iterable[int]] = None,
        **module_kwargs,
    ) -> None:
        super().__init__()
        self.n_obs = summary_stats.n_cells
        self.n_vars = summary_stats.n_vars
        self.n_perturbations = summary_stats.n_perturbations
        self.n_cont_covariates = 1  # include (inferred) size factor by default
        if "n_targeted_elements" in summary_stats:
            # self.n_elements = summary_stats.n_targeted_elements
            assert summary_stats.n_targeted_elements == guide_by_element.shape[1]
        self.n_elements = guide_by_element.shape[1]
        self.guide_by_element = guide_by_element
        if "n_extra_continuous_covs" in summary_stats:
            self.n_cont_covariates += summary_stats.n_extra_continuous_covs
        if "n_extra_categorical_covs" in summary_stats:
            self.n_cat_covariates = summary_stats.n_extra_categorical_covs
            self.n_cat_list = n_cats_per_cov
        else:
            self.n_cat_covariates = 0
            self.n_cat_list = []

        self.n_batches = summary_stats.n_batch
        self.likelihood = likelihood

        self.global_mean = PyroParam(torch.tensor(0.0))
        self.global_slope = PyroParam(torch.tensor(0.0))
        self.global_offset = PyroParam(torch.tensor(0.0))
        self.global_disp_noise = PyroSample(dist.LogNormal(0,0.1))

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
            pyro.plate("cells", self.n_obs, dim=-2, subsample=idx),
            pyro.plate("guides", self.n_perturbations, dim=-2),
            pyro.plate("elements", self.n_elements, dim=-2),
            pyro.plate("batches", self.n_batches, dim=-2),
            pyro.plate("vars", self.n_vars, dim=-1),
            pyro.plate("cont_covariates", self.n_cont_covariates, dim=-2),
        )

    def forward(self, idx, **tensor_dict):
        (
            cell_plate,
            perturbation_plate,
            element_plate,
            batch_plate,
            var_plate,
            cont_cov_plate,
        ) = self.create_plates(idx)
        batch = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        size_factor = tensor_dict[REGISTRY_KEYS.SIZE_FACTOR_KEY]
        perturbations = tensor_dict[REGISTRY_KEYS.PERTURBATION_KEY]
        cont_covariates = tensor_dict[REGISTRY_KEYS.CONT_COVS_KEY]

        # Estimate strength of gRNA effect sharing
        pooling_prior_loc = torch.tensor(-3.0, device=idx.device)
        pooling_prior_scale = torch.tensor(1.0, device=idx.device)
        log_pooling = pyro.sample(
            "log_pooling", dist.Normal(pooling_prior_loc, pooling_prior_scale)
        )

        with var_plate:
            # mean and dispersion of each gene's expression
            gene_mean_prior_scale = torch.tensor(3.0, device=idx.device)
            nb_log_mean_gene = pyro.sample(
                "log_var_mean", dist.Normal(self.global_mean, gene_mean_prior_scale)
            )
            nb_log_disp_gene = pyro.sample(
                "log_var_dispersion",
                dist.Normal(
                    nb_log_mean_gene * self.global_slope + self.global_offset,
                    self.global_disp_noise,
                ),
            )

            if self.likelihood == "lnnb":
                # additional noise for LogNormalNegativeBinomial likelihood
                noise_prior_rate = torch.tensor(10.0, device=idx.device)
                multiplicative_noise = pyro.sample(
                    "multiplicative_noise", dist.Exponential(noise_prior_rate)
                )

            if self.likelihood == "nb_mix":
                # mixture_logits = pyro.sample("mixture_logits", dist.Normal(-1.0, 0.01))
                mixture_probs = torch.tensor([0.1, 0.9], device=idx.device)

            with batch_plate:
                # batch effects: n_batches x n_vars
                batch_effect_prior_scale = torch.tensor(1.0, device=idx.device)
                batch_effect_size = pyro.sample(
                    "batch_effect", dist.Normal(0.0, batch_effect_prior_scale)
                )
                batch_effects = batch_effect_size[batch.squeeze(), ...]

            cov_prior_sigma = torch.tensor(1.0, device=idx.device)
            with cont_cov_plate:
                # covariate effects: n_cont_covariates x n_vars
                cont_cov_effect_size = pyro.sample(
                    "cont_cov_effect", dist.Normal(0.0, cov_prior_sigma)
                )
                covariate_effects = cont_covariates @ cont_cov_effect_size

            with element_plate:
                # element effects: n_elements x n_vars
                element_mean_lfc_prior_scale = torch.tensor(0.05, device=idx.device)
                element_disp_lfc_prior_scale = torch.tensor(0.05, device=idx.device)
                element_mean_lfc = pyro.sample(
                    "element_mean_lfc", dist.Cauchy(0.0, element_mean_lfc_prior_scale)
                )
                element_disp_lfc = pyro.sample(
                    "element_disp_lfc", dist.Cauchy(0.0, element_disp_lfc_prior_scale)
                )

            with perturbation_plate:
                # perturbation effects: n_perturbations x n_vars
                guide_by_element = self.guide_by_element.to(device=idx.device)
                perturb_mean_lfc = pyro.sample(
                    "perturb_mean_lfc",
                    dist.Normal(guide_by_element @ element_mean_lfc, log_pooling.exp()),
                )
                perturb_disp_lfc = pyro.sample(
                    "perturb_disp_lfc",
                    dist.Normal(guide_by_element @ element_disp_lfc, 0.1),
                )

            # calculate overall parameter values for unperturbed cells
            nb_log_mean_ctrl = (
                nb_log_mean_gene + size_factor + batch_effects + covariate_effects
            )
            nb_log_disp_ctrl = nb_log_disp_gene.expand(nb_log_mean_ctrl.shape)

            # add perturbation effects to per-gene parameters
            nb_log_mean = nb_log_mean_ctrl + perturbations @ perturb_mean_lfc
            nb_log_dispersion = nb_log_disp_ctrl + perturbations @ perturb_disp_lfc

            with cell_plate:
                observations = tensor_dict.get(REGISTRY_KEYS.X_KEY)
                if self.likelihood == "lnnb":
                    return pyro.sample(
                        "obs",
                        LogNormalNegativeBinomial(
                            logits=nb_log_mean
                            - nb_log_dispersion
                            - multiplicative_noise**2 / 2,
                            total_count=nb_log_dispersion.exp(),
                            multiplicative_noise_scale=multiplicative_noise,
                            num_quad_points=8,
                        ),
                        obs=observations,
                    )
                elif self.likelihood == "nb":
                    return pyro.sample(
                        "obs",
                        dist.NegativeBinomial(
                            logits=nb_log_mean - nb_log_dispersion,
                            total_count=nb_log_dispersion.exp(),
                        ),
                        obs=observations,
                    )
                elif self.likelihood == "nb_mix":
                    logits = torch.stack(
                        (
                            nb_log_mean_ctrl - nb_log_disp_ctrl,
                            nb_log_mean - nb_log_dispersion,
                        ),
                        dim=-1,
                    )
                    total_counts = torch.stack(
                        broadcast_all(nb_log_disp_ctrl.exp(), nb_log_dispersion.exp()),
                        dim=-1,
                    )
                    mixture_dist = dist.Categorical(mixture_probs)
                    component_dist = dist.NegativeBinomial(
                        total_count=total_counts, logits=logits
                    )
                    mix_dist = dist.MixtureSameFamily(mixture_dist, component_dist)
                    obs = pyro.sample("obs", mix_dist, obs=observations)
                    return obs
