from typing import Iterable, Optional

import pyro
import pyro.distributions as dist
import torch
from pyro.infer.autoguide import AutoNormal, init_to_mean
from scvi.module.base import PyroBaseModuleClass

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


class PerTurboPyroModule(PyroBaseModuleClass):
    def __init__(
        self,
        summary_stats,
        guide_by_element: torch.Tensor,
        likelihood="lnnb",
        factors=None,
        fit_dispersion=False,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        **module_kwargs,
    ) -> None:
        super().__init__()
        self.factors = factors
        self.n_cells = summary_stats.n_cells
        self.n_vars = summary_stats.n_vars
        self.n_perturbations = summary_stats.n_perturbations
        self.n_cont_covariates = 1  # include (inferred) size factor by default
        self.fit_dispersion = fit_dispersion
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
        self._guide = AutoNormal(
            self.model, init_loc_fn=init_to_mean, create_plates=self.create_plates, init_scale=0.2
        )

        self.n_batches = summary_stats.n_batch
        self.likelihood = likelihood

    # required to override broken method in PyroBaseModuleClass
    def on_load(self, model):
        pass

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
            pyro.plate("guides", self.n_perturbations, dim=-2),
            pyro.plate("elements", self.n_elements, dim=-2),
            pyro.plate("batches", self.n_batches, dim=-2),
            pyro.plate("vars", self.n_vars, dim=-1),
            pyro.plate("cont_covariates", self.n_cont_covariates, dim=-2),
        )

    def model(self, idx, **tensor_dict):
        pyro.module("perturbo", self)
        (
            cell_plate,
            perturbation_plate,
            element_plate,
            batch_plate,
            feature_plate,
            cont_cov_plate,
        ) = self.create_plates(idx)
        batch = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        size_factor = tensor_dict[REGISTRY_KEYS.SIZE_FACTOR_KEY]
        perturbations = tensor_dict[REGISTRY_KEYS.PERTURBATION_KEY]
        cont_covariates = tensor_dict[REGISTRY_KEYS.CONT_COVS_KEY]

        # Estimate strength of gRNA effect sharing
        pooling_prior_loc = torch.tensor(-3.0)
        pooling_prior_scale = torch.tensor(1.0)
        log_pooling = pyro.sample(
            "log_pooling", dist.Normal(pooling_prior_loc, pooling_prior_scale)
        )

        with feature_plate:
            # mean and dispersion of each gene's expression
            gene_mean_prior_scale = torch.tensor(3.0)
            gene_disp_prior_scale = torch.tensor(1.0)
            nb_log_mean_gene = pyro.sample(
                "log_feature_mean", dist.Normal(0.0, gene_mean_prior_scale)
            )
            nb_log_disp_gene = pyro.sample(
                "log_feature_dispersion", dist.Normal(0.0, gene_disp_prior_scale)
            )

            if self.likelihood == "lnnb":
                # additional noise for LogNormalNegativeBinomial likelihood
                noise_prior_rate = torch.tensor(10.0)
                multiplicative_noise = pyro.sample(
                    "multiplicative_noise", dist.Exponential(noise_prior_rate)
                )

            with batch_plate:
                # batch effects: n_batches x n_vars
                batch_effect_prior_scale = torch.tensor(1.0)
                batch_effect_size = pyro.sample(
                    "batch_effect", dist.Normal(0.0, batch_effect_prior_scale)
                )
                batch_effects = batch_effect_size[batch.squeeze(), ...]

            cov_prior_sigma = torch.tensor(1.0)
            with cont_cov_plate:
                # covariate effects: n_cont_covariates x n_vars
                cont_cov_effect_size = pyro.sample(
                    "cont_cov_effect", dist.Normal(0.0, cov_prior_sigma)
                )
                covariate_effects = cont_covariates @ cont_cov_effect_size

            with element_plate:
                # element effects: n_elements x n_vars
                element_mean_lfc_prior_scale = torch.tensor(0.05)
                element_disp_lfc_prior_scale = torch.tensor(0.05)
                element_mean_lfc = pyro.sample(
                    "element_mean_lfc", dist.Cauchy(0.0, element_mean_lfc_prior_scale)
                )
                if self.fit_dispersion:
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
                if self.fit_dispersion:
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
            nb_log_dispersion = nb_log_disp_ctrl
            if self.fit_dispersion:
                nb_log_dispersion += perturbations @ perturb_disp_lfc

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

    @property
    def guide(self):
        return self._guide


    def get_element_effects(self):
        """Return the perturbation effects on each variable's mean and variance."""
        element_mu = self.guide.quantiles([0.5])["element_mean_lfc"].squeeze(0)
        element_mu_plus_sigma = self.guide.quantiles([0.6827])["element_mean_lfc"].squeeze(0)
        element_sigma = element_mu_plus_sigma-element_mu

        return (element_mu.detach().cpu().numpy(), element_sigma.detach().cpu().numpy())

    def get_perturbation_effects(self):
        """Return the perturbation effects on each variable's mean and variance."""
        q_mu = self.guide.quantiles([0.5])["perturb_mean_lfc"].squeeze(0)
        q_mu_plus_sigma = self.guide.quantiles([0.6827])["perturb_mean_lfc"].squeeze(0)
        q_sigma = q_mu_plus_sigma-q_mu


        return (q_mu.numpy(), q_sigma.numpy())
