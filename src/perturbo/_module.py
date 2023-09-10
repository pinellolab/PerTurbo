from collections.abc import Iterable
from typing import Optional

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
        guide_by_element: Optional[torch.Tensor] = None,
        var_by_element: Optional[torch.Tensor] = None,
        likelihood: str = "lnnb",
        n_factors=None,
        fit_dispersion=False,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        **module_kwargs,
    ) -> None:
        super().__init__()
        # set user-defined options for model behavior
        self.fit_dispersion = fit_dispersion
        self.likelihood = likelihood
        self.lnnb_quad_points = 8

        # copy data summary stats
        self.n_cells = summary_stats.n_cells
        self.n_vars = summary_stats.n_vars
        self.n_perturbations = summary_stats.n_perturbations
        self.n_cont_covariates = 1  # include (inferred) size factor by default
        self.has_elements = "n_targeted_elements" in summary_stats
        if self.has_elements:
            assert summary_stats.n_targeted_elements == guide_by_element.shape[1]
            self.n_elements = guide_by_element.shape[1]
        else:
            self.n_elements = self.n_perturbations
        if "n_extra_continuous_covs" in summary_stats:
            self.n_cont_covariates += summary_stats.n_extra_continuous_covs
        if "n_extra_categorical_covs" in summary_stats:
            self.n_cat_covariates = summary_stats.n_extra_categorical_covs
            self.n_cat_list = n_cats_per_cov
        else:
            self.n_cat_covariates = 0
            self.n_cat_list = []
        self.n_batches = summary_stats.n_batch

        self._guide = AutoNormal(
            self.model,
            init_loc_fn=init_to_mean,
            create_plates=self.create_plates,
            init_scale=0.2,
        )

        ## register hyperparameters as buffers so they get automatically moved to GPU by scvi-tools
        if guide_by_element is not None:
            self.register_buffer("guide_by_element", guide_by_element.to_sparse_coo())
        if var_by_element is not None:
            self.register_buffer(
                "element_by_var_idx", var_by_element.T.to_sparse_coo().indices()
            )
            self.register_buffer(
                "guide_by_var_idx",
                (guide_by_element @ var_by_element.T).to_sparse_coo().indices(),
            )
            self.n_element_effects = self.element_by_var_idx.shape[1]
            self.n_guide_effects = self.guide_by_var_idx.shape[1]
        # self.register_buffer("guide_by_var", (guide_by_element @ var_by_element.T).to_sparse())
        self.register_buffer("gene_mean_prior_scale", torch.tensor(3.0))
        self.register_buffer("gene_disp_prior_scale", torch.tensor(1.0))
        self.register_buffer("batch_effect_prior_scale", torch.tensor(1.0))
        self.register_buffer("element_mean_lfc_prior_scale", torch.tensor(0.05))
        self.register_buffer("element_disp_lfc_prior_scale", torch.tensor(0.05))
        self.register_buffer("pooling_prior_loc", torch.tensor(-3.0))
        self.register_buffer("pooling_prior_scale", torch.tensor(1.0))
        self.register_buffer("covariate_prior_sigma", torch.tensor(1.0))
        self.register_buffer("covariate_prior_sigma", torch.tensor(0.1))

        if self.likelihood == "lnnb":
            self.register_buffer(
                "noise_prior_rate", torch.tensor(10.0, requires_grad=False)
            )

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
            pyro.plate("element_effects", self.n_element_effects, dim=-1),
            pyro.plate("guide_effects", self.n_guide_effects, dim=-1),
        )

    def model(self, idx, **tensor_dict):
        pyro.module("perturbo", self)
        (
            cell_plate,
            perturbation_plate,
            element_plate,
            batch_plate,
            feature_plate,
            cont_covariate_plate,
            element_effects_plate,  # sparse mode
            guide_effects_plate,  # sparse mode
        ) = self.create_plates(idx)
        batch = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        size_factor = tensor_dict[REGISTRY_KEYS.SIZE_FACTOR_KEY]
        perturbations = tensor_dict[REGISTRY_KEYS.PERTURBATION_KEY]
        cont_covariates = tensor_dict[REGISTRY_KEYS.CONT_COVS_KEY]

        # Estimate strength of gRNA effect sharing
        element_mean_pooling = pyro.sample(
            "element_mean_pooling",
            dist.LogNormal(self.pooling_prior_loc, self.pooling_prior_scale),
        )

        with element_effects_plate:
            element_effects_values = pyro.sample(
                "element_mean_lfc",
                dist.Cauchy(0.0, self.element_mean_lfc_prior_scale),
            )
            element_mean_lfc = torch.sparse_coo_tensor(
                self.element_by_var_idx, element_effects_values
            )

        with guide_effects_plate:
            guide_effects_values = pyro.sample(
                "perturb_mean_lfc",
                dist.Normal(0.0, element_mean_pooling),
            )
            perturb_mean_lfc = (
                torch.sparse_coo_tensor(self.guide_by_var_idx, guide_effects_values)
                + self.guide_by_element @ element_mean_lfc
            )

        with feature_plate:
            # mean and dispersion of each gene's expression
            nb_log_mean_gene = pyro.sample(
                "log_feature_mean", dist.Normal(0.0, self.gene_mean_prior_scale)
            )
            nb_log_disp_gene = pyro.sample(
                "log_feature_dispersion", dist.Normal(0.0, self.gene_disp_prior_scale)
            )

            if self.likelihood == "lnnb":
                # additional noise for LogNormalNegativeBinomial likelihood
                multiplicative_noise = pyro.sample(
                    "multiplicative_noise", dist.Exponential(self.noise_prior_rate)
                )

            with batch_plate:
                # batch effects: n_batches x n_vars
                batch_effect_size = pyro.sample(
                    "batch_effect", dist.Normal(0.0, self.batch_effect_prior_scale)
                )
                batch_effects = batch_effect_size[batch.squeeze(), ...]

            with cont_covariate_plate:
                # covariate effects: n_cont_covariates x n_vars
                cont_covariate_effect_size = pyro.sample(
                    "cont_covariate_effect",
                    dist.Normal(0.0, self.covariate_prior_sigma),
                )
                covariate_effects = cont_covariates @ cont_covariate_effect_size

            # with element_plate:
            #     # element effects: n_elements x n_vars
            #     element_mean_lfc = pyro.sample(
            #         "element_mean_lfc",
            #         dist.Cauchy(0.0, self.element_mean_lfc_prior_scale),
            #     )
            #     if self.fit_dispersion:
            #         element_disp_lfc = pyro.sample(
            #             "element_disp_lfc",
            #             dist.Cauchy(0.0, self.element_disp_lfc_prior_scale),
            #         )

            # if self.has_elements:
            #     with perturbation_plate:
            #         # perturbation effects: n_perturbations x n_vars
            #         perturb_mean_lfc = (
            #             pyro.sample(
            #                 "perturb_mean_lfc", dist.Normal(0.0, element_mean_pooling)
            #             )
            #             + self.guide_by_element @ element_mean_lfc
            #         )
            #         if self.fit_dispersion:
            #             perturb_disp_lfc = pyro.sample(
            #                 "perturb_disp_lfc",
            #                 dist.Normal(
            #                     self.guide_by_element @ element_disp_lfc,
            #                     self.element_disp_pooling,
            #                 ),
            #             )
            # else:
            #     perturb_mean_lfc = element_mean_lfc
            #     if self.fit_dispersion:
            #         perturb_disp_lfc = element_disp_lfc

            # calculate overall parameter values for unperturbed cells
            nb_log_mean_ctrl = (
                nb_log_mean_gene + size_factor + batch_effects + covariate_effects
            )
            nb_log_disp_ctrl = nb_log_disp_gene.expand(nb_log_mean_ctrl.shape)

            # add perturbation effects to per-gene parameters
            nb_log_mean = nb_log_mean_ctrl + perturbations @ perturb_mean_lfc
            nb_log_dispersion = nb_log_disp_ctrl
            # if self.fit_dispersion:
            #     nb_log_dispersion += perturbations @ perturb_disp_lfc

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
                            num_quad_points=self.lnnb_quad_points,
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
        loc, scale = self.guide._get_loc_and_scale("element_mean_lfc")
        if len(loc.shape) == 1:
            loc = torch.sparse_coo_tensor(
                self.element_by_var_idx, loc.clone()
            ).to_dense()
            scale = torch.sparse_coo_tensor(
                self.element_by_var_idx, scale.clone()
            ).to_dense()
        return (loc.detach().cpu().numpy(), scale.detach().cpu().numpy())

    def get_perturbation_effects(self):
        """Return the perturbation effects on each variable's mean and variance."""
        if not self.has_elements:
            return self.get_element_effects()
        loc1, scale1 = self.get_element_effects()
        loc2, scale2 = self.guide._get_loc_and_scale("perturb_mean_lfc")
        if len(loc2.shape) == 1:
            loc2 = torch.sparse_coo_tensor(self.element_by_var_idx, loc2.clone())
            scale2 = torch.sparse_coo_tensor(self.element_by_var_idx, scale2.clone())
        loc = self.guide_by_element @ loc1 + loc2
        scale = ((self.guide_by_element @ scale1) ** 2 + scale2**2).sqrt()
        return (loc.detach().cpu().numpy(), scale.detach().cpu().numpy())
