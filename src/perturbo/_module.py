from collections.abc import Iterable
from typing import Literal, Optional

import pyro
import pyro.distributions as dist
import torch
from pandas import DataFrame
from pyro import poutine
from pyro.distributions import constraints
from pyro.infer import config_enumerate
from pyro.infer.autoguide import AutoGuideList, AutoNormal, init_to_mean, init_to_median
from pyro.ops.indexing import Vindex
from scvi.module.base import PyroBaseModuleClass

from ._constants import REGISTRY_KEYS


class LogNormalNegativeBinomial(dist.LogNormalNegativeBinomial):
    def sample(self, sample_shape=torch.Size()):
        normals = (
            dist.Normal(0, self.multiplicative_noise_scale).expand(self.batch_shape).sample(sample_shape=sample_shape)
        )
        return dist.NegativeBinomial(total_count=self.total_count, logits=self.logits + normals).sample()


class PerTurboPyroModule(PyroBaseModuleClass):
    def __init__(
        self,
        summary_stats,
        gene_summary_stats: Optional[DataFrame] = None,
        guide_by_element: Optional[torch.Tensor] = None,
        gene_by_element: Optional[torch.Tensor] = None,
        likelihood: Optional[str] = "nb",
        effect_prior_dist: Literal["cauchy", "normal_mixture"] = "cauchy",
        n_factors=None,
        dispersion_effects=False,
        merge_guides=False,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        **module_kwargs,
    ) -> None:
        super().__init__()
        # set user-defined options for model behavior
        self.dispersion_effects = dispersion_effects
        self.likelihood = likelihood
        self.merge_guides = merge_guides
        self.lnnb_quad_points = 8
        self.n_factors = n_factors
        self.effect_prior_dist = effect_prior_dist

        # copy data summary stats
        self.n_cells = summary_stats.n_cells
        self.n_genes = summary_stats.n_vars
        self.n_perturbations = summary_stats.n_perturbations
        self.n_cont_covariates = 1  # include (inferred) size factor by default
        self.multi_guide = "n_targeted_elements" in summary_stats
        if self.multi_guide:
            assert summary_stats.n_targeted_elements == guide_by_element.shape[1]
        else:
            guide_by_element = torch.eye(self.n_perturbations)
        self.n_elements = guide_by_element.shape[1]

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
            poutine.block(self.model, hide="has_effect"),
            init_loc_fn=init_to_median,
            create_plates=self.create_plates,
        )

        # self._guide = AutoGuideList(self.model, create_plates=self.create_plates)
        # self._guide.append(
        #     AutoNormal(poutine.block(self.model, hide="element_effects"), init_loc_fn=init_to_mean, init_scale=0.1)
        # )
        # self._guide.append(self.local_guide)
        # self._guide.append(
        #     AutoNormal(poutine.block(self.model, expose="element_effects"), init_loc_fn=init_to_median, init_scale=0.05)
        # )

        ## register hyperparameters as buffers so they get automatically moved to GPU by scvi-tools

        self.local_effects = gene_by_element is not None

        # self.register_buffer("guide_by_element", guide_by_element.to_sparse_coo())
        # # else:
        self.register_buffer("guide_by_element", guide_by_element.to_dense())

        if self.local_effects:
            if gene_by_element.shape[1] != self.n_elements:
                raise ValueError("Number of inferred elements does not match gene_by_element matrix shape")

            self.register_buffer("element_by_gene_idx", gene_by_element.T.to_sparse_coo().indices().to_dense())
            self.register_buffer(
                "guide_by_gene_idx",
                (guide_by_element @ gene_by_element.T).to_sparse_coo().indices(),
            )
        self.n_element_effects = self.element_by_gene_idx.shape[1] if self.local_effects else 8

        # intialize NegBin gene params to empirical mean estimates
        if gene_summary_stats is not None:
            e_x = gene_summary_stats["_gene_mean"].values
            # v_x = gene_summary_stats["_gene_variance"].values
            epsilon = 1e-12
            gene_logits = torch.tensor(e_x + epsilon).log()
            self.register_buffer("gene_mean_prior_loc", gene_logits)
            self.register_buffer("gene_disp_prior_loc", torch.tensor(1.0))
        else:
            self.register_buffer("gene_mean_prior_loc", torch.tensor(0.0))
            self.register_buffer("gene_disp_prior_loc", torch.tensor(0.0))

        self.register_buffer("zero", torch.tensor(0.0))
        self.register_buffer("gene_mean_prior_scale", torch.tensor(3.0))
        self.register_buffer("gene_disp_prior_scale", torch.tensor(3.0))
        self.register_buffer("batch_effect_prior_scale", torch.tensor(3.0))
        self.register_buffer("element_effects_prior_scale", torch.tensor(0.01))
        self.register_buffer("covariate_prior_sigma", torch.tensor(3.0))
        self.register_buffer("covariate_disp_prior_sigma", torch.tensor(1.0))
        self.register_buffer("logit_efficacy_alpha", torch.tensor(5.0))
        self.register_buffer("logit_efficacy_beta", torch.tensor(1.0))

        self.register_buffer("spike_slab_prior_scales", torch.tensor([0.05, 1.0]))
        self.register_buffer("spike_slab_prior_probs", torch.tensor([0.9, 0.1]))

        if self.n_factors is not None:
            self.register_buffer("factor_element_prior_scale", torch.tensor(0.0001))
            self.register_buffer("factor_gene_prior_scale", torch.tensor(0.0001))

        if self.likelihood == "lnnb":
            self.register_buffer("noise_prior_rate", torch.tensor(2.0, requires_grad=False))

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        fit_size_factor_covariate = False

        if fit_size_factor_covariate:
            # tack on size factor after the other continuous covariates
            size_factor = tensor_dict[REGISTRY_KEYS.SIZE_FACTOR_KEY]
        else:
            size_factor = torch.zeros_like(tensor_dict[REGISTRY_KEYS.SIZE_FACTOR_KEY])
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
            pyro.plate("genes", self.n_genes, dim=-1),
            pyro.plate("covariates", self.n_cont_covariates, dim=-2),
            pyro.plate("elements_sparse", self.n_element_effects, dim=-1),
            pyro.plate("guides_sparse", self.n_perturbations, dim=-1),
            pyro.plate("factors_1", self.n_factors, dim=-1),
            pyro.plate("factors_2", self.n_factors, dim=-2),
        )

    def local_guide(self, idx, **tensor_dict):
        # Create learnable parameters.
        if self.local_effects:
            mix_probs = pyro.param(
                "element_effect_prob",
                self.spike_slab_prior_probs.expand((self.n_element_effects, -1)),
                constraint=constraints.simplex,
            )
            effect_loc = pyro.param("element_effect_loc", torch.zeros((self.n_element_effects, 2)))
            effect_scale = pyro.param(
                "element_effect_scale",
                self.spike_slab_prior_scales.expand((self.n_element_effects, -1)),
                constraint=constraints.positive,
            )
            with self.guide.plates["elements_sparse"]:
                pyro.sample(
                    "element_effects",
                    dist.MixtureSameFamily(dist.Categorical(mix_probs), dist.Normal(effect_loc, effect_scale)),
                )
        else:
            mix_probs = pyro.param(
                "element_effect_prob",
                self.spike_slab_prior_probs.expand((self.n_elements, self.n_genes, -1)),
                constraint=constraints.simplex,
            )
            effect_loc = pyro.param("element_effect_loc", torch.zeros((self.n_elements, self.n_genes, 2)))
            effect_scale = pyro.param(
                "element_effect_scale",
                self.spike_slab_prior_scales.expand((self.n_elements, self.n_genes, -1)),
                constraint=constraints.positive,
            )
            with self.guide.plates["elements"], self.guide.plates["genes"]:
                pyro.sample(
                    "element_effects",
                    dist.MixtureSameFamily(dist.Categorical(mix_probs), dist.Normal(effect_loc, effect_scale)),
                )

    def model(self, idx, **tensor_dict):
        pyro.module("perturbo", self)
        (
            cell_plate,
            guide_plate,
            element_plate,
            batch_plate,
            gene_plate,
            cont_covariate_plate,
            element_effects_plate,  # sparse mode
            guide_effects_plate,  # sparse mode
            factor_plate_1,
            factor_plate_2,
        ) = self.create_plates(idx)
        batch = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        size_factor = tensor_dict[REGISTRY_KEYS.SIZE_FACTOR_KEY]
        perturbations = tensor_dict[REGISTRY_KEYS.PERTURBATION_KEY]
        cont_covariates = tensor_dict[REGISTRY_KEYS.CONT_COVS_KEY]

        # comp_dist = dist.Normal(0.0, self.spike_slab_prior_scales)

        # if self.effect_prior_dist == "normal_mixture":
        # effects_dist = dist.MixtureSameFamily(mix_dist, comp_dist)

        # else:
        # effects_dist = dist.Cauchy(0.0, self.element_effects_prior_scale)

        # if self.local_effects:
        #     with element_effects_plate:
        #         element_local_effects_values = pyro.sample("element_effects", effects_dist)
        #         batch_dims = element_local_effects_values.shape[:-1]
        #         element_local_effects = torch.sparse_coo_tensor(
        #             self.element_by_gene_idx,
        #             element_local_effects_values,
        #             size=batch_dims + torch.Size((self.n_elements, self.n_genes)),
        #         )
        # else:

        with element_plate, gene_plate:
            # print(el_idx[..., None].shape, gn_idx)
            mix_dist = dist.Categorical(probs=self.spike_slab_prior_probs)
            has_effect = pyro.sample("has_effect", mix_dist, infer={"enumerate": "parallel"})
            print("has_effect", has_effect.shape)
            mix_scales = Vindex(self.spike_slab_prior_scales)
            print("mix_scales", mix_scales.shape)
            beta = pyro.sample("element_effects", dist.Normal(0.0, mix_scales))
            element_local_effects = beta
            # element_local_effects = beta

            # raise Exception(has_effect.max())
            # element_local_effects = pyro.sample("element_effects", effects_dist)

        # estimate a single efficacy value per guide
        # if self.multi_guide and not self.merge_guides:
        #     with guide_plate:
        #         guide_efficacy_values = pyro.sample(
        #             "guide_efficacy",
        #             dist.Beta(self.logit_efficacy_alpha, self.logit_efficacy_beta),
        #         )
        #     # fix weird broadcasting error
        #     guide_efficacy = guide_efficacy_values.expand(-1, self.n_elements) * self.guide_by_element

        #     # alternative: estimate efficacy for each guide--gene *cis* pair
        #     # with guide_effects_plate:
        #     #     guide_efficacy_values = pyro.sample(
        #     #         "guide_efficacy",
        #     #         dist.Beta(self.logit_efficacy_alpha, self.logit_efficacy_beta),
        #     #     )
        #     # guide_efficacy = torch.sparse_coo_tensor(
        #     #     self.guide_by_element_idx,
        #     #     guide_efficacy_values,
        #     #     size=(self.n_perturbations, self.n_elements),
        #     # )
        # else:
        guide_efficacy = self.guide_by_element

        if self.n_factors is not None:
            with element_plate, factor_plate_1:
                element_x_factor = pyro.sample(
                    "element_factors",
                    dist.Normal(0.0, self.factor_element_prior_scale),
                )
            with factor_plate_2, gene_plate:
                factor_x_gene = pyro.sample(
                    "factor_gene_loadings",
                    dist.Cauchy(0.0, self.factor_gene_prior_scale),
                )
            element_factor_effects = element_x_factor @ factor_x_gene
            total_perturbation_effect = guide_efficacy @ (element_factor_effects + element_local_effects)
        else:
            total_perturbation_effect = torch.matmul(guide_efficacy, element_local_effects)

        with gene_plate:
            # mean and dispersion of each gene's expression
            gene_base_log_mean = pyro.sample(
                "log_gene_mean",
                dist.Normal(self.gene_mean_prior_loc, self.gene_mean_prior_scale),
            )
            gene_log_dispersion = pyro.sample(
                "log_gene_dispersion",
                dist.Normal(self.gene_disp_prior_loc, self.gene_disp_prior_scale),
            )

            if self.likelihood == "lnnb":
                # additional noise for LogNormalNegativeBinomial likelihood
                multiplicative_noise = pyro.sample("multiplicative_noise", dist.Exponential(self.noise_prior_rate))
                # multiplicative_noise = 1 / self.noise_prior_rate

            with batch_plate:
                # batch effects: n_batches x n_genes
                batch_effect_size = pyro.sample("batch_effect", dist.Normal(0.0, self.batch_effect_prior_scale))
                batch_effects = batch_effect_size[batch.squeeze(), ...]
                if self.dispersion_effects:
                    batch_disp_effect_size = pyro.sample(
                        "batch_disp_effect",
                        dist.Normal(0.0, self.batch_effect_prior_scale),
                    )
                    batch_disp_effects = batch_disp_effect_size[batch.squeeze(), ...]

            with cont_covariate_plate:
                # covariate effects: n_cont_covariates x n_genes
                cont_covariate_effect_size = pyro.sample(
                    "cont_covariate_effect",
                    dist.Normal(0.0, self.covariate_prior_sigma),
                )
                covariate_effects = cont_covariates @ cont_covariate_effect_size
                if self.dispersion_effects:
                    cont_covariate_disp_effect_size = pyro.sample(
                        "cont_covariate_disp_effect",
                        dist.Normal(0.0, self.covariate_disp_prior_sigma),
                    )
                    covariate_disp_effects = cont_covariates @ cont_covariate_disp_effect_size

            # calculate overall parameter values for unperturbed cells
            nb_log_mean_ctrl = gene_base_log_mean + size_factor + batch_effects + covariate_effects

            if not self.dispersion_effects:
                nb_log_dispersion = gene_log_dispersion
            else:
                nb_log_dispersion = gene_log_dispersion + batch_disp_effects + covariate_disp_effects

            # add perturbation effects to per-gene parameters
            nb_log_mean = nb_log_mean_ctrl + perturbations @ total_perturbation_effect

            with cell_plate:
                observations = tensor_dict.get(REGISTRY_KEYS.X_KEY)
                if self.likelihood == "lnnb":
                    return pyro.sample(
                        "obs",
                        LogNormalNegativeBinomial(
                            logits=nb_log_mean - nb_log_dispersion - multiplicative_noise**2 / 2,
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
                else:
                    raise NotImplementedError(f"{self.likelihood} likelihood not implemented")

    @property
    def guide(self):
        return self._guide
