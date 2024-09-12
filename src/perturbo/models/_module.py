from typing import Iterable, Literal, Mapping, Optional  # noqa: UP035

import pyro
import pyro.distributions as dist
import torch
from pandas import DataFrame
from pyro import poutine
from pyro.infer import config_enumerate
from pyro.infer.autoguide import AutoGuideList, AutoNormal, init_to_mean, init_to_median
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
        likelihood: Literal["nb", "lnnb"] = "nb",
        effect_prior_dist: Literal["cauchy", "normal_mixture", "normal"] = "normal",
        n_factors=None,
        n_pert_factors=None,
        low_moi=True,
        dispersion_effects=False,
        merge_guides_mode: Literal["partial", "shared", "independent"] = "partial",
        prior_param_dict: Optional[Mapping[str, torch.Tensor]] = None,
        **module_kwargs,
    ) -> None:
        """
        PerTurboPyroModule: Pyro module underlying perturbo.

        Args:
        ---
        summary_stats: summary stats object from scvi model.
        gene_summary_stats: dict containing empirical gene mean values.
        guide_by_element: Binary array encoding which element(s) are targeted by each guide
        gene_by_element: Binary array encoding which element(s) may target each gene *a priori*
        likelihood: Observation likelihood, either NegativeBinomial or LogNormalNegativeBinomial.
        effect_prior_dist: Effect size prior, either Cauchy or NormalMixture ("soft" spike & slab)
        n_factors: Number of cell-specific factors ("probabilistic PCs")
        n_pert_factors: Number of perturbation-specific factors ("probabilistic contrastive PCs")
        low_moi: Is the screen low-MOI? (one guide per cell)
        dispersion_effects: Allow for different gene-level dispersion by batch?
        prior_params: dict containing hyperparameter names and tensors to set prior values
        merge_guides_mode: Should the model should pool information across guides targeting same element?
        """
        super().__init__()
        # set user-defined options for model behavior
        self.dispersion_effects = dispersion_effects
        self.likelihood = likelihood
        self.merge_guides_mode = merge_guides_mode
        self.lnnb_quad_points = 8
        self.n_factors = n_factors
        self.n_pert_factors = n_pert_factors
        self.effect_prior_dist = effect_prior_dist
        self.low_moi = low_moi

        # copy data summary stats
        self.n_cells = summary_stats.n_cells
        self.n_genes = summary_stats.n_vars
        self.n_perturbations = summary_stats.n_perturbations
        self.n_cont_covariates = 1  # include (inferred) size factor as covariate always
        self.multi_guide = ("n_targeted_elements" in summary_stats) and (merge_guides_mode != "independent")
        self.discrete_sites = ["perturbed"]

        # validate guide -> element mapping or use identity matrix as default
        if ("n_targeted_elements" in summary_stats) and (merge_guides_mode != "independent"):
            assert summary_stats.n_targeted_elements == guide_by_element.shape[1]
        else:
            guide_by_element = torch.eye(self.n_perturbations)
        self.n_elements = guide_by_element.shape[1]

        if "n_extra_continuous_covs" in summary_stats:
            self.n_cont_covariates += summary_stats.n_extra_continuous_covs
        # if "n_extra_categorical_covs" in summary_stats:
        #     self.n_cat_covariates = summary_stats.n_extra_categorical_covs
        #     self.n_cat_list = n_cats_per_cov
        # else:
        #     self.n_cat_covariates = 0
        #     self.n_cat_list = []
        self.n_batches = summary_stats.n_batch

        self._guide = AutoNormal(
            self.model,
            init_loc_fn=init_to_median,
            create_plates=self.create_plates,
        )

        self._guide = AutoGuideList(self.model, create_plates=self.create_plates)
        self._guide.append(
            AutoNormal(
                poutine.block(self.model, hide=["element_effects"] + self.discrete_sites),
                init_loc_fn=init_to_mean,
                init_scale=0.1,
            )
        )
        self._guide.append(
            AutoNormal(poutine.block(self.model, expose="element_effects"), init_loc_fn=init_to_median, init_scale=0.05)
        )

        ## register hyperparameters as buffers so they get automatically moved to GPU by scvi-tools
        self.register_buffer("guide_by_element", guide_by_element.to_sparse_coo())

        self.local_effects = gene_by_element is not None
        if self.local_effects:
            if gene_by_element.shape[1] != self.n_elements:
                raise ValueError("Number of inferred elements does not match gene_by_element matrix shape")

            self.register_buffer("element_by_gene_idx", gene_by_element.T.to_sparse_coo().indices())
            self.register_buffer(
                "guide_by_gene_idx",
                (guide_by_element @ gene_by_element.T).to_sparse_coo().indices(),
            )
        self.n_element_effects = self.element_by_gene_idx.shape[1] if self.local_effects else 8

        # intialize NegBin gene params to empirical mean estimates
        if gene_summary_stats is not None:
            e_x = gene_summary_stats["_gene_mean"].values
            epsilon = 1e-6
            self.register_buffer("gene_mean_prior_loc", torch.tensor(e_x + epsilon).log())
            self.register_buffer("gene_disp_prior_loc", torch.tensor(1.0))
        else:
            self.register_buffer("gene_mean_prior_loc", torch.tensor(0.0))
            self.register_buffer("gene_disp_prior_loc", torch.tensor(0.0))

        # set prior hyperparameters
        self.register_buffer("zero", torch.tensor(0.0))
        self.register_buffer("gene_mean_prior_scale", torch.tensor(3.0))
        self.register_buffer("gene_disp_prior_scale", torch.tensor(3.0))
        self.register_buffer("batch_effect_prior_scale", torch.tensor(3.0))
        self.register_buffer("element_effects_prior_scale", torch.tensor(0.01))
        self.register_buffer("covariate_prior_sigma", torch.tensor(3.0))
        self.register_buffer("covariate_disp_prior_sigma", torch.tensor(1.0))
        self.register_buffer("logit_efficacy_alpha", torch.tensor(5.0))
        self.register_buffer("logit_efficacy_beta", torch.tensor(1.0))
        self.register_buffer("has_guide_prior", torch.tensor(0.9))
        self.register_buffer(
            "spike_slab_prior_scales",
            torch.tensor([1 - self.element_effects_prior_scale, self.element_effects_prior_scale]),
        )
        self.register_buffer("spike_slab_prior_probs", torch.tensor([0.001, 0.999]))
        self.register_buffer("factor_element_prior_scale", torch.tensor(0.1))
        self.register_buffer("factor_gene_prior_scale", torch.tensor(0.1))
        self.register_buffer("noise_prior_rate", torch.tensor(2.0))

        # override hyperparameters with user-provided values from prior_param_dict
        if prior_param_dict is not None:
            for k, v in prior_param_dict.items():
                assert isinstance(v, torch.Tensor) and k in self.named_buffers
                assert v.shape == self.get_buffer(k).shape
                self.register_buffer(k, v)

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
        return (tensor_dict[REGISTRY_KEYS.INDICES_KEY].squeeze(),), tensor_dict

    def create_plates(self, idx, **tensor_dict):
        return (
            pyro.plate("Cells", self.n_cells, dim=-2, subsample=idx),
            pyro.plate("Guides", self.n_perturbations, dim=-2),
            pyro.plate("Elements", self.n_elements, dim=-2),
            pyro.plate("Batches", self.n_batches, dim=-2),
            pyro.plate("Genes", self.n_genes, dim=-1),
            pyro.plate("Covariates", self.n_cont_covariates, dim=-2),
            pyro.plate("Elements_sparse", self.n_element_effects, dim=-1),
            pyro.plate("Guides_sparse", self.n_perturbations, dim=-1),
            pyro.plate("Cell_factors", self.n_factors, dim=-3),
            pyro.plate("Pert_factors", self.n_pert_factors, dim=-3),
        )

    @config_enumerate
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
            cell_factor_plate,
            pert_factor_plate,
        ) = self.create_plates(idx)
        batch = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        size_factor = tensor_dict[REGISTRY_KEYS.SIZE_FACTOR_KEY]
        perturbations = tensor_dict[REGISTRY_KEYS.PERTURBATION_KEY]
        cont_covariates = tensor_dict[REGISTRY_KEYS.CONT_COVS_KEY]

        # Effect size priors
        if self.effect_prior_dist == "normal_mixture":
            comp_dist = dist.Normal(0.0, self.spike_slab_prior_scales)
            mix_dist = dist.Categorical(probs=self.spike_slab_prior_probs)
            effects_dist = dist.MixtureSameFamily(mix_dist, comp_dist)
        elif self.effect_prior_dist == "cauchy":
            effects_dist = dist.Cauchy(0.0, self.element_effects_prior_scale)
        elif self.effect_prior_dist == "normal":
            effects_dist = dist.Normal(0.0, self.element_effects_prior_scale)

        # Sample cis/trans effect sizes
        if self.local_effects:
            with element_effects_plate:
                element_local_effects_values = pyro.sample("element_effects", effects_dist)
                element_local_effects = torch.sparse_coo_tensor(
                    self.element_by_gene_idx,
                    element_local_effects_values,
                    size=(self.n_elements, self.n_genes),
                )
        else:
            with element_plate, gene_plate:
                element_local_effects = pyro.sample("element_effects", effects_dist)

        # Pool guide information based on user-specified strategy
        if self.merge_guides_mode != "shared":
            with guide_plate:
                guide_efficacy_values = pyro.sample(
                    "guide_efficacy",
                    dist.Beta(self.logit_efficacy_alpha, self.logit_efficacy_beta),
                )
            # fix weird broadcasting error
            guide_efficacy = guide_efficacy_values.expand(-1, self.n_elements) * self.guide_by_element
        else:
            guide_efficacy = self.guide_by_element
            # alternative: estimate efficacy for each guide--gene *cis* pair
            # with guide_effects_plate:
            #     guide_efficacy_values = pyro.sample(
            #         "guide_efficacy",
            #         dist.Beta(self.logit_efficacy_alpha, self.logit_efficacy_beta),
            #     )
            # guide_efficacy = torch.sparse_coo_tensor(
            #     self.guide_by_element_idx,
            #     guide_efficacy_values,
            #     size=(self.n_perturbations, self.n_elements),
            # )

        # Sample dense or factorized perturbation effects
        if self.n_pert_factors is None:
            total_perturbation_effect = guide_efficacy @ element_local_effects
        else:
            with pert_factor_plate, element_plate:
                pert_factors = pyro.sample(
                    "pert_factors",
                    dist.Laplace(0.0, self.factor_element_prior_scale),
                )
            with pert_factor_plate, gene_plate:
                pert_loadings = pyro.sample(
                    "pert_loadings",
                    dist.Laplace(0.0, self.factor_gene_prior_scale),
                )
            element_factor_effects = torch.einsum("fei,fjg->eg", pert_factors, pert_loadings)
            total_perturbation_effect = guide_efficacy @ (element_factor_effects + element_local_effects)

        # Sample cell-specific factors (linear unobserved confounders) if using
        if self.n_factors is not None:
            with cell_factor_plate, cell_plate:
                cell_factors = pyro.sample(
                    "cell_factors",
                    dist.Laplace(0.0, self.factor_element_prior_scale),
                )
            with cell_factor_plate, gene_plate:
                cell_loadings = pyro.sample(
                    "cell_loadings",
                    dist.Laplace(0.0, self.factor_gene_prior_scale),
                )
            cell_factor_effects = torch.einsum("fci,fjg->cg", cell_factors, cell_loadings)
        else:
            cell_factor_effects = 0

        with gene_plate:
            # Sample parameters of baseline gene expression distribution
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

            # Calculate final expression distribution parameters for unperturbed cells
            nb_log_mean_ctrl = (
                gene_base_log_mean + size_factor + batch_effects + covariate_effects + cell_factor_effects
            )
            with cell_plate:
                if self.low_moi:
                    has_guide_prob = self.has_guide_prior
                else:
                    has_guide_prob = perturbations @ guide_efficacy_values
                has_effect = pyro.sample("perturbed", dist.Bernoulli(has_guide_prob))

            if not self.dispersion_effects:
                nb_log_dispersion = gene_log_dispersion
            else:
                nb_log_dispersion = gene_log_dispersion + batch_disp_effects + covariate_disp_effects

            # Add perturbation effects to gene expression parameters
            nb_log_mean = nb_log_mean_ctrl + (perturbations @ total_perturbation_effect) * has_effect

            # Sample read counts from distributions
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
                    raise NotImplementedError(f"'{self.likelihood}' likelihood not implemented")

    @property
    def guide(self):
        return self._guide
