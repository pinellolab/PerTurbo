from collections import defaultdict
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Literal

import pyro
import pyro.distributions as dist
import torch
from pandas import DataFrame
from pyro import poutine
from pyro.infer import config_enumerate
from pyro.infer.autoguide import AutoDelta, AutoGuideList, AutoNormal, init_to_median
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
        gene_summary_stats: DataFrame | None = None,
        guide_by_element: torch.Tensor | None = None,
        gene_by_element: torch.Tensor | None = None,
        likelihood: Literal["nb", "lnnb"] = "nb",
        effect_prior_dist: Literal["cauchy", "normal_mixture", "normal", "laplace"] = "normal",
        n_factors: int | None = None,
        n_pert_factors: int | None = None,
        use_interactions: bool = False,
        efficiency_mode: Literal["mixture", "scaled"] = "scaled",
        dispersion_effects: bool = False,
        use_crispr_factor: bool = False,
        merge_guides_mode: Literal["partial", "shared"] = "partial",
        prior_param_dict: Mapping[str, torch.Tensor] | None = None,
        **module_kwargs,
    ) -> None:
        """
        Pyro module underlying perturbo.

        Parameters
        ----------
        summary_stats:
            summary stats object from scvi model.
        gene_summary_stats:
            dict containing observed gene expression mean/variance.
        guide_by_element:
            Binary array encoding which element(s) are targeted by each guide.
        gene_by_element:
            Binary array encoding which element(s) may target each gene *a priori*.
        likelihood:
            Observation likelihood, either NegativeBinomial ("nb") or LogNormalNegativeBinomial ("lnnb").
        effect_prior_dist:
            Effect size prior, either Cauchy or NormalMixture ("soft" spike & slab)
        n_factors:
            Number of cell-specific latent factors
        n_pert_factors:
            Number of perturbation-specific latent factors
        use_interactions:
            If using cell factors, allow interactions between perturbations and cell factors
        efficiency_mode:
            Guide efficiency is fraction of cells perturbed ("mixture") or linear scaling of effect size ("scaled").
            "Mixture" mode currently requires (at most) one guide per cell.
        dispersion_effects:
            Allow for different gene-level dispersion by batch?
        prior_params:
            dict containing hyperparameter names and tensors to set prior values
        merge_guides_mode:
            "shared" treats ("partial") across guides targeting same element?
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
        self.use_interactions = use_interactions
        self.efficiency_mode = efficiency_mode
        self.local_effects = gene_by_element is not None
        self.use_crispr_factor = use_crispr_factor

        # copy data summary stats
        self.n_cells = summary_stats.n_cells
        self.n_genes = summary_stats.n_vars
        self.n_perturbations = summary_stats.n_perturbations
        self.n_cont_covariates = 1  # include (inferred) size factor as covariate always

        self.discrete_sites = []
        if efficiency_mode == "mixture":
            self.discrete_sites.append("perturbed")

        # validate guide -> element mapping or use identity matrix as default
        if "n_targeted_elements" not in summary_stats:
            guide_by_element = torch.eye(self.n_perturbations)
        else:
            assert summary_stats.n_targeted_elements == guide_by_element.shape[1]
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

        self.delta_sites = []
        # self.delta_sites = ["cell_loadings"]
        # self.delta_sites = ["cell_factors", "cell_loadings"]
        if efficiency_mode == "ps":
            self.delta_sites.append("perturbed")

        self._guide = AutoGuideList(self.model, create_plates=self.create_plates)
        self._guide.append(
            AutoNormal(
                poutine.block(self.model, hide=["element_effects"] + self.delta_sites + self.discrete_sites),
                init_loc_fn=lambda x: init_to_median(x, num_samples=100),
                init_scale=0.1,
            )
        )
        self._guide.append(
            AutoNormal(
                poutine.block(self.model, expose="element_effects"),
                init_loc_fn=lambda x: init_to_median(x, num_samples=100),
                init_scale=0.05,
            )
        )
        if self.delta_sites:
            self._guide.append(
                AutoDelta(
                    poutine.block(self.model, expose=self.delta_sites),
                    init_loc_fn=lambda x: init_to_median(x, num_samples=100),
                )
            )

        ## register hyperparameters as buffers so they get automatically moved to GPU by scvi-tools

        # guide_by_element encoding
        self.register_buffer("guide_by_element", guide_by_element)

        if self.local_effects:
            assert gene_by_element.shape[1] == self.n_elements
            self.register_buffer("element_by_gene", gene_by_element.T)
            self.register_buffer("element_by_gene_idx", gene_by_element.T.to_sparse_coo().indices())
            # self.register_buffer("guide_by_gene_idx", (guide_by_element @ gene_by_element.T).to_sparse_coo().indices())
        self.n_element_effects = self.element_by_gene_idx.shape[1] if self.local_effects else 1

        # global hyperparams
        self.register_buffer("zero", torch.tensor(0.0))
        self.register_buffer("one", torch.tensor(1.0))

        # per-gene hyperparams
        if gene_summary_stats is not None:
            e_x = gene_summary_stats["_gene_mean"].values
            epsilon = 1e-6
            self.register_buffer("gene_mean_prior_loc", torch.tensor(e_x + epsilon).log())
            self.register_buffer("gene_disp_prior_loc", torch.tensor(1.0))
        else:
            self.register_buffer("gene_mean_prior_loc", torch.tensor(0.0))
            self.register_buffer("gene_disp_prior_loc", torch.tensor(0.0))

        self.register_buffer("gene_mean_prior_scale", torch.tensor(3.0))
        self.register_buffer("gene_disp_prior_scale", torch.tensor(3.0))

        # batch/covariate hyperparams
        self.register_buffer("batch_effect_prior_scale", torch.tensor(3.0))
        self.register_buffer("covariate_prior_sigma", torch.tensor(3.0))
        self.register_buffer("covariate_disp_prior_sigma", torch.tensor(1.0))

        # efficiency hyperparams
        self.register_buffer("logit_efficacy_alpha", torch.tensor(5.0))
        self.register_buffer("logit_efficacy_beta", torch.tensor(1.0))
        self.register_buffer("has_guide_prior", torch.tensor(0.9))

        ##  element effect size hyperparams

        # Normal/Laplace/Cauchy prior
        element_prior_scale_default = 1.0
        element_prior_scales = {"cauchy": 0.01, "laplace": 0.1}
        model_element_prior_scale = element_prior_scales.get(effect_prior_dist, element_prior_scale_default)
        self.register_buffer("element_effects_prior_scale", torch.tensor(model_element_prior_scale))

        # normal mixture prior hyperparams
        self.register_buffer("spike_slab_prior_scales", torch.tensor([1.0, 0.1]))
        self.register_buffer("spike_slab_prior_probs", torch.tensor([0.01, 0.99]))

        # (contrastive) factor model hyperparams
        self.register_buffer("cell_factor_prior_scale", torch.tensor(1.0))
        self.register_buffer("cell_loading_prior_scale", torch.tensor(0.1))
        self.register_buffer("pert_factor_prior_scale", torch.tensor(0.1))
        self.register_buffer("pert_loading_prior_scale", torch.tensor(0.1))

        # for LogNormalNegativeBinomial likelihood hyperparams
        self.register_buffer("noise_prior_rate", torch.tensor(2.0))

        # override with user-provided values from prior_param_dict
        if prior_param_dict is not None:
            for k, v in prior_param_dict.items():
                buffer_keys = [k for k, v in self.named_buffers()]
                assert isinstance(v, torch.Tensor) and k in buffer_keys
                assert v.shape == self.get_buffer(k).shape
                self.register_buffer(k, v)

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        print(tensor_dict.keys())

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

    def infer_data_dims(self, idx, **tensor_dict):
        """Infer model dimensions based on model args/kwargs and make sure they match as needed"""

        dims = defaultdict(lambda: 1)  # give any unspecified dims a value of 1
        if not tensor_dict:
            # raise Exception(idx, tensor_dict)
            return dims

        def check_and_validate_dict(k, v, d):
            if k not in d:
                d[k] = v
            else:
                assert d[k] == v

        # check cell index
        if idx is not None:
            n_cells = idx.shape[0]
            check_and_validate_dict("n_cells", n_cells, dims)

        # check counts matrix
        X = tensor_dict.get(REGISTRY_KEYS.X_KEY)
        if X is not None:
            n_cells, n_genes = X.shape
            check_and_validate_dict("n_cells", n_cells, dims)
            check_and_validate_dict("n_genes", n_genes, dims)

        # check batch labels
        batch = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        if batch is not None:
            n_cells, n_batches = batch.shape
            check_and_validate_dict("n_cells", n_cells, dims)
            check_and_validate_dict("n_batches", n_batches, dims)

        # check size factors
        size_factor = tensor_dict[REGISTRY_KEYS.SIZE_FACTOR_KEY]
        if size_factor is not None:
            size_factor, _ = size_factor.shape
            check_and_validate_dict("n_cells", n_cells, dims)

        # check guides matrix (required)
        guides = tensor_dict[REGISTRY_KEYS.PERTURBATION_KEY]
        n_cells, n_guides = guides.shape
        check_and_validate_dict("n_cells", n_cells, dims)
        check_and_validate_dict("n_guides", n_guides, dims)

        # check continuous covariates matrix
        covariates = tensor_dict[REGISTRY_KEYS.CONT_COVS_KEY]
        n_cells, n_cont_covariates = covariates.shape
        assert n_cont_covariates == self.n_cont_covariates, "n_cont_covariates must not change"

        # check guide by element matrix
        # guide_by_element = tensor_dict[REGISTRY_KEYS.GUIDE_BY_ELEMENT_KEY]
        # n_guides, n_elements = guide_by_element.shape
        # check_and_validate_dict("n_guides", n_guides, dims)
        # check_and_validate_dict("n_elements", n_elements, dims)

        # check guide by element matrix
        # gene_by_element = tensor_dict[REGISTRY_KEYS.GENE_BY_ELEMENT_KEY]
        # n_genes, n_elements = gene_by_element.shape
        # check_and_validate_dict("n_genes", n_genes, dims)
        # check_and_validate_dict("n_elements", n_elements, dims)

        # set all other dims based on model args
        dims["n_cont_covariates"] = self.n_cont_covariates
        dims["n_factors"] = self.n_factors
        dims["n_pert_factors"] = self.n_pert_factors

        if self.local_effects:
            dims["n_element_effects"] = self.element_by_gene_idx.shape[1]

        print(dims)
        return dims

    def create_plates(self, idx, **tensor_dict):
        dims = self.infer_data_dims(idx, **tensor_dict)
        return (
            pyro.plate("Cells", self.n_cells, dim=-2, subsample=idx),
            pyro.plate("Guides", self.n_perturbations, dim=-2),
            pyro.plate("Elements", self.n_elements, dim=-2),
            pyro.plate("Batches", self.n_batches, dim=-2),
            pyro.plate("Genes", self.n_genes, dim=-1),
            pyro.plate("Covariates", self.n_cont_covariates, dim=-2),
            pyro.plate("Elements_sparse", self.n_element_effects, dim=-1),
            pyro.plate("Cell_factors", self.n_factors, dim=-3),
            pyro.plate("Pert_factors", self.n_pert_factors, dim=-3),
        )

        return (
            pyro.plate("Cells", dims["n_cells"], dim=-2, subsample=idx),
            pyro.plate("Guides", dims["n_perturbations"], dim=-2),
            pyro.plate("Elements", dims["n_elements"], dim=-2),
            pyro.plate("Batches", dims["n_batches"], dim=-2),
            pyro.plate("Genes", dims["n_genes"], dim=-1),
            pyro.plate("Covariates", dims["n_cont_covariates"], dim=-2),
            pyro.plate("Elements_sparse", dims["n_element_effects"], dim=-1),
            pyro.plate("Cell_factors", dims["n_factors"], dim=-3),
            pyro.plate("Pert_factors", dims["n_pert_factors"], dim=-3),
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
            cell_factor_plate,
            pert_factor_plate,
        ) = self.create_plates(idx)
        dims = self.infer_data_dims(idx, **tensor_dict)
        batch = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        size_factor = tensor_dict[REGISTRY_KEYS.SIZE_FACTOR_KEY]
        guides_observed = tensor_dict[REGISTRY_KEYS.PERTURBATION_KEY]
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
        elif self.effect_prior_dist == "laplace":
            effects_dist = dist.Laplace(0.0, self.element_effects_prior_scale)

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
        if self.merge_guides_mode == "shared":
            guide_efficacy_values = self.one.expand((self.n_perturbations, 1))
        else:
            with guide_plate:
                guide_efficacy_values = pyro.sample(
                    "guide_efficacy",
                    dist.Beta(self.logit_efficacy_alpha, self.logit_efficacy_beta),
                )
            # fix weird broadcasting error
        guide_efficacy_by_element = guide_efficacy_values.expand(-1, self.n_elements) * self.guide_by_element

        # Sample dense or factorized perturbation effects
        if self.n_pert_factors is None:
            element_factor_effects = 0
        else:
            with pert_factor_plate, element_plate:
                pert_factors = pyro.sample("pert_factors", dist.Laplace(0.0, self.pert_factor_prior_scale))
            with pert_factor_plate, gene_plate:
                pert_loadings = pyro.sample("pert_loadings", dist.Laplace(0.0, self.pert_loading_prior_scale))
            element_factor_effects = torch.einsum("fei,fjg->eg", pert_factors, pert_loadings)

        # Sample cell-specific factors (linear unobserved confounders) if using
        if self.n_factors is not None:
            with cell_factor_plate, cell_plate:
                cell_factors = pyro.sample(
                    "cell_factors",
                    dist.Laplace(0.0, self.cell_factor_prior_scale),
                )
            with cell_factor_plate, gene_plate:
                cell_loadings = pyro.sample(
                    "cell_loadings",
                    dist.Laplace(0.0, self.cell_loading_prior_scale),
                )
            cell_factor_effects = torch.einsum("fci,fjg->cg", cell_factors, cell_loadings)

            if self.use_interactions and self.n_pert_factors is not None:
                with cell_factor_plate, element_plate:
                    pert_cell_factors = pyro.sample(
                        "pert_cell_factors",
                        dist.Laplace(0.0, self.cell_factor_prior_scale),
                    )
                element_factor_effects = (
                    torch.einsum("fei,fjg->eg", pert_cell_factors, cell_loadings) + element_factor_effects
                )
        else:
            cell_factor_effects = 0

        if self.local_effects:
            # override factor effects
            element_effects = (1 - self.element_by_gene) * element_factor_effects + element_local_effects
        else:
            element_effects = element_factor_effects + element_local_effects

        # Account for cell-specific latent "perturbation status" variable(s)
        with cell_plate:
            if self.efficiency_mode == "scaled":
                cell_element_efficacy = guides_observed @ guide_efficacy_by_element
            elif self.efficiency_mode == "mixture":
                pert_prob = guides_observed @ guide_efficacy_values
                perturbed = pyro.sample("perturbed", dist.Bernoulli(pert_prob))
                cell_element_efficacy = perturbed * guides_observed @ self.guide_by_element
                # only targeting guides can have a CRISPR effect
            else:
                raise Exception("efficiency_mode must be either 'scaled' or 'mixture'")

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

            # Calculate final expression distribution parameters for unperturbed cells
            nb_log_mean_ctrl = (
                gene_base_log_mean + size_factor + batch_effects + covariate_effects + cell_factor_effects
            )

            if not self.dispersion_effects:
                nb_log_dispersion = gene_log_dispersion
            else:
                nb_log_dispersion = gene_log_dispersion + batch_disp_effects

            # Add perturbation effects to gene expression parameters
            # raise Exception(cell_guide_efficacy.shape, self.guide_by_element.shape, element_effects.shape)
            mean_perturbation_effect = cell_element_efficacy @ element_effects

            # if desired, add mean "perturbation effect" from perturbation modality
            if self.use_crispr_factor:
                crispr_loading = pyro.sample("crispr_loading", dist.Laplace(self.zero, self.one))
                crispr_effect = torch.log2(cell_element_efficacy.sum(dim=-1, keepdim=True) + 1) * crispr_loading
                mean_perturbation_effect += crispr_effect

            nb_log_mean = nb_log_mean_ctrl + mean_perturbation_effect
            # nb_log_mean = nb_log_mean_ctrl + (cell_guide_efficacy @ self.guide_by_element @ element_effects)

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
