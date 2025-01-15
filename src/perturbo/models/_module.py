from collections.abc import Mapping
from typing import Literal

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer.autoguide import AutoDelta, AutoGuideList, AutoNormal, init_to_median, init_to_value
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
        n_cells: int | None = None,
        n_genes: int | None = None,
        n_perturbations: int | None = None,
        n_elements: int | None = None,
        n_cont_covariates: int | None = None,
        n_batches: int | None = 1,
        gene_means: torch.Tensor | None = None,
        guide_by_element: torch.Tensor | None = None,
        gene_by_element: torch.Tensor | None = None,
        guide_noise: bool = False,
        likelihood: Literal["nb", "lnnb"] = "nb",
        effect_prior_dist: Literal["cauchy", "normal_mixture", "normal", "laplace"] = "laplace",
        n_factors: int | None = None,
        n_pert_factors: int | None = None,
        use_interactions: bool = False,
        efficiency_mode: Literal["mixture", "scaled"] = "scaled",
        dispersion_effects: bool = False,
        fit_guide_efficacy: bool = True,
        prior_param_dict: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """
        Pyro module underlying perturbo.

        Parameters
        ----------
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
        self.fit_guide_efficacy = fit_guide_efficacy
        self.lnnb_quad_points = 8
        self.n_factors = n_factors
        self.n_pert_factors = n_pert_factors
        self.effect_prior_dist = effect_prior_dist
        self.use_interactions = use_interactions
        self.efficiency_mode = efficiency_mode
        self.local_effects = gene_by_element is not None
        self.guide_noise = guide_noise

        # copy data summary stats
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.n_perturbations = n_perturbations
        self.n_cont_covariates = 1  # include (inferred) size factor as covariate always

        self.discrete_sites = []
        if efficiency_mode == "mixture":
            self.discrete_sites.append("perturbed")

        # validate guide -> element mapping or use identity matrix as default
        if guide_by_element is None:
            guide_by_element = torch.eye(self.n_perturbations)
        else:
            assert n_elements is not None, "n_elements must be specified if not equal to n_guides"
        self.n_elements = guide_by_element.shape[1]

        if n_cont_covariates is not None:
            self.n_cont_covariates += n_cont_covariates

        self.n_batches = n_batches

        self.delta_sites = []
        # self.delta_sites = ["cell_factors"]
        # self.delta_sites = ["cell_factors", "cell_loadings", "pert_factors", "pert_loadings"]

        self._guide = AutoGuideList(self.model, create_plates=self.create_plates)
        # init_values = {}
        # if gene_means is not None:
        #     init_values["log_gene_mean"] = torch.log(gene_means)
        # if control_pcs is not None and n_factors is not None:
        #     init_values["cell_loadings"] = control_pcs

        self._guide.append(
            AutoNormal(
                poutine.block(self.model, hide=self.delta_sites + self.discrete_sites),
                init_loc_fn=lambda x: init_to_median(x, num_samples=100),
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
        if gene_means is not None:
            e_x = gene_means
            epsilon = 1 / self.n_cells
            self.register_buffer("gene_mean_prior_loc", torch.tensor(e_x + epsilon).log().squeeze())
        else:
            self.register_buffer("gene_mean_prior_loc", torch.tensor(0.0))
        self.register_buffer("gene_disp_prior_loc", torch.tensor(1.0))

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
        effect_prior_scales = {"cauchy": 0.01, "laplace": 0.1, "normal": 1.0}
        model_effect_prior_scale = effect_prior_scales[effect_prior_dist]
        self.register_buffer("element_effects_prior_scale", torch.tensor(model_effect_prior_scale))
        self.register_buffer("guide_effects_prior_scale", torch.tensor(model_effect_prior_scale))

        # normal mixture prior hyperparams
        self.register_buffer("spike_slab_prior_scales", torch.tensor([1.0, 0.1]))
        self.register_buffer("spike_slab_prior_probs", torch.tensor([0.01, 0.99]))

        # (contrastive) factor model hyperparams
        self.register_buffer("cell_factor_prior_scale", torch.tensor(1.0))
        self.register_buffer("cell_loading_prior_scale", torch.tensor(1.0))
        self.register_buffer("pert_factor_prior_scale", torch.tensor(0.1))
        self.register_buffer("pert_loading_prior_scale", torch.tensor(1.0))

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
        # dims = self.infer_data_dims(idx, **tensor_dict)
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
        # set to the correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                element_local_effects = element_local_effects  # fix device mismatch error

        # Pool guide information based on user-specified strategy
        if not self.fit_guide_efficacy:
            guide_efficacy = self.one.expand((self.n_perturbations, 1))
        else:
            with guide_plate:
                guide_efficacy = pyro.sample(
                    "guide_efficacy",
                    dist.Beta(self.logit_efficacy_alpha, self.logit_efficacy_beta),
                )

        # Sample dense or factorized perturbation effects
        if self.n_pert_factors is None:
            element_factor_effects = 0
        else:
            with pert_factor_plate, element_plate:
                pert_factors = pyro.sample("pert_factors", dist.Laplace(0.0, self.pert_factor_prior_scale))
            with pert_factor_plate, gene_plate:
                pert_loadings = pyro.sample("pert_loadings", dist.Laplace(0.0, self.pert_loading_prior_scale))
            # pert_factor_scale_term = pyro.sample(
            #     "pert_factor_scale_term",
            #     dist.LogNormal(-self.one, self.one),
            # )
            element_factor_effects = torch.einsum("fei,fjg->eg", pert_factors, pert_loadings)

        # fix device mismatch error
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        guides_observed = guides_observed

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
            # cell_factor_scale_term = pyro.sample(
            #     "cell_factor_scale_term",
            #     dist.LogNormal(self.zero, self.one),
            # )
            cell_factor_effects = torch.einsum("fci,fjg->cg", cell_factors, cell_loadings)

            # if self.use_interactions and self.n_pert_factors is not None:
            #     with cell_factor_plate, element_plate:
            #         pert_cell_factors = pyro.sample(
            #             "pert_cell_factors",
            #             dist.Laplace(0.0, self.cell_factor_prior_scale),
            #         )
            #     element_factor_effects = (
            #         torch.einsum("fei,fjg->eg", pert_cell_factors, cell_loadings) + element_factor_effects
            #     )
        else:
            cell_factor_effects = 0

        if self.local_effects:
            # override factor effects
            element_effects = (1 - self.element_by_gene) * element_factor_effects + element_local_effects
            # guide_factor_efects = self.guide_by_element @ ((1 - self.element_by_gene) * element_factor_effects)
            # guide_local_effects = (guide_efficacy * self.guide_by_element) @ element_local_effects
            # guide_effects = guide_factor_efects + guide_local_effects
        else:
            element_effects = element_factor_effects + element_local_effects

        guide_effects = (guide_efficacy * self.guide_by_element) @ element_effects

        # compute/sample guide effects as function of element effects
        with guide_plate, gene_plate:
            if self.guide_noise:
                guide_effects = pyro.sample(
                    "guide_effects",
                    dist.Laplace(self.zero, self.guide_effects_prior_scale),
                )
            else:
                guide_effects = pyro.deterministic("guide_effects", guide_effects)

        # Account for cell-specific latent "perturbation status" variable(s)
        with cell_plate:
            if self.efficiency_mode == "scaled":
                perturbed = 1
            elif self.efficiency_mode == "mixture":
                pert_prob = guides_observed @ guide_efficacy
                perturbed = pyro.sample("perturbed", dist.Bernoulli(pert_prob), infer={"enumerate": "parallel"})
            else:
                raise Exception("efficiency_mode must be either 'scaled' or 'mixture'")

        mean_perturbation_effect = perturbed * guides_observed @ guide_effects
        # if self.use_crispr_factor:
        #     log_guide_counts = torch.log2(1 + guides_observed.sum(dim=-1, keepdim=True))
        #     with gene_plate:
        #         crispr_loading = pyro.sample(
        #             "crispr_loading", dist.Laplace(self.zero, self.pert_loading_prior_scale * 2)
        #         )
        #         mean_perturbation_effect += log_guide_counts @ crispr_loading.unsqueeze(dim=-2)
        # log2_1_plus_n_targeting_guides = torch.log2(1 + guides_observed.sum(dim=-1, keepdim=True))
        # mean_perturbation_effect += log2_1_plus_n_targeting_guides @ crispr_loading.unsqueeze(dim=-2)

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
                multiplicative_noise = pyro.sample("multiplicative_noise", dist.Exponential(self.noise_prior_rate)).to(
                    device
                )
                # multiplicative_noise = 1 / self.noise_prior_rate

            with batch_plate:
                # batch effects: n_batches x n_genes
                batch_effect_size = pyro.sample("batch_effect", dist.Normal(0.0, self.batch_effect_prior_scale)).to(
                    device
                )
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

            nb_log_mean_ctrl = (
                gene_base_log_mean + size_factor + batch_effects + covariate_effects + cell_factor_effects
            )

            if not self.dispersion_effects:
                nb_log_dispersion = gene_log_dispersion
            else:
                nb_log_dispersion = gene_log_dispersion + batch_disp_effects

            nb_log_mean = nb_log_mean_ctrl + mean_perturbation_effect

            # Sample read counts from distributions
            with cell_plate:
                observations = tensor_dict.get(REGISTRY_KEYS.X_KEY)
                if self.likelihood == "lnnb":
                    nb_log_mean = nb_log_mean
                    nb_log_dispersion = nb_log_dispersion
                    multiplicative_noise = multiplicative_noise
                    self.lnnb_quad_points = self.lnnb_quad_points
                    observations = observations
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
                    nb_log_mean = nb_log_mean
                    nb_log_dispersion = nb_log_dispersion
                    observations = observations
                    return pyro.sample(
                        "obs",
                        dist.NegativeBinomial(
                            logits=nb_log_mean,
                            total_count=nb_log_dispersion.exp(),
                        ),
                        obs=observations,
                    )
                else:
                    raise NotImplementedError(f"'{self.likelihood}' likelihood not implemented")

    @property
    def guide(self):
        return self._guide
