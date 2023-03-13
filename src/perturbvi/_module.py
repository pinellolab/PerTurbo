import pyro
import pyro.distributions as dist
import torch
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.infer.autoguide import AutoDelta, AutoNormal
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
        likelihood="nb",
        fit_lib_size_effect=False,
        **module_kwargs,
    ) -> None:
        super().__init__()
        self.n_cells = summary_stats.n_cells
        self.n_vars = summary_stats.n_vars
        self.n_perturbations = summary_stats.n_perturbations
        self.n_batches = summary_stats.n_batch
        self.likelihood = likelihood
        self.fit_lib_size_effect = fit_lib_size_effect
        self._guide = AutoNormal(self.model)

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        return (tensor_dict[REGISTRY_KEYS.INDICES_KEY],), tensor_dict

    def create_plates(self, idx, subsample_size=None, **tensor_dict):
        # if self.n_vars < subsample_size:
        #     subsample_size = None

        return (
            pyro.plate("cells", self.n_cells, dim=-2, subsample=idx),
            pyro.plate("perturbations", self.n_perturbations, dim=-2),
            pyro.plate("batches", self.n_batches, dim=-2),
            pyro.plate("vars", self.n_vars, dim=-1, subsample_size=subsample_size),
            # pyro.plate("covariates", self.n_batches, dim=-2),
        )

    def model(self, idx, **tensor_dict):
        pyro.module("perturbvi", self)
        cell_plate, perturbation_plate, batch_plate, var_plate = self.create_plates(idx)
        batch = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        size_factor = tensor_dict[REGISTRY_KEYS.SIZE_FACTOR_KEY]
        perturbations = tensor_dict[REGISTRY_KEYS.PERTURBATION_KEY]
        log_var_mean_global = pyro.sample("log_var_mean_global", dist.Cauchy(0.0, 2.0))

        with var_plate:
            if self.fit_lib_size_effect:
                library_size_effect = pyro.sample(
                    "library_size_effect", dist.Normal(1.0, 0.1)
                )
            else:
                library_size_effect = 1
            if self.likelihood == "nb_mix":
                mixture_logits = pyro.sample("mixture_logits", dist.Normal(-1.0, 0.01))
                mixture_logits = -1.0
            with batch_plate:
                batch_effect_size = pyro.sample("batch_effect", dist.Normal(0.0, 1.0))
                batch_effects = batch_effect_size[batch.squeeze(), ...]
            with perturbation_plate:
                # spike_frac = 1e-4
                # spike_slab_mix = dist.Categorical(
                #     torch.tensor((1.0 - spike_frac, spike_frac))
                # )
                # spike_slab_means = torch.tensor((0.0, 0.0))
                # spike_slab_vars = torch.tensor((0.1, 1.0))
                # spike_slab_comp = dist.Normal(spike_slab_means, spike_slab_vars)
                # spike_slab_dist = dist.MixtureSameFamily(
                #     spike_slab_mix, spike_slab_comp
                # )
                # perturb_mean_lfc = pyro.sample("perturb_mean_lfc", spike_slab_dist)
                # perturb_disp_lfc = pyro.sample(
                #     "perturb_disp_lfc", dist.Cauchy(0.0, 0.1)
                # )
                perturb_mean_lfc = pyro.sample("perturb_mean_lfc", dist.Cauchy(0, 0.1))
                perturb_disp_lfc = pyro.sample("perturb_disp_lfc", dist.Normal(0, 0.1))

            log_var_mean = (
                pyro.sample("log_var_mean", dist.Normal(0.0, 3.0)) + log_var_mean_global
            )
            log_var_dispersion = pyro.sample(
                "log_var_dispersion", dist.Normal(0.0, 2.0)
            )

            nb_log_dispersion = log_var_dispersion + perturbations @ perturb_disp_lfc
            nb_log_mean = (
                log_var_mean
                + perturbations @ perturb_mean_lfc
                + size_factor * library_size_effect
                + batch_effects
            )

            with cell_plate:
                if self.likelihood == "nb_mix":
                    return pyro.sample(
                        "obs",
                        NegativeBinomialMixture(
                            mu1=log_var_mean.exp(),
                            theta1=(-log_var_dispersion).exp(),
                            mu2=nb_log_mean.exp(),
                            theta2=(-nb_log_dispersion).exp(),
                            mixture_logits=mixture_logits,
                            validate_args=True,
                        ),
                        obs=tensor_dict[REGISTRY_KEYS.X_KEY],
                    )

                return pyro.sample(
                    "obs",
                    NegativeBinomial(
                        theta=(-nb_log_dispersion).exp(),
                        mu=nb_log_mean.exp(),
                    ),
                    obs=tensor_dict[REGISTRY_KEYS.X_KEY],
                )

    def guide(self, idx, **tensor_dict):
        return self._guide(idx, **tensor_dict)

    # def guide(self, idx, init_scale=0.2, **tensor_dict):
    #     pyro.module("perturbvi", self)
    #     # scale_factor = pyro.param("scale_factor", torch.tensor(init_scale).log()).exp()
    #     cell_plate, perturbation_plate, batch_plate, var_plate = self.create_plates(idx)

    #     if self.fit_lib_size_effect:
    #         library_size_effect_mu = pyro.param(
    #             "library_size_effect.mu", lambda: torch.full((self.n_vars,), init_scale)
    #         )
    #         library_size_effect_sigma = pyro.param(
    #             "library_size_effect.sigma",
    #             lambda: torch.full((self.n_vars,), init_scale),
    #             constraint=dist.constraints.positive,
    #         )
    #     # shared global mean parameter
    #     log_var_mean_global_mu = pyro.param(
    #         "log_var_mean_global.mu", lambda: torch.tensor((0.0,))
    #     )
    #     pyro.sample("log_var_mean_global", dist.Delta(log_var_mean_global_mu))

    #     log_var_mean_mu = pyro.param(
    #         "log_var_mean.mu", lambda: torch.zeros((self.n_vars,))
    #     )
    #     log_var_disp_mu = pyro.param(
    #         "log_var_disp.mu", lambda: torch.zeros((self.n_vars,))
    #     )
    #     if self.likelihood == "nb_mix":
    #         mixture_logits_mu = pyro.param(
    #             "mixture_logits.mu", lambda: torch.zeros((self.n_vars,))
    #         )
    #         mixture_logits_sigma = pyro.param(
    #             "mixture_logits.sigma",
    #             lambda: torch.full((self.n_vars,), init_scale),
    #             constraint=dist.constraints.positive,
    #         )

    #     batch_effect_mu = pyro.param(
    #         "batch_effect.mu", lambda: torch.zeros((self.n_batches, self.n_vars))
    #     )
    #     batch_effect_sigma = pyro.param(
    #         "batch_effect.sigma",
    #         lambda: torch.full((self.n_batches, self.n_vars), init_scale),
    #         constraint=dist.constraints.positive,
    #     )

    #     log_var_mean_sigma = pyro.param(
    #         "log_var_mean.sigma",
    #         lambda: torch.full((self.n_vars,), init_scale),
    #         constraint=dist.constraints.positive,
    #     )
    #     log_var_disp_sigma = pyro.param(
    #         "log_var_disp.sigma",
    #         lambda: torch.full((self.n_vars,), init_scale),
    #         constraint=dist.constraints.positive,
    #     )

    #     perturb_mean_lfc_mu = pyro.param(
    #         "perturb_mean_lfc.mu",
    #         lambda: torch.zeros((self.n_perturbations, self.n_vars)),
    #     )
    #     perturb_disp_lfc_mu = pyro.param(
    #         "perturb_disp_lfc.mu",
    #         lambda: torch.zeros((self.n_perturbations, self.n_vars)),
    #     )
    #     perturb_lfc_mu = torch.stack((perturb_mean_lfc_mu, perturb_disp_lfc_mu), dim=-1)

    #     perturb_lfc_scale_tril = pyro.param(
    #         "perturb_lfc.scale_tril",
    #         lambda: torch.eye(2).repeat((self.n_perturbations, self.n_vars, 1, 1))
    #         * init_scale,
    #         constraint=dist.constraints.lower_cholesky,
    #     )

    #     with var_plate:
    #         if self.fit_lib_size_effect:
    #             pyro.sample(
    #                 "library_size_effect",
    #                 dist.Normal(library_size_effect_mu, library_size_effect_sigma),
    #             )
    #         pyro.sample(
    #             "log_var_mean", dist.Normal(log_var_mean_mu, log_var_mean_sigma)
    #         )
    #         pyro.sample(
    #             "log_var_dispersion", dist.Normal(log_var_disp_mu, log_var_disp_sigma)
    #         )
    #         if self.likelihood == "nb_mix":
    #             pyro.sample(
    #                 "mixture_logits",
    #                 dist.Normal(mixture_logits_mu, mixture_logits_sigma),
    #             )
    #         with batch_plate:
    #             pyro.sample(
    #                 "batch_effect", dist.Normal(batch_effect_mu, batch_effect_sigma)
    #             )

    #         with perturbation_plate:
    #             perturb_lfc = pyro.sample(
    #                 "perturb_lfc",
    #                 dist.MultivariateNormal(
    #                     perturb_lfc_mu,
    #                     scale_tril=perturb_lfc_scale_tril,
    #                 ),
    #                 infer={"is_auxiliary": True},
    #             )

    #             pyro.sample("perturb_mean_lfc", dist.Delta(perturb_lfc[..., 0]))
    #             pyro.sample("perturb_disp_lfc", dist.Delta(perturb_lfc[..., 1]))

    @staticmethod
    def get_perturbation_effects():
        """Return the perturbation effects on each variable's mean and variance."""
        store = pyro.get_param_store()
        return (
            store["perturb_mean_lfc.mu"].detach().cpu().numpy(),
            store["perturb_disp_lfc.mu"].detach().cpu().numpy(),
        )
