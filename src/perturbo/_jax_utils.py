import jax.numpy as jnp
import numpyro
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS, SVI, TraceMeanField_ELBO


def render_perturbseq_model(model):
    gene_obs = jnp.zeros((1, 1))
    guide_obs = jnp.zeros((1,))
    return numpyro.render_model(
        model,
        model_args=(gene_obs,),
        model_kwargs={"guide_obs": guide_obs},
        render_distributions=True,
        render_params=True,
    )


def run_mcmc(args, kwargs, model, unconstrained_locs=None, dense_mass=False, num_samples=1000, random_seed=0):
    n_chains = 1
    kernel = NUTS(model, dense_mass=dense_mass)
    mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=1000, num_chains=n_chains)
    mcmc.run(PRNGKey(random_seed), *args, **kwargs, init_params=unconstrained_locs)
    return mcmc


def run_svi(args, kwargs, model, guide, lr=0.01, n_steps=1000, random_seed=0):
    adam = numpyro.optim.Adam(step_size=lr)
    svi = SVI(model, guide, adam, loss=TraceMeanField_ELBO())
    svi_result = svi.run(PRNGKey(random_seed), n_steps, *args, **kwargs)
    return svi_result
