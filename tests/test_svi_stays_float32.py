"""Both SVI stages keep their parameters in float32 although the package enables float64.

float64 is enabled at import for the CRT's tails. If it leaked into the SVI
parameters, every cells-by-genes intermediate of the likelihood would double,
and a chunk that fits on a 40 GB card would not.
"""
import shutil
from pathlib import Path

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
import pandas as pd
import pytest
import scipy.sparse as sp

from perturbo.api import fit_from_path
from perturbo.core import _pin_svi_params_float32
import perturbo.model as model_module
from perturbo.sparse_design import indexed_design_from_matrix


def _write_screen(path: Path, n_cells=360, n_genes=24, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array(["non-targeting"] * 160 + ["A"] * 100 + ["B"] * 100)
    mu = np.exp(rng.normal(1.5, 0.4, size=n_genes))[None, :] * np.ones((n_cells, 1))
    mu[labels == "A", 1] *= 0.4
    counts = rng.negative_binomial(5.0, 5.0 / (5.0 + mu)).astype(np.float32)
    adata = ad.AnnData(
        X=sp.csr_matrix(counts),
        obs=pd.DataFrame({"pert": labels, "total_umis": counts.sum(axis=1).astype(np.int64) + 20},
                         index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )
    adata.write_h5ad(path)


def test_control_and_effect_likelihoods_and_parameters_are_float32(tmp_path, monkeypatch):
    assert jax.config.jax_enable_x64, "the package is expected to enable float64 at import"
    likelihood_dtypes = []
    sample_observations = model_module._sample_observations

    def record_likelihood(**kwargs):
        value = sample_observations(**kwargs)
        likelihood_dtypes.append((kwargs["logits"].dtype, kwargs["theta"].dtype))
        return value

    monkeypatch.setattr(model_module, "_sample_observations", record_likelihood)
    screen = tmp_path / "screen.h5ad"
    _write_screen(screen)
    # The repository pins a fixed pytest temp base, so this directory survives between
    # runs; inspecting a previous run's parameter bundles would fail on artefacts this
    # run never wrote.
    out = tmp_path / "out"
    if out.exists():
        shutil.rmtree(out)
    fit_from_path(
        str(screen), out_dir=str(out), perturbation_key="pert", control_substring="non-targeting",
        library_size_key="total_umis", size_factor_mode="observed", likelihood="nb",
        num_steps_control=5, num_steps_betas=5, save_model_params=True,
    )
    npz_files = sorted(out.rglob("*.npz"))
    assert npz_files, "the run should save its stage-one fit and its model parameters"
    offenders = []
    for path in npz_files:
        with np.load(path, allow_pickle=True) as bundle:
            for key in bundle.files:
                value = bundle[key]
                if value.dtype.kind == "f" and value.dtype != np.float32 and "loss" not in key:
                    offenders.append(f"{path.name}:{key}:{value.dtype}")
    assert not offenders, "float64 leaked into saved SVI parameters: " + ", ".join(offenders[:8])
    assert likelihood_dtypes
    assert set(likelihood_dtypes) == {(jnp.dtype("float32"), jnp.dtype("float32"))}


@pytest.mark.parametrize("likelihood", ["nb", "censored_nb", "lnnb", "mixture_nb"])
def test_every_count_likelihood_evaluates_in_float32(likelihood):
    counts = jnp.asarray([[0, 2], [1, 4]], dtype=jnp.int32)

    def model():
        return model_module._sample_observations(
            counts=counts,
            likelihood=likelihood,
            logits=jnp.asarray([[0.2, -0.4], [1.1, 0.3]], dtype=jnp.float64),
            theta=jnp.asarray([3.0, 7.0], dtype=jnp.float64),
            noise_scale=jnp.asarray([0.2, 0.3], dtype=jnp.float64),
            logits_outlier=jnp.asarray([[1.2, 0.6], [2.1, 1.3]], dtype=jnp.float64),
            theta_outlier=jnp.asarray([1.5, 2.5], dtype=jnp.float64),
            pi_outlier=jnp.asarray([0.05, 0.1], dtype=jnp.float64),
            count_censoring_threshold=jnp.asarray([3, 3]),
        )

    trace = numpyro.handlers.trace(numpyro.handlers.seed(model, jax.random.key(1))).get_trace()
    assert trace["obs"]["fn"].log_prob(counts).dtype == jnp.float32


def test_float32_nb_gradient_tracks_float64_reference():
    design = jnp.asarray([[1.0, -0.5], [1.0, 0.25], [1.0, 1.5]])
    counts = jnp.asarray([[0, 3], [2, 1], [5, 4]])
    theta = jnp.asarray([2.5, 8.0])
    beta = jnp.asarray([[0.3, -0.2], [0.15, 0.4]])

    def objective(value, dtype):
        logits = design.astype(dtype) @ value.astype(dtype)
        return dist.NegativeBinomialLogits(
            logits=logits, total_count=theta.astype(dtype)
        ).log_prob(counts).sum()

    grad32 = jax.grad(lambda value: objective(value, jnp.float32))(beta.astype(jnp.float32))
    grad64 = jax.grad(lambda value: objective(value, jnp.float64))(beta.astype(jnp.float64))
    np.testing.assert_allclose(grad32, grad64, rtol=2e-6, atol=2e-6)


def test_indexed_high_moi_full_batch_likelihood_and_update_are_float32(monkeypatch):
    counts = jnp.asarray([[2, 1], [0, 3], [4, 2], [1, 0], [3, 5], [2, 2]])
    guides = np.array(
        [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]],
        dtype=np.float32,
    )
    guide_to_element = jnp.asarray([[1, 0], [1, 0], [0, 1]], dtype=jnp.float32)
    elements = (guides @ np.asarray(guide_to_element) > 0).astype(np.float32)
    seen = []
    sample_observations = model_module._sample_observations

    def record_likelihood(**kwargs):
        seen.append((kwargs["logits"].dtype, kwargs["theta"].dtype))
        return sample_observations(**kwargs)

    monkeypatch.setattr(model_module, "_sample_observations", record_likelihood)
    model = model_module.GuideSharedNegativeBinomialModel
    guide = AutoNormal(model, create_plates=model_module.create_plates)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    kwargs = dict(
        counts=counts,
        pert_id=indexed_design_from_matrix(elements),
        guide_matrix=indexed_design_from_matrix(guides),
        guide_to_element=guide_to_element,
        size_factors=jnp.linspace(-0.2, 0.2, counts.shape[0], dtype=jnp.float64)[:, None],
        covariates=jnp.linspace(-1.0, 1.0, counts.shape[0], dtype=jnp.float64)[:, None],
        num_cells=counts.shape[0], num_genes=counts.shape[1], num_perts=2,
        num_guides=3, guide_effect_strategy="shared",
    )
    state = _pin_svi_params_float32(svi, svi.init(jax.random.key(3), **kwargs))
    before = svi.get_params(state)["beta_auto_loc"]
    state, loss = svi.update(state, **kwargs)
    after = svi.get_params(state)["beta_auto_loc"]

    assert np.isfinite(loss)
    assert seen and set(seen) == {(jnp.dtype("float32"), jnp.dtype("float32"))}
    assert after.dtype == jnp.float32
    assert np.isfinite(np.asarray(after)).all()
    assert not np.array_equal(before, after), "the indexed beta gradient should update the fit"


def test_the_pin_changes_dtypes_and_nothing_else():
    """The pin must leave every parameter's value where initialisation put it.

    Reading the constrained parameters and handing them back to the optimizer as if
    they were unconstrained moved every positive scale from s to exp(s): a guide
    initialised with scale 0.1 restarted at 1.1, the initial loss rose by 40% on the
    Replogle essential controls and stage one needed thousands of steps to recover.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal

    from perturbo.core import _pin_svi_params_float32

    def model(y):
        mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
        theta = numpyro.sample("theta", dist.LogNormal(0.0, 2.0))
        numpyro.sample("y", dist.Normal(mu, theta), obs=y)

    y = jnp.asarray([1.0, 2.0, 3.0])
    guide = AutoNormal(model, init_scale=0.1)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    state = svi.init(jax.random.PRNGKey(0), y)
    before = svi.get_params(state)
    after_state = _pin_svi_params_float32(svi, state)
    after = svi.get_params(after_state)

    assert set(before) == set(after)
    for name in before:
        assert after[name].dtype == jnp.float32, name
        assert jnp.allclose(jnp.asarray(before[name], jnp.float32), after[name], rtol=1e-6), (
            name, before[name], after[name]
        )
    scales = [after[n] for n in after if n.endswith("_scale")]
    assert scales and all(float(jnp.max(s)) < 0.2 for s in scales), "scales must stay at their 0.1 initialisation"
    # and the objective is untouched by the pin
    loss_before = svi.evaluate(state, y)
    loss_after = svi.evaluate(after_state, y)
    assert jnp.allclose(loss_before, loss_after, rtol=1e-4), (loss_before, loss_after)
