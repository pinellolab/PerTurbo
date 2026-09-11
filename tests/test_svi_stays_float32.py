"""Both SVI stages keep their parameters in float32 although the package enables float64.

float64 is enabled at import for the CRT's tails. If it leaked into the SVI
parameters, every cells-by-genes intermediate of the likelihood would double,
and a chunk that fits on a 40 GB card would not.
"""
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from perturbo.api import fit_from_path


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


def test_control_and_effect_parameters_are_float32(tmp_path):
    import jax
    assert jax.config.jax_enable_x64, "the package is expected to enable float64 at import"
    screen = tmp_path / "screen.h5ad"; _write_screen(screen)
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
