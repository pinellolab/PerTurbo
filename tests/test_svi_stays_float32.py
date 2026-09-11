"""Both SVI stages keep their parameters in float32 although the package enables float64.

float64 is enabled at import for the CRT's tails. If it leaked into the SVI
parameters, every cells-by-genes intermediate of the likelihood would double,
and a chunk that fits on a 40 GB card would not.
"""
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
    out = tmp_path / "out"
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
