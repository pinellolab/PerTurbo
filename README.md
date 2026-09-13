# PerTurbo

PerTurbo is a NumPyro/JAX implementation of Bayesian Perturb-seq analysis for
low- and high-MOI single-cell CRISPR screens. Version 2 is the production
successor to the experimental Cortado implementation.

Statistical testing builds on [SCEPTRE](https://doi.org/10.1186/s13059-024-03254-2)
and [spaCRT](https://arxiv.org/abs/2407.08911). These statistical methods are prior
work; PerTurbo provides an implementation integrated with its analysis workflow.

## Installation

PerTurbo requires Python 3.11 or newer.

```bash
pip install perturbo
# or, for development in this checkout
uv sync --group test --group dev
```

For a compatible NVIDIA/CUDA 12 environment, install `perturbo[cuda]`.

## Container image

Build the GPU-ready image locally with:

```bash
docker build --tag perturbo:local .
docker run --rm --gpus all perturbo:local --help
```

The image uses JAX's CUDA 12 pip wheels, so the host must provide the NVIDIA
Container Toolkit and a Linux NVIDIA driver version 525 or newer. Do not set
`LD_LIBRARY_PATH` in the container: JAX uses its pip-installed CUDA libraries.
GitHub Actions builds the `linux/amd64` image. Version tags publish releases;
pushes to `v2-port` refresh the mutable `v2-dev` tag, and manual dispatches can
also publish. Pull requests build and smoke-test the image without publishing.

## Quick start

```python
import perturbo

perturbo.setup_mudata(
    mdata,
    modalities={"rna_layer": "rna", "perturbation_layer": "grna"},
    guide_by_element_key="element_targeted",
)

model = perturbo.PERTURBO(mdata, likelihood="nb", guide_random_effects=True)
model.train(steps=2500, batch_size=1024, accelerator="gpu")
model.save("perturbo_bundle", overwrite=True)
```

For reproducible file-based runs, use the CLI:

```bash
perturbo --input screen.h5mu --out-dir perturbo_outputs/run --modality-key rna \
  --perturbation-modality-key grna --perturbation-element-varm-key element_targeted
```

### Memory and chunking

Use `--backed` for files larger than memory. For mutually exclusive assignments,
perturbation chunks reduce both the number of cells and the number of fitted
effects. Splitting co-occurring perturbations would omit predictors and can bias
effects; when these assignments trigger automatic chunking, the CLI instead
loads 256 genes at a time and retains every cell and predictor. Set
`--gene-chunk-size 128` to choose a smaller block explicitly.

Gene blocks currently support the plain NB model with observed or fixed-zero
size factors, shared guide effects, no latent factors, no guide random effects,
and no perturbation dispersion or baseline uncertainty propagation. Full-panel
library sizes and the control centering are preserved across blocks.
Stage-two SVI defaults to 1,024 cells per step
on this path; use `--minibatch-size-betas` and `--num-epochs-betas` to control
training coverage. Stochastic fits at different block widths need not be
numerically identical after a finite number of steps.

For either chunking strategy, compare effect estimates with a longer training
budget. The default 500 steps can underestimate strong knockdowns.

Gene blocks bound the count and likelihood buffers, but control fitting still
loads up to `--max-control-cells` across all genes, and the final effect
tables scale as perturbations × genes. A full atlas is therefore a cluster job.
For a laptop smoke test, load a raw-count AnnData with `backed="r"`, select
controls and a few perturbations, save full-panel cell totals in an observation
column, then select a few hundred genes and pass that column with
`--library-size-key`. A small debug run checks execution, not full-scale speed,
convergence, or statistical calibration.

The main Python entry points are `PERTURBO` / `PerTurboModel`, `fit_from_path`,
`setup_mudata`, `fit_control`, `fit_perturbation_effects`, and the posterior
table and trained-model simulation helpers.

## Migrating from PyTorch PerTurbo and Cortado

The deprecated PyTorch/Pyro implementation is available only through the
optional `perturbo[legacy]` extra and `perturbo.legacy` namespace. It is not
loaded by a normal PerTurbo import.

| Previous surface | PerTurbo 2 surface |
| --- | --- |
| `cortado.PERTURBO` | `perturbo.PERTURBO` |
| `cortado.CortadoModel` | `perturbo.PerTurboModel` |
| `cortado` CLI | `perturbo` CLI |
| PyTorch `perturbo.PERTURBO` | `perturbo.legacy.PERTURBO` |

PerTurbo reads Cortado MuData registrations and fit bundles, warns once during
the upgrade, and writes the v2 `_perturbo_setup` and bundle format thereafter.

## Development

```bash
uv sync --group test --group dev
uv run pytest
uv build
```

The v2 production source is ported from the Cortado repository, which remains the
home for experiments, benchmarks, notebooks, apps and paper analyses. What ships
here is the analysis package: the models, the two-stage fit,
the result tables, preprocessing and the simulation entry
points. The benchmark harness, the evaluation scorers, the Streamlit applications
and the research diagnostics stay in Cortado, which is why a default install needs
neither statsmodels nor scikit-learn.
