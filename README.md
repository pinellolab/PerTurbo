# PerTurbo

PerTurbo is a NumPyro/JAX implementation of Bayesian Perturb-seq analysis for
low- and high-MOI single-cell CRISPR screens. Version 2 is the production
successor to the experimental Cortado implementation.

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
The established GitHub Actions recipe builds the `linux/amd64` image and
publishes it to GHCR only for version tags (or an explicit manual dispatch).

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

For a CIS-only or otherwise preselected analysis, pass a CSV, TSV, or Parquet
table whose required columns are `element` and `gene`:

```bash
perturbo --input screen.h5mu --out-dir perturbo_outputs/cis --modality-key rna \
  --perturbation-modality-key grna --perturbation-element-varm-key element_targeted \
  --pairs-to-test cis_pairs.parquet --minibatch-size-betas 1024
```

Only those exact coefficients are sampled; the pair list is not expanded to
an element-by-gene Cartesian product. PerTurbo subsets the RNA genes and
perturbation elements to the pair-list union before transfer to JAX. A beta
minibatch uses one global compiled fit by default. Use
`--perturbation-chunk-size` explicitly only when the full design does not fit
in device memory, since each distinct chunk shape can require compilation.

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

The v2 production source was ported from Cortado commit `efc923e`. The Cortado
repository remains the home for experiments, benchmarks, notebooks, apps, and
paper analyses.
