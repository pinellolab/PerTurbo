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

### The conditional randomization test

The conditional randomization test runs alongside the Bayesian effect estimates
by default (`--no-crt` opts out). It asks whether a gene's expression differs by more than it would had
the guide landed in a different set of cells with the same covariates, and it
approximates its tail with a saddlepoint calculation without resampling, so a genome-scale
screen can be tested against every gene:

```bash
perturbo --input screen.h5mu --out-dir perturbo_outputs/run --modality-key rna \
  --perturbation-modality-key grna --perturbation-element-varm-key element_targeted \
  --crt --crt-mechanism propensity --crt-tail-families saddlepoint \
  --crt-saddlepoint-only --crt-polish-baseline
```

Add `--crt-only` to stop after the test and skip the effect estimates, which is
the cheaper path for calibration checks and power calculations. The test serves
both screen designs, chosen with `--crt-pool`: with one perturbation per cell
(`control-anchored`) each target is tested inside the control pool plus its own
cells, and with many perturbations per cell (`all-cells`) each element is tested
as a marginal association over all cells. The default, `auto`, measures the
design from the data: every cell is used when the median guides per cell
exceeds 3, the control pool otherwise, and the command line prints what it
measured and what it chose. Either way it reports how many cells carry only
control guides and warns when they are fewer than 1,000 or under 1%. A
low-MOI screen may arrive with a guide-to-element map; the control-anchored
test collapses the assignment to elements and sets aside cells carrying more
than one, reporting the count. See `docs/crt_quickstart.md`.

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
library sizes and the control centering are preserved across blocks. The
all-cells CRT reuses its propensity fit, and multiple-testing correction happens
once over the full tested family. Stage-two SVI defaults to 1,024 cells per step
on this path; use `--minibatch-size-betas` and `--num-epochs-betas` to control
training coverage. Stochastic fits at different block widths need not be
numerically identical after a finite number of steps.

For either chunking strategy, compare effect estimates with a longer training
budget. The default 500 steps can underestimate strong knockdowns; CRT baseline
polishing does not establish convergence of the Bayesian effect estimates.

Gene blocks bound the count and likelihood buffers, but control fitting still
loads up to `--max-control-cells` across all genes, and the final effect and CRT
tables scale as perturbations × genes. A full atlas is therefore a cluster job.
For a laptop smoke test, load a raw-count AnnData with `backed="r"`, select
controls and a few perturbations, save full-panel cell totals in an observation
column, then select a few hundred genes and pass that column with
`--library-size-key`. A small debug run checks execution, not full-scale speed,
convergence, or statistical calibration.

### Reporting a subset of pairs

A cis window, or any other preselected pair set, is a question about the
multiple-testing family rather than about the fit: the estimates and p-values for
a pair do not depend on which other pairs were requested. Pass a CSV, TSV or
Parquet table with columns `element` and `gene`:

```bash
perturbo --input screen.h5mu --out-dir perturbo_outputs/run --modality-key rna \
  --perturbation-modality-key grna --perturbation-element-varm-key element_targeted \
  --pairs-to-test cis_pairs.parquet
```

The run is unchanged; every pair is still fitted and tested. Beside
`element_effects.parquet` PerTurbo writes `element_effects_requested_pairs.parquet`,
holding the requested rows with Benjamini-Hochberg recomputed within that set. One
run therefore yields both a cis-scale comparison and the transcriptome-wide
analysis. In PerTurbo 2.0 this flag restricted the fit itself; it no longer does,
and the command line says so at startup.

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
here is the analysis package: the models, the two-stage fit, the conditional
randomization test, the result tables, preprocessing and the simulation entry
points. The benchmark harness, the evaluation scorers, the Streamlit applications
and the research diagnostics stay in Cortado, which is why a default install needs
neither statsmodels nor scikit-learn.
