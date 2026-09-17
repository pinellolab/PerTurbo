# PerTurbo

PerTurbo is a NumPyro/JAX implementation of Bayesian Perturb-seq analysis for
low- and high-MOI single-cell CRISPR screens. Version 2 is the production
successor to the experimental Cortado implementation.

Statistical testing builds on [SCEPTRE](https://doi.org/10.1186/s13059-024-03254-2)
and [spaCRT](https://arxiv.org/abs/2407.08911). These statistical methods are prior
work; PerTurbo provides an implementation integrated with its analysis workflow.

## Installation

PerTurbo requires Python 3.11 or newer. It is **not published on PyPI**; install
it from GitHub, from a container image, or from a checkout.

```bash
# from GitHub, pinned to a release tag
pip install "perturbo @ git+https://github.com/pinellolab/PerTurbo.git@v2.0.0rc9"

# same, with JAX's CUDA 12 wheels for an NVIDIA GPU
pip install "perturbo[cuda] @ git+https://github.com/pinellolab/PerTurbo.git@v2.0.0rc9"

# from a checkout
git clone https://github.com/pinellolab/PerTurbo.git
cd PerTurbo && uv sync
```

Pin a tag rather than a branch: `v2-port` moves, and the mutable `v2-dev`
container tag follows it.

## Container image

Released images are published to GHCR, which is usually easier than installing
CUDA wheels yourself:

```bash
docker pull ghcr.io/pinellolab/perturbo:v2.0.0rc9
docker run --rm --gpus all ghcr.io/pinellolab/perturbo:v2.0.0rc9 --help
```

On a cluster without Docker:

```bash
apptainer pull perturbo.sif docker://ghcr.io/pinellolab/perturbo:v2.0.0rc9
apptainer exec --nv perturbo.sif perturbo --help
```

Pin a version tag, or a digest, rather than `v2-dev` or `latest`, both of which
move. To build the image locally instead:

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

The command line is the supported way to run a screen. It takes an `.h5ad` or
`.h5mu` file, writes every result to a directory, and records what it did.

### A low-MOI screen in an AnnData

One perturbation per cell, named by a column of `obs`:

```bash
perturbo --input screen.h5ad --out-dir results/ \
  --perturbation-key perturbation \
  --control-substring non-targeting \
  --library-size-key total_counts \
  --batch-covariate batch
```

`--perturbation-key` names the `obs` column holding each cell's perturbation,
and `--control-substring` matches the control labels within it. The CRT runs by
default and resamples each perturbation against the control cells plus its own.

### A high-MOI screen in a MuData

Several perturbations per cell, in a second modality:

```bash
perturbo --input screen.h5mu --out-dir results/ \
  --modality-key rna \
  --perturbation-modality-key grna \
  --perturbation-layer guide_assignment \
  --control-substring non-targeting \
  --library-size-key total_counts \
  --batch-covariate batch
```

`--perturbation-layer` should name the binary guide-by-cell assignment, not raw
guide counts: with raw counts nearly every cell looks multiply assigned. To test
elements rather than individual guides, add
`--perturbation-element-varm-key element_targeted` (a guide-by-element indicator
in `varm`) and `--perturbation-element-names-uns-key element_names`.

### Useful defaults, and what to override

`--crt` is on, `--size-factor-mode` is `observed`, and `--crt-pool` is `auto`,
which reads the assignments and picks `control-anchored` for a low-MOI screen
and `all-cells` for a high-MOI one. Set `--crt-pool` explicitly when the
declared design and the assignments disagree, or when the control cells are
concentrated in one batch. Add `--pairs-to-test pairs.parquet` (columns
`element,gene`) to have a second table q-corrected over just those pairs; the
fit and the test still cover every pair. `--no-crt` fits effects alone.

### What a run writes

| file | contents |
| --- | --- |
| `element_effects.parquet` | one row per element-gene pair: effect size, CRT p-value and q-value |
| `element_effects_requested_pairs.parquet` | the same rows for `--pairs-to-test`, q-corrected within that set |
| `crt_metadata.json` | the pool used and why, and the counts behind it |
| `covariate_metadata.json` | the covariates and batch levels the design carried |
| `control_fit.npz` | the stage-one control fit |
| `guide_efficiency.parquet` | per-guide efficiency, when guide effects are estimated |

### Python interface

For embedding PerTurbo in a larger analysis, the same fit is available in
Python. Prefer the CLI for whole screens: it handles chunking, records its
configuration, and is reproducible from a command line.

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

## License

PerTurbo is distributed under the GNU General Public License, version 3 or
any later version (GPL-3.0-or-later). See [LICENSE](LICENSE) for the full terms
and [NOTICE](NOTICE) for retained copyright and third-party license notices.
