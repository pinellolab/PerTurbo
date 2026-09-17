# 🏎️ PerTurbo

## Robust GPU-accelerated differential expression for Perturb-seq

**PerTurbo** delivers fast and accurate differential expression analyses for
large-scale Perturb-seq experiments of up to millions of cells and thousands of CRISPR perturbations.
It uses GPU accelerated routines for both Bayesian effect size estimation and 
frequentist statistical testing to obtain fast and accurate inferences based on
count-based regression models.

Specifically, we use stochastic variational inference from NumPyro to obtain
posterior estimates of perturbation effect sizes based on a Bayesian generalized linear model,
we obtain calibrated frequentist p-values using a custom JAX implementation of the score-based 
conditional randomization test, using a saddlepoint approximation to obtain calibrated tail probabilities
based on the statistical approaches proposed in (Barry et al. 2024) and (Niu et al. 2024).

## Installation

PerTurbo requires Python 3.11 or newer. It is **not yet published on PyPI**; install
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

One perturbation per cell, named by a column of `adata.obs`:

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


### Outputs

| file | contents |
| --- | --- |
| `element_effects.parquet` | one row per element-gene pair: effect size, CRT p-value and q-value |
| `element_effects_requested_pairs.parquet` | the same rows for `--pairs-to-test`, q-corrected within that set |
| `crt_metadata.json` | the pool used and why, and the counts behind it |
| `covariate_metadata.json` | the covariates and batch levels the design carried |
| `control_fit.npz` | the stage-one control fit |
| `guide_efficiency.parquet` | per-guide efficiency, when guide effects are estimated |


## License

PerTurbo is distributed under the GNU General Public License, version 3 or
any later version (GPL-3.0-or-later). See [LICENSE](LICENSE) for the full terms
and [NOTICE](NOTICE) for retained copyright and third-party license notices.

## Manuscript
Coming soon!
