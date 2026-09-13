# PerTurbo

PerTurbo analyzes Perturb-seq count data to estimate how each targeted gene,
enhancer, or other element changes gene expression. A Bayesian model provides
effect estimates and uncertainty; a conditional randomization test (CRT)
provides complementary frequentist p-values and multiple-testing correction.
Both use JAX for CPU or NVIDIA GPU computation.

The v2 workflow accepts AnnData and MuData, supports low- and high-MOI screens,
and uses gene blocks to retain co-occurring perturbation predictors while
bounding expression buffers. Normal installation uses NumPyro/JAX and does not
require PyTorch, Pyro, scvi-tools, or pertpy.

## Start here

- [Install PerTurbo](docs/installation.md) and verify CPU/GPU availability.
- [Run your first analysis](docs/quickstart.md) on a small synthetic screen.
- [Prepare real data](docs/data_preparation.md): raw counts, called guides, controls, and mappings.
- [Choose run settings](docs/running_analyses.md): covariates, CRT pools, gene blocks, and fitting budgets.
- [Understand the method](docs/method.md) and [interpret results](docs/results_guide.md).
- [Use the Python workflow](docs/python_api.md) or consult the [API reference](docs/api.md).
- [Troubleshoot a run](docs/troubleshooting.md).

These docs describe the v2 release candidate. As checked on 13 September 2026,
PyPI publication is pending. The tagged source can be installed with:

```bash
python -m pip install "perturbo @ https://github.com/pinellolab/PerTurbo/archive/refs/tags/v2.0.0rc6.zip"
```

See the installation guide for an isolated environment, CUDA support, and the
current tag/package-version mismatch. The documentation website is also pending;
the guides are available in this repository meanwhile.

## A file-based analysis

For a prepared MuData containing RNA counts and called guides:

```bash
perturbo --input screen.h5mu --out-dir results \
  --modality-key rna --perturbation-modality-key grna \
  --perturbation-element-varm-key element_targeted \
  --perturbation-element-names-uns-key element_names \
  --control-substring non-targeting --library-size-key library_size
```

The CLI defaults to full-batch fitting and a propensity saddlepoint CRT for
compatible models. For high-MOI memory control, use gene blocks; the [run guide](docs/running_analyses.md)
explains supported combinations. Verify the selected CRT pool and compare a
longer fitting budget before interpreting a real screen. Bayesian effect
uncertainty and CRT p-values are distinct quantities, and calibration depends
on the design and model assumptions.

## Development and compatibility

See [Contributing](docs/contributing.md), [release notes](CHANGELOG.md), and the
[scverse-readiness assessment](docs/scverse_readiness.md).

```bash
uv sync --group test --group dev
uv run pytest
uv build
```

The deprecated PyTorch implementation is isolated under `perturbo.legacy` and
requires the optional `legacy` extra. Cortado registrations and fit bundles are
read with compatibility handling and written in the PerTurbo v2 format.
