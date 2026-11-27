# Contributing

Use Python 3.11+ and `uv` for local development.

```bash
uv sync --locked --group test --group dev
uv run --locked --group test pytest
uv run ruff check src tests
uv build
```

Keep production code under `src/perturbo`. Benchmark runners, applications,
exploratory notebooks, and manuscript analysis belong in the Cortado research repository.
The default test environment must not acquire Torch, Pyro, or scvi-tools; test
the deprecated implementation only with `uv sync --extra legacy --group test`.

## Where the source of truth lives

`src/perturbo` is a rename of the analysis package developed in the Cortado
research repository, and changes usually start there. When you edit shared
implementation files, make the same edit in both places; a change that lands
only here will be overwritten by the
next port. What this repository owns outright is the packaging: the Dockerfile,
the workflows, the legacy extra and its isolation test.

## Two constraints that are easy to break

The package enables float64 at import, but **the SVI parameters must stay
float32**. Under float64 the array
constructors and `AutoNormal`'s scale initialisation produce float64, the model
casts its design matrices to the parameter dtype, and every cells-by-genes
intermediate of the likelihood doubles; a screen that fitted on a 40 GB card
then does not. `_pin_svi_params_float32` enforces it after `svi.init` and
`tests/test_svi_stays_float32.py` guards it, so keep new initial values and
cell-axis arrays in float32.

**The step size and the step count are a pair.** Adam moves a coefficient by
roughly the step size per step, so their product must exceed the largest effect
in nats with margin. The defaults, 0.01 and 500 steps per stage, were chosen
against simulated screens with known effects; lowering one without raising the
other shrinks large effects toward zero.

## Containers and GPU installations

Install this checkout with `python -m pip install ".[cuda]"` on a compatible NVIDIA CUDA 12 host. The published rc12 container can be inspected with Docker or Apptainer:

```bash
docker run --rm --gpus all ghcr.io/pinellolab/perturbo:v2.0.0rc12 perturbo --help
apptainer pull perturbo.sif docker://ghcr.io/pinellolab/perturbo:v2.0.0rc12
apptainer exec --nv perturbo.sif perturbo --help
```

For an unpublished checkout, build the image locally with `docker build -t perturbo:local .`. A version tag or image digest identifies the software used for an analysis. The `v2-dev` image tracks changes on the integration branch. The image installs JAX's CUDA libraries; use the host driver and NVIDIA Container Toolkit to expose the GPU.

## Release validation

Run the default suite on Python 3.11 and 3.12 and the legacy suite separately with `uv run --locked --group test --extra legacy pytest tests/legacy`. Build the source archive and wheel, install the wheel into a fresh environment, and run the executable notebooks there. A clean base install should import PerTurbo without importing Torch or scvi-tools. The release wheel and source archive include the software license and retained notices.

For a strict local documentation build, run `uv run --locked --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html`.

## Publishing to PyPI

Publishing a GitHub release triggers `.github/workflows/release.yaml`. The job uses the GitHub environment `pypi` and requests an OIDC token. In the PerTurbo project's PyPI **Publishing** settings, register the GitHub publisher with owner `pinellolab`, repository `PerTurbo`, workflow filename `release.yaml`, and environment `pypi`. A pending publisher can create the project on its first successful upload. Choose a fresh package version for each release and check that it matches the Git tag. [PyPI Trusted Publishing setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)

## Development references

```{toctree}
:maxdepth: 1

high_moi_padding_handoff.md
propensity_tail_diagnostics.md
rc11_followup_validation.md
```
