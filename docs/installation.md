# Installation

Use a separate Python environment for PerTurbo so its numerical dependencies
do not conflict with another analysis. Python 3.11 or newer is required;
Python 3.12 is a useful starting point because it is included in the project's
test configuration.

## Install the current release candidate

As checked on 13 September 2026, `perturbo` is not yet available from PyPI.
Install the tagged source snapshot below. The tag identifies the code used
by this draft; it is preferable to installing a moving development branch
when you need to reproduce an analysis.

In a Linux or macOS terminal:

```bash
python3.12 -m venv perturbo-env
source perturbo-env/bin/activate
python -m pip install --upgrade pip
python -m pip install "perturbo @ https://github.com/pinellolab/PerTurbo/archive/refs/tags/v2.0.0rc6.zip"
```

If your cluster supplies Python through environment modules or a managed
environment, use its supported Python environment instead of the first two
commands. On Windows, environment activation differs; the package's automated
test configuration currently targets Linux, and Linux is the documented GPU
route.

```{note}
The `v2.0.0rc6` tag currently contains package metadata reporting `2.0.0rc5`.
Until release metadata is synchronized, record both the tag and the reported
package version. The [release-readiness assessment](scverse_readiness.md)
tracks this and the pending registry publication.
```

## Use an NVIDIA GPU

On a compatible Linux NVIDIA system, replace the last installation command with:

```bash
python -m pip install "perturbo[cuda] @ https://github.com/pinellolab/PerTurbo/archive/refs/tags/v2.0.0rc6.zip"
```

The `cuda` extra installs JAX's CUDA 12 dependencies. JAX currently documents a
Linux NVIDIA driver minimum of 525 for CUDA 12; ask your cluster administrator
which supported setup to use if the driver or library setup is uncertain.
Installing a Python package does not install the host's NVIDIA driver.
See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for current compatibility details.

An Apple Silicon laptop can run the small tutorial on CPU. JAX's Apple GPU
support is experimental and is not the GPU path documented for PerTurbo here.
Large screens belong on an appropriately provisioned compute node; fitting
a small tutorial does not establish that a full screen will fit in laptop memory.

## Verify the installation

```bash
perturbo --help
python -c "import perturbo, jax; print('PerTurbo:', perturbo.__version__); print('Devices:', jax.devices())"
```

CPU devices are sufficient for [Your first analysis](quickstart.md). For GPU
runs, perform the device check **inside the GPU job allocation**, not on the
cluster login node. Confirm that a GPU appears, then request `--device gpu` in
the analysis command. A CUDA installation in a login shell does not guarantee
that a submitted job has a GPU assigned.

The first fit includes compilation. Its first progress update can therefore
take longer than later updates. See [Troubleshooting](troubleshooting.md) for
device discovery, memory, and slow-start issues.

## What is installed

The normal package uses JAX/NumPyro for inference, AnnData/MuData for data, and
NumPy/SciPy/pandas/PyArrow for numerical work and tables. It does not require
scvi-tools, pertpy, PyTorch, or Pyro for the v2 workflow. You can use those
packages elsewhere in your analysis without making them part of a minimal
PerTurbo installation.

The optional `legacy` extra is for the deprecated PyTorch implementation.
It is not needed for these tutorials. The `docs` extra installs documentation
build tools and is also unnecessary for analysis.

## Containers and development checkouts

The repository includes a Dockerfile for a Linux GPU-capable image:

```bash
git clone --branch v2.0.0rc6 https://github.com/pinellolab/PerTurbo.git
cd PerTurbo
docker build --tag perturbo:local .
docker run --rm --gpus all perturbo:local --help
```

GPU containers need a compatible host driver and NVIDIA Container Toolkit.
Consult the cluster's container policy; many clusters use Apptainer rather
than Docker. A working container does not remove the need to allocate a GPU.

For development, follow [Contributing](contributing.md). After installation,
run `python -m pip freeze > environment.txt` and retain the environment record
with the command and results from a real experiment.
