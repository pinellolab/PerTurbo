<!-- [![Tests][badge-tests]][link-tests] -->

# 🏎️ PerTurbo: Fast analysis of single-cell perturbation studies

**PerTurbo** is a scalable analysis tool for Perturb-seq and similar large single-cell CRISPR screens (e.g. CROP-seq, ECCITE-seq). PerTurbo performs a Bayesian differential expression analysis on the raw count data which accounts for the sparsity of single-cell data and the variability in targeting efficiency across CRISPR sgRNA constructs ("guides").

## Installation


### pip (recommended)
Create a virtual environment with Python version 3.10 or later using [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html).

Then, with the environment activated, clone this repo and install it using `pip`.

```
git clone https://github.com/pinellolab/PerTurbo.git
cd PerTurbo
pip install -e .
```

## Examples

See the [notebooks](https://github.com/pinellolab/PerTurbo/tree/main/docs/notebooks) folder for some tutorial examples.