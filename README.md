<!-- [![Tests][badge-tests]][link-tests] -->

# 🏎️ PerTurbo: Fast analysis of single-cell perturbation studies

**PerTurbo** is a scalable analysis tool for Perturb-seq and similar large single-cell CRISPR screens (e.g. CROP-seq, ECCITE-seq). PerTurbo performs a Bayesian differential expression analysis on the raw count data which accounts for the sparsity of single-cell data and the variability in targeting efficiency across CRISPR sgRNA constructs ("guides"). Under the hood, PerTurbo uses PyTorch and Pyro for GPU-accelerated stochastic variational inference to achieve scalability to millions of cells and thousands of perturbations in both low- and high-MOI screens.

## Installation

### pip (recommended)
To install PerTurbo, clone this repo and create an editable local installation using `pip`.

```
git clone https://github.com/pinellolab/PerTurbo.git
cd PerTurbo
pip install -e .
```

## Examples

See the [notebooks](https://github.com/pinellolab/PerTurbo/tree/main/docs/notebooks) folder for some tutorial examples.