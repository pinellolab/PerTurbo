<!-- [![Tests][badge-tests]][link-tests] -->

# 🏎️ PerTurbo: Fast analysis of single-cell perturbation studies

**PerTurbo** is a scalable Bayesian analysis tool for Perturb-seq and similar large single-cell CRISPR screens (e.g. CROP-seq, ECCITE-seq). PerTurbo performs count-based regression to estimate perturbation effects on each gene based on a hierarchical statistical model which accounts for the sparsity of single cell RNA-seq data and variability in efficiency between different CRISPR sgRNA constructs ("guides"). Under the hood, PerTurbo relies on PyTorch and Pyro to perform GPU-accelerated stochastic variational inference, which unlocks fast and robust transcriptome-wide analyses of millions of cells and thousands of perturbations in both low- and high-MOI screens.

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