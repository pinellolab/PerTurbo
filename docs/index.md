# PerTurbo

Estimate perturbation effects on gene expression, identify differentially expressed genes, and inspect guide efficiencies in Perturb-seq screens. Each tutorial starts with an observed screen. Its companion notebook uses the fitted baseline to simulate recovery across effect sizes and cell budgets.

- For one construct per cell, fit the [Replogle K562 essential-wide screen](tutorials/replogle.ipynb), then explore [GATA1 regulon power](tutorials/replogle_power.ipynb).
- For several guides per cell, fit the [Gasperini enhancer screen](tutorials/gasperini.ipynb), then explore [cis and locus-level power](tutorials/gasperini_power.ipynb).
- For sequencing reads, start with the [CRISPR_Pipeline samplesheet and preprocessing guide](https://github.com/pinellolab/CRISPR_Pipeline/blob/8061b88d871b8fa2720c8484906de9b6d9cd7330/README.md).
- For an existing fit, see [reading results and exporting fitted MuData](fitted_export_pilot.md).

```{toctree}
:maxdepth: 1

tutorials/index.md
api.md
fitted_export_pilot.md
changelog.md
contributing.md
references.md
```
