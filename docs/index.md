# PerTurbo

Estimate perturbation effects on gene expression, identify differentially expressed genes, and inspect guide efficiencies in Perturb-seq screens. The tutorials walk from raw counts and guide assignments to effect estimates and adjusted p-values.

- For one construct per cell, start with the [Replogle K562 essential-wide tutorial](tutorials/replogle.ipynb).
- For several guides per cell, start with the [Gasperini enhancer screen tutorial](tutorials/gasperini.ipynb).
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
