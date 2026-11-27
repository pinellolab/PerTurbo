# Analysis tutorials

Run PerTurbo on real K562 count data, from checking guide assignments to reading effect estimates and adjusted p-values. Both notebooks include executed plots and come with compact, reproducible inputs.

| Notebook | Cells | Design | What you can inspect |
| --- | ---: | --- | --- |
| [Gasperini enhancer screen](gasperini.ipynb) | 12,000 | Multiple guides per cell, two guides per selected element | Candidate enhancer and promoter effects on LMO2, plus guide-specific efficiencies |
| [Replogle essential-wide screen](replogle.ipynb) | 2,297 | One construct per cell, including 1,275 non-targeting cells | On-target knockdown and responses of other genes across three constructs |

## Run a notebook

From the root of this PerTurbo checkout, launch JupyterLab and open either notebook. Choose the Python kernel from the same environment. The inputs are in the adjacent `inputs/` directory (3.4 MiB together). Running all cells fits the model on CPU and writes the results under `runs/gasperini/` or `runs/replogle/`.

```bash
uv run --with jupyterlab --with ipykernel jupyter lab docs/tutorials
```

For a local Jupyter setup, you can also download the {download}`Gasperini notebook <gasperini.ipynb>` and {download}`Replogle notebook <replogle.ipynb>`. Put the {download}`Gasperini MuData <inputs/gasperini.h5mu>` and {download}`Replogle AnnData <inputs/replogle.h5ad>` in an adjacent `inputs/` folder and use a kernel with PerTurbo installed.

Both examples open `fitted.h5mu` and select associations from the `effects` DataFrame in `mdata.uns["_perturbo"]`. Gasperini also has an `efficiency` DataFrame for guide-level estimates.

## Rebuild the teaching subsets

The notebooks run with the supplied files. To inspect or change the subsetting, use {download}`prepare_inputs.py <prepare_inputs.py>` from this directory. The script preserves the original cell IDs, raw counts, library sizes and batch annotations. Changing the cells, genes or constructs changes the fit and the family over which multiple testing is corrected.

**Gasperini et al. (2019).** The source is the K562 at-scale CRISPRi screen. The extract contains a seeded random sample of cells and a fixed panel of promoter, enhancer and non-targeting guides. If you already have the full `GSE120861_at_scale_screen.h5mu` with `gene` and `guide` modalities, use the second command directly. [GEO deposit GSE120861](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120861)

To build the full MuData from GEO, place the following at-scale supplementary files in one directory. Keep the matrix, genes, cell metadata and guide groups compressed. Decompress the gene–guide pair table to the filename below.

```text
GSE120861_at_scale_screen.exprs.mtx.gz
GSE120861_at_scale_screen.genes.txt.gz
GSE120861_at_scale_screen.phenoData.txt.gz
GSE120861_grna_groups.at_scale.txt.gz
GSE120861_gene_gRNAgroup_pair_table.at_scale.txt
```

```bash
uv run python -c 'from perturbo.preprocessing import write_gasperini_geo_h5mu; write_gasperini_geo_h5mu("geo", "at_scale")'
uv run python prepare_inputs.py gasperini geo/GSE120861_at_scale_screen.h5mu gasperini-rebuilt.h5mu
```

**Replogle et al. (2022).** The extract derives from the deposited raw single-cell expression matrix for the day-six K562 essential-wide screen. It retains three targeted genes and the six most abundant non-targeting construct labels, with six response genes. The source file is approximately 10.7 GB. The dataset is distributed under CC BY 4.0 by Joseph Replogle and Jonathan Weissman. [Figshare dataset](https://doi.org/10.25452/figshare.plus.20029387.v1) · [Raw K562 essential-wide h5ad](https://ndownloader.figshare.com/files/35773219)

```bash
uv run python prepare_inputs.py replogle K562_essential_raw_singlecell_01.h5ad replogle-rebuilt.h5ad
```

To use a rebuilt file, set `source` in the notebook's first code cell to its path.

```{toctree}
:hidden:

gasperini
replogle
```
