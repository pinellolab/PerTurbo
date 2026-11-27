# Analysis and power tutorials

Start with the observed screen in either row. Then use its fitted baseline in the matching simulation notebook to vary effect size and cells per perturbation.

| Screen | Fit the observed data | Simulate a power grid |
| --- | --- | --- |
| Gasperini K562 enhancer screen | [High-MOI analysis](gasperini.ipynb) · 8,000 cells, 100 elements, 256 genes, 387 cis candidates | [Cis detection and locus-level power](gasperini_power.ipynb) |
| Replogle K562 essential-wide screen | [Low-MOI analysis](replogle.ipynb) · 6,400 targeting cells, 1,024 controls, 100 targeting constructs, 256 genes | [Trans detection and regulon power](replogle_power.ipynb) |

The two count inputs total about 5 MiB. Both retain the deposited cell identifiers, raw counts, batch annotations and full-library UMI totals. The teaching panels use the same study designs as the manuscript on smaller test families and shorter fitting schedules. On one Apple Silicon laptop, the observed-data fits took about two minutes for Gasperini and three minutes for Replogle. The subsequent full teaching grids took about three and a half minutes and one and three-quarter minutes, respectively. Hardware, thread settings and concurrent workloads affect these times.

## Run the notebooks

From the repository root:

```bash
uv run --with jupyterlab --with ipykernel jupyter lab docs/tutorials
```

Use the kernel from the same environment as PerTurbo, with the notebook working directory set to `docs/tutorials/`. Run the analysis notebook first. It writes `runs/gasperini/fitted.h5mu` or `runs/replogle/fitted.h5mu`, then the corresponding power notebook consumes that file. The analysis notebook reuses a completed fit when its input digest and command match. To try another fit, use a fresh output directory.

## Rebuild the teaching subsets

The notebooks run with the supplied files. To inspect or change the subsetting, run {download}`prepare_inputs.py <prepare_inputs.py>` from the repository root as shown below. The script preserves the original cell IDs, raw counts, library sizes and batch annotations. Changing the cells, genes or constructs changes the fit and the family over which multiple testing is corrected.

**Gasperini et al. (2019).** The source is the K562 at-scale CRISPRi screen. The extract samples cells uniformly with seed 12. It includes the PPP2R5A neighborhood and a candidate enhancer near LMO2, then fills the panel with mapped candidate enhancers, TSS controls and non-targeting elements. The builder uses the candidate keys bundled in `examples/manuscript/gasperini-cis-candidates.parquet`. It retains the original full-screen guide count even though most guide columns are omitted. A cell lacking a retained guide can still carry other guides in the original screen. If you already have the full `GSE120861_at_scale_screen.h5mu` with `gene` and `guide` modalities, use the second command directly. [GEO deposit GSE120861](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120861)

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
uv run python docs/tutorials/prepare_inputs.py gasperini geo/GSE120861_at_scale_screen.h5mu gasperini-rebuilt.h5mu
```

**Replogle et al. (2022).** The extract derives from the deposited raw single-cell expression matrix for the day-six K562 essential-wide screen. It samples 100 targeting constructs with at least 64 cells and measured on-target genes, keeping 64 cells per construct and 1,024 cells sampled from all non-targeting constructs. The response panel includes their on-target genes, the teaching GATA1 regulon (`docs/tutorials/inputs/gata1_targets.txt`) and additional measured genes. Selection uses labels, construct cell counts and a fixed seed, without filtering on screen DE results. The source file is approximately 10.7 GB. The dataset is distributed under CC BY 4.0 by Joseph Replogle and Jonathan Weissman. [Figshare dataset](https://doi.org/10.25452/figshare.plus.20029387.v1) · [Raw K562 essential-wide h5ad](https://ndownloader.figshare.com/files/35773219)

```bash
uv run python docs/tutorials/prepare_inputs.py replogle K562_essential_raw_singlecell_01.h5ad replogle-rebuilt.h5ad
```

To use a rebuilt file, set `source` in the notebook's first code cell to its path.

```{toctree}
:hidden:

gasperini
gasperini_power
replogle
replogle_power
```
