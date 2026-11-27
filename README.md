# PerTurbo

PerTurbo estimates how CRISPR perturbations change gene expression in single cells. For each perturbed element and response gene, it reports an estimated natural-log fold change (lnFC) and a p-value. Screens with several guides per element can also estimate guide-specific efficiencies. The analysis runs on CPU or an NVIDIA GPU.

## Start with your data

| What do you have? | Where to start |
| --- | --- |
| RNA counts and one assigned construct per cell | [Replogle low-MOI notebook](docs/tutorials/replogle.ipynb). Compare targeted cells with labeled non-targeting controls. |
| RNA counts and multiple guide assignments per cell | [Gasperini high-MOI notebook](docs/tutorials/gasperini.ipynb). Map guides to elements, estimate their relative efficiencies, and test associations across all cells. |
| RNA and guide sequencing reads | [CRISPR_Pipeline input and samplesheet guide](https://github.com/pinellolab/CRISPR_Pipeline/blob/8061b88d871b8fa2720c8484906de9b6d9cd7330/README.md). Start with preprocessing and guide assignment. The pipeline has its own configuration for downstream inference. |

Both notebooks include small real datasets, complete commands, and plotted results. [Tutorial inputs and setup](docs/tutorials/index.md)

## Install

From this checkout, install the code and verify the command:

```bash
python -m pip install .
perturbo --help
```

Use Python 3.11 or 3.12. For NVIDIA CUDA 12 support, install `".[cuda]"` in place of `.`. The published reference version for these tutorials is [v2.0.0rc12](https://github.com/pinellolab/PerTurbo/releases/tag/v2.0.0rc12). To install that specific version:

```bash
python -m pip install "perturbo @ git+https://github.com/pinellolab/PerTurbo.git@v2.0.0rc12"
```

The named FDR columns, fitted MuData export, and relative-mode gene blocking described below are available in this development checkout. The tutorial commands for running a screen also work with rc12. [Container and contributor setup](docs/contributing.md)

## Run a screen

Use raw RNA counts. When you subset genes, retain the library sizes calculated from the full expression matrix. The following commands use the two inputs shipped with this checkout.

**One construct per cell.** In the Replogle example, `gene_transcript` identifies the perturbation and `non-targeting` identifies the control cells. `UMI_count` supplies the library size and `gem_group` identifies the batch.

```bash
perturbo --input docs/tutorials/inputs/replogle.h5ad --out-dir runs/replogle \
  --perturbation-key gene_transcript --control-substring non-targeting \
  --gene-name-key gene_name --library-size-key UMI_count --batch-covariate gem_group \
  --crt-pool control-anchored --guide-effect-strategy shared \
  --num-steps-control 2500 --num-steps-betas 1000 --step-size 0.01
```

**Multiple guides per cell.** The Gasperini input stores binary assignments in the `guide` modality. Its guide-by-element matrix identifies guides aimed at the same element, and its ordered element names identify the matrix columns. The relative fit estimates an element effect and a guide multiplier for each response gene.

```bash
perturbo --input docs/tutorials/inputs/gasperini.h5mu --out-dir runs/gasperini \
  --modality-key gene --perturbation-modality-key guide --gene-name-key symbol \
  --perturbation-element-varm-key guide_intended_target_pairs \
  --perturbation-element-names-uns-key intended_targets \
  --library-size-key total_umis --continuous-covariates total_umis percent_mito \
  --batch-covariate prep_batch --crt-pool all-cells --guide-effect-strategy relative \
  --num-steps-control 2500 --num-steps-betas 1000 --step-size 0.01
```

For a large fit, `--gene-chunk-size` can limit the number of response genes fitted together in stage two. See `perturbo --help` for supported modes and memory controls.

## Read the results

```python
import mudata

fit = mudata.read_h5mu("runs/gasperini/fitted.h5mu")
effects = fit.uns["_perturbo"]["effects"]
hits = effects.loc[effects["q_global"] < 0.1, ["element", "gene", "posterior_mean", "q_global"]]
print(hits.sort_values("q_global").head())
```

A `posterior_mean` of −0.69 corresponds to roughly half the expression. In relative mode it is the parent element effect. The `efficiency` table reports the guide multiplier in `guide_relative_efficiency_mean`. Match the tables by **both element and response gene** to see the guides for a hit. Shared mode fixes the multiplier to one.

With `--report-low-information`, the effects table also reports how many assigned cells expressed the gene (`observed_nonzero`) and how many the null model expected (`expected_nonzero`). For tested pairs, `low_information` is true when both counts fall below the threshold (5 by default). This annotation leaves the p-value and BH correction unchanged.

Choose the correction family that matches the question. All three use Benjamini–Hochberg on finite values of the same primary `p_value`.

| Question | Column | Tests corrected together |
| --- | --- | --- |
| Which associations can I report across my screen? | `q_global` | All tested element–gene pairs |
| Which genes respond to this specific perturbation? | `q_element` | Tested response genes for that element |
| Which of my predefined local associations are detected? | `q_cis` | All tested pairs in the supplied cis list, pooled across elements |

To request `q_cis`, supply the complete candidate set with `--cis-pairs cis.tsv` (`element` and `gene` columns, using the fitted IDs). Other rows have missing `q_cis`. The column is omitted when no candidate has a finite test. The metadata records the primary test and family sizes. Filtering a result table preserves its named q-values. In rc12 saddlepoint-only runs, use `crt_saddlepoint_q_value` for screen-wide correction.

The default output is a self-contained `fitted.h5mu`. Use `--output-format bundle` for a compact directory linked to the original assays. See [loading a fit and relocating its input](docs/fitted_export_pilot.md).

## License

PerTurbo is licensed under [GPL-3.0-or-later](LICENSE). [NOTICE](NOTICE) records the retained MIT and NumPyro attributions. Tutorial datasets are credited to their original providers in the [tutorial guide](docs/tutorials/index.md).
