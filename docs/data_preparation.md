# Prepare your data

PerTurbo needs two kinds of measurements for the **same cells**: raw gene
counts and the perturbations assigned to each cell. Negative-control labels
and measured technical covariates tell it how the experiment was designed.
Finish read processing, cell filtering, and guide calling before this step.

## Choose an input layout

| Your data | Suitable layout | How the CLI identifies perturbations |
| --- | --- | --- |
| Exactly one perturbation label per cell | AnnData `.h5ad` | An `.obs` column, selected with `--perturbation-key` |
| A cell-by-guide assignment matrix, especially with multiple guides per cell | MuData `.h5mu` with RNA and guide modalities | A guide modality plus a guide-to-element map |

Use the MuData layout when retaining individual guide identities matters.
Do not turn a high-MOI cell's combination of guides into one categorical label:
that would describe a different set of predictors from the individual targets.

### Expression counts

The RNA modality's `.X` must contain finite, nonnegative integer-valued
expression counts. Floating storage is acceptable only if the values really
are integers. Log-normalized values, scaled expression, imputed values, and
Pearson residuals are not the input to the count likelihood.

The CLI reads RNA counts from `.X`. If your raw counts are in a layer, create
an analysis object with that layer in `.X` before saving. For an object small
enough to copy into memory:

```python
import anndata as ad

adata = ad.read_h5ad("processed_small_screen.h5ad")
counts_adata = ad.AnnData(
    X=adata.layers["counts"].copy(),
    obs=adata.obs.copy(),
    var=adata.var.copy(),
)
counts_adata.write_h5ad("count_screen.h5ad")
```

The name `counts` has no special meaning: verify that this layer contains
the original counts. Avoid copying a full atlas just to rearrange its storage;
prepare large input files in your existing preprocessing pipeline.

### One label per cell in AnnData

For a low-MOI screen, `adata.obs["perturbation"]` might contain
`non-targeting`, `target_A`, or `target_B`. Each value represents an element
or guide label to analyze. Missing labels are not an appropriate encoding
of known negative controls.

```bash
perturbo --input count_screen.h5ad --out-dir results \
  --perturbation-key perturbation --control-substring non-targeting
```

If two guides targeting the same element are stored as two different labels,
this layout treats them as different predictors. Use a guide-to-element map
when they should share an element effect.

## RNA and guides in MuData

The recommended guide-resolved layout is:

```text
mdata
  rna.X                         cells × genes: raw expression counts
  rna.obs                       cell metadata and full-panel library sizes
  rna.var_names                 gene identifiers
  grna.X                        cells × guides: called guide assignments
  grna.var_names                guide identifiers
  grna.varm["element_targeted"]  guides × elements: mapping
  grna.uns["element_names"]      element names in mapping-column order
```

Names such as `rna`, `grna`, and `element_targeted` are examples, not required
names. Pass the matching names as CLI flags. The [first tutorial](quickstart.md)
constructs a complete small object with exactly this layout.

### Guide assignments and mapping

Use a sparse binary matrix for called guide presence: `1` means the guide was
assigned to the cell, and `0` means it was not. Raw guide UMI counts are not a
substitute for a guide-calling decision. If called assignments are in a guide
layer, select it with `--perturbation-layer`.

For the standard mapped workflow, assign each guide to at most one element.
Several guides may target the same element. In the shared-effect model a cell
carrying two guides for the same element contributes one element exposure;
the element is not counted twice.

Include named negative-control columns in the guide and element metadata.
The `--control-substring` must identify those controls without also matching
real targets. Keep unrelated targets distinct. Give each control group a
deliberate name so you can inspect its calibration later.

Here is a map for two controls and two guides per target:

```python
import numpy as np
import scipy.sparse as sp

# grna.var_names must already be in this order:
# non-targeting_1, non-targeting_2, target_A_1, target_A_2, target_B_1, target_B_2
parent = np.array([0, 0, 1, 1, 2, 2])
grna.varm["element_targeted"] = sp.csr_matrix(
    (np.ones(6, dtype=np.float32), (np.arange(6), parent)),
    shape=(6, 3),
)
grna.uns["element_names"] = np.array(
    ["non-targeting", "target_A", "target_B"]
)
```

A sparse `.varm` matrix has no column labels. The `.uns` array supplies them,
so pass **both** `--perturbation-element-varm-key element_targeted` and
`--perturbation-element-names-uns-key element_names`. A mapping stored as a
pandas DataFrame can instead carry its element names in its columns.

### Align cells before combining modalities

Cell barcodes must identify the same biological cells in the same order in
RNA and guide matrices. Use unique observation names, explicitly align the
modalities, and record any cells discarded because one modality is missing.
Combining two matrices with equal row counts is not evidence that they align.

```python
assert rna.obs_names.is_unique
assert grna.obs_names.is_unique
assert rna.obs_names.equals(grna.obs_names)
```

Keep covariates in the RNA modality's `.obs`. Use stable, unique gene and
element identifiers; requested-pair files match these identifiers, not an
external annotation guessed by the software.

## Preserve library sizes when selecting genes

The observed size factor accounts for differences in per-cell sequencing
depth. Compute a full-panel library-size column **before** selecting a smaller
gene panel, and pass it with `--library-size-key library_size`:

```python
rna.obs["library_size"] = np.asarray(rna.X.sum(axis=1)).ravel()
small_rna = rna[:, selected_gene_names].copy()
```

Here `selected_gene_names` is your preselected list. Without an explicit column,
PerTurbo computes totals from the analyzed expression matrix. Those totals
would change if you manually removed genes before saving the input. Internal
gene blocking preserves the full input panel's library sizes across blocks.

`--size-factor-key` has a different meaning: it expects an already transformed
offset, not raw UMI totals. Choose one of these two options. See
[Running analyses](running_analyses.md) for the transformation and covariates.

## Large files and a first subset

AnnData can open an `.h5ad` with `ad.read_h5ad(path, backed="r")`, allowing a
small cell/gene subset to be materialized without loading the complete count
matrix. Preserve full-panel library sizes and the cell/guide alignment when
building that subset. Keep adequate controls and state which cells and genes
were selected.

For a real high-MOI fit, keep every retained cell's co-occurring element
predictors. Use `--backed --gene-chunk-size 256` to bound expression buffers
while preserving that joint design. A debug subset can establish that loading
and execution work; it cannot establish full-screen speed or calibration.

Before a large run, inspect the control count, guide-count distribution, mapping
shape, gene identifiers, covariate completeness, and a few raw count entries.
The startup log reports the pool selected for the CRT. Confirm that it matches
your experimental design rather than accepting an unexpected choice silently.
