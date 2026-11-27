# Fitted outputs

By default, `perturbo --input screen.h5mu --out-dir run` writes `run/fitted.h5mu` with the input assays, annotations, and results. The file can move independently of the input. Inspect `mdata.uns["_perturbo"]["effects"]` for element–gene results and `efficiency` for fitted guide efficiencies. The same default applies to `fit_from_path` and `PerTurboModel.save`.

Use `--output-format bundle` to write `run/bundle/` with `metadata.json`, `fit_arrays.npz`, and result Parquets. When applicable it includes `guide_efficiency.parquet` and `cell_covariates.parquet`. A CLI bundle references the exact input by relative path, original path and SHA-256. When saving a model trained on in-memory assays, its compact bundle includes `source.h5mu`.

```python
from perturbo import PerTurboModel

model = PerTurboModel.load("run/fitted.h5mu")
model = PerTurboModel.load("run/bundle", source_path="/new/location/screen.h5mu")
```

Explicit `--pairs-to-test` results travel in `_perturbo["requested_pairs"]` or in the compact bundle’s `element_effects_requested_pairs.parquet`.

The relocated input must be the same file, including its original bytes. Rewriting or repacking the assay changes its checksum. For scripts expecting the rc12 sidecars, select `--output-format legacy`.

## Converting an existing rc12 bundle

A fitted MuData file keeps the original assays, fit settings, effects, and test results together. The converter reads the directory bundle into memory, so the input and result tables must fit in RAM. The command below uses the named FDR columns and recorded primary test from a new fit.

```bash
python -m perturbo.fitted_export runs/gasperini gasperini-export
```

The export directory must be new. For a legacy rc12 bundle, identify the primary test explicitly. Include `--cis` only if the saved requested-pairs table contains the predefined cis family.

```sh
python -m perturbo.fitted_export /path/to/rc12_bundle /path/to/new_export \
  --source /path/to/original_assays.h5mu --primary saddlepoint --cis
```

New fits with named FDR columns record their primary test and family sizes in
`crt_metadata.json`. For these bundles, omit `--primary` and `--cis` when
exporting. The exporter copies `p_value`, `q_global`, `q_element`, and the
optional `q_cis` from the main effect table. It preserves the saved values
and missing entries in `mdata.uns["_perturbo"]["effects"]`.

For older rc12 bundles, supply the original assays when the saved input path has moved. Use `--primary resampling` for primary resampling results. Omit `--primary` for effects-only fits. Supply `--cis` only with independent knowledge that the requested pairs were a cis family. Outputs are published to a new directory after native MuData and Parquet results read back identically. Old bundle files remain untouched.

```python
import mudata

screen = mudata.read_h5mu("/path/to/new_export/fitted.h5mu")
pt = screen.uns["_perturbo"]
hits = pt["effects"].query("q_global < 0.1")
```

A shared fit stores its strategy and omits the fixed-efficiency table. Relative fits use the saved ordered guide and gene IDs to align the rc12 NPZ arrays to the pair table. Both DataFrames and their Parquet versions carry the same IDs, columns, row order, dtypes and missing values. Metadata, baseline arrays, loss traces, offsets, known cell indices and covariate designs also have directory equivalents. Baseline-row identity is recorded as unknown unless the bundle supplies an explicit mapping. Existing covariate transforms are applied to the retained source observations when there is no precomputed covariate table.

## Pilot coverage

The reference Gasperini subset has 12,988 cells, 9 genes and 18 guides. Its relative-effect run yielded 81 element–gene rows, 162 guide–gene rows and three transformed covariates. The fit had 10,000 baseline offsets. Native modality IDs and counts and all baseline arrays were independently compared with the original files. This input has no primary significance test. Tests exercise missing and finite p/q-values including 1e-200, cis family alignment, reversed guide/gene table ordering and failure handling.

## Diagnostic fields

| Directory bundle | Fitted MuData |
| --- | --- |
| Primary p-value and named BH families | In `_perturbo/effects` as `p_value`, `q_global`, `q_element` and optional `q_cis` |
| Detected-cell annotations | In the same table when present in the bundle. New fits opt in with `--report-low-information`. Older prefixed names map to `observed_nonzero`, `expected_nonzero` and `low_information`. |
| Element posterior z, normal-tail `posterior_prob`, and empirical p-value | Omitted as columns. The saved posterior mean and scale determine the posterior z. The empirical p-value uses additional null information. |
| Test z-score, method-specific p/q columns and log p-values | Omitted as separate columns. Retain the original bundle for the log p-value or a secondary testing method. |
| Per-pair tail validity, failure reason, fallback indicators, root residual and fitted null moments | Available in the original test table for the applicable testing method. |
| Guide-effect z-score | Recomputed from the saved guide-effect mean and scale when restoring the relative fit. |
| Control and effect fitting losses, covariate and test metadata | Stored under `_perturbo` when available. This includes the run-wide detection summaries and the lists of genes withheld in strict mode. |

## Restore a fitted model

`PerTurboModel.load("fitted.h5mu")` restores the shared or relative stage-two fit and its saved control baseline. The reader streams native result-table columns into dense posterior arrays. It opens the RNA counts backed and reconstructs a low-MOI construct design from the original per-cell treatment labels. A source-cell filter combined with saved offsets or a gene-name override currently materializes the retained RNA slice. Budget memory for the fitted matrices themselves. The converter above still reads its inputs into RAM.

Compare restored parameters against an existing directory bundle with

```sh
python -m perturbo.fitted_restore fitted.h5mu --reference-bundle /path/to/rc12_bundle
```

This checks the saved posterior means and scales, baseline parameters, size factors and fitted identifier axes. The reader recomputes derived z-values from their saved operands. The comparison allows three machine epsilons of relative error for CPU/GPU division. Repeated, missing or unknown pair identifiers fail restoration. Relative recovery on the real 9-gene subset passed these checks.

The legacy converter reads rc12 bundles. Restoration currently covers complete shared and relative effect tables without fitted perturbation dispersion. The manuscript tables omit legacy posterior diagnostics outside this scope. Authenticate replacement source files separately, since matching gene IDs and shapes alone cannot establish input identity.
