# Fitted MuData pilot

Local prototype against tag `v2.0.0rc12`. The command packages saved results without running a model or correcting p-values. Use it on small files that fit in memory.

```sh
python -m perturbo.fitted_export /path/to/rc12_bundle /path/to/new_export \
  --source /path/to/original_assays.h5mu --primary saddlepoint --cis
```

Supply the original assays when the saved input path has moved. Use `--primary resampling` for primary resampling results. Omit `--primary` for effects-only fits. Supply `--cis` only with independent knowledge that the requested pairs were a cis family. Outputs are published to a new directory after native MuData and Parquet results read back identically. Old bundle files remain untouched.

```python
import mudata

screen = mudata.read_h5mu("/path/to/new_export/fitted.h5mu")
pt = screen.uns["_perturbo"]
hits = pt["effects"].query("q_global < 0.1")
```

A shared fit stores its strategy and omits the fixed-efficiency table. Relative fits use the saved ordered guide and gene IDs to align the rc12 NPZ arrays to the pair table. Both DataFrames and their Parquet versions carry the same IDs, columns, row order, dtypes and missing values. Metadata, baseline arrays, loss traces, offsets, known cell indices and covariate designs also have directory equivalents. Baseline-row identity is recorded as unknown unless the bundle supplies an explicit mapping. Existing covariate transforms are applied to the retained source observations when there is no precomputed covariate table.

## Pilot coverage

The reference Gasperini subset has 12,988 cells, 9 genes and 18 guides. Its relative-effect run yielded 81 element–gene rows, 162 guide–gene rows and three transformed covariates. The fit had 10,000 baseline offsets. Native modality IDs and counts and all baseline arrays were independently compared with the original files. This input has no primary significance test. Tests exercise missing and finite p/q-values including 1e-200, cis family alignment, reversed guide/gene table ordering and failure handling.

## Restore a fitted model

`PerTurboModel.load("fitted.h5mu")` restores the shared or relative stage-two fit and its saved control baseline. The reader streams native result-table columns into dense posterior arrays. It opens the RNA counts backed and reconstructs a low-MOI construct design from the original per-cell treatment labels. A source-cell filter combined with saved offsets or a gene-name override currently materializes the retained RNA slice. Budget memory for the fitted matrices themselves. The converter above still reads its inputs into RAM.

Compare restored parameters against an existing directory bundle with

```sh
python -m perturbo.fitted_restore fitted.h5mu --reference-bundle /path/to/rc12_bundle
```

This checks the saved posterior means and scales, baseline parameters, size factors and fitted identifier axes. The reader recomputes derived z-values from their saved operands. The comparison allows three machine epsilons of relative error for CPU/GPU division. Repeated, missing or unknown pair identifiers fail restoration. Relative recovery on the real 9-gene subset passed these checks.

The fit CLI continues writing rc12 bundles. Restoration currently covers complete shared and relative effect tables without fitted perturbation dispersion. The manuscript tables omit legacy posterior diagnostics outside this scope. Authenticate replacement source files separately, since matching gene IDs and shapes alone cannot establish input identity.
