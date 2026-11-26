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

This is an explicit conversion experiment. The existing CLI fit writer and model restoration API still use the rc12 layout. The new directory contains manuscript-scope tables rather than the full set of legacy posterior diagnostics. Replaced source files must be authenticated separately; gene-ID and shape checks alone cannot establish input identity. The converter reads the assays and tables into RAM and is unsuitable for full GWPS. Large-scale work needs a bounded-memory writer and a benchmark on real native DataFrame encoding. AnnData and MuData cross-version reserialization remains to be tested.
