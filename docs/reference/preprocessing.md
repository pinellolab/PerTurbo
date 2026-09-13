# Preprocessing reference

`perturbo.preprocessing` supplies count-threshold utilities and specialized
Gasperini input preparation. The general input contract is described in
[data preparation](../data_preparation.md). These helpers do not replace guide
calling or cell quality control.

## Count utilities

Sparse inputs use their existing row and gene ordering. Thresholds must be
computed on the intended analysis population and reused in the same gene order;
recomputing them independently for each perturbation can change the analysis.
The example below counts outliers without changing the source counts.

```python
from perturbo.preprocessing import (
    compute_gene_clip_thresholds,
    count_gene_outliers_per_cell,
)

thresholds = compute_gene_clip_thresholds(
    rna.X, percentile=99.5, threshold_floor=2, row_chunk_size=50_000
)
outliers = count_gene_outliers_per_cell(
    rna.X, thresholds, row_chunk_size=50_000
)
```

```{eval-rst}
.. autofunction:: perturbo.preprocessing.compute_gene_clip_thresholds

.. autofunction:: perturbo.preprocessing.count_gene_outliers_per_cell

.. autofunction:: perturbo.preprocessing.winsorize_counts_to_gene_thresholds

.. autofunction:: perturbo.preprocessing.to_dense_array
```

Winsorization changes counts; censoring instead changes the likelihood assigned
to observations above a cutoff. Enabling one is not a substitute for configuring
the other.

## Gasperini dataset helpers

These functions encode specific source-file layouts and matching rules. Inspect
their signatures and summary objects before applying them to a new dataset.
They are not required when an AnnData/MuData object is already prepared.

```{eval-rst}
.. autoclass:: perturbo.preprocessing.GasperiniGeoSpec
   :members:

.. py:data:: perturbo.preprocessing.GASPERINI_GEO_SPECS

   Mapping of supported dataset names to ``GasperiniGeoSpec`` source-file specifications.

.. autofunction:: perturbo.preprocessing.build_gasperini_geo_mudata

.. autofunction:: perturbo.preprocessing.summarize_gasperini_geo_inputs

.. autofunction:: perturbo.preprocessing.write_gasperini_geo_h5mu

.. autofunction:: perturbo.preprocessing.plot_gasperini_element_gene_histogram

.. autoclass:: perturbo.preprocessing.GasperiniAtScaleSubsetSummary
   :members:

.. autofunction:: perturbo.preprocessing.build_gasperini_atscale_tss_subset

.. autofunction:: perturbo.preprocessing.write_gasperini_atscale_tss_subset_h5mu

.. autoclass:: perturbo.preprocessing.GasperiniPilotMatchSummary
   :members:

.. autofunction:: perturbo.preprocessing.build_at_scale_pilot_matched_mudata

.. autofunction:: perturbo.preprocessing.write_at_scale_pilot_matched_h5mu
```
