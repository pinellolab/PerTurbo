Preparing inputs
================

The example notebooks ship with prepared counts. Use these functions to rebuild the Gasperini RNA and guide modalities from GEO files, inspect study subsets or process count matrices.

Read a GEO deposit
------------------

GASPERINI_GEO_SPECS maps dataset names to their expected input files.

.. autoclass:: perturbo.preprocessing.GasperiniGeoSpec

.. autofunction:: perturbo.preprocessing.summarize_gasperini_geo_inputs

.. autofunction:: perturbo.preprocessing.build_gasperini_geo_mudata

.. autofunction:: perturbo.preprocessing.write_gasperini_geo_h5mu

.. autofunction:: perturbo.preprocessing.plot_gasperini_element_gene_histogram

Match the pilot screen
----------------------

.. autoclass:: perturbo.preprocessing.GasperiniPilotMatchSummary

.. autofunction:: perturbo.preprocessing.build_at_scale_pilot_matched_mudata

.. autofunction:: perturbo.preprocessing.write_at_scale_pilot_matched_h5mu

Select a TSS panel
------------------

.. autoclass:: perturbo.preprocessing.GasperiniAtScaleSubsetSummary

.. autofunction:: perturbo.preprocessing.build_gasperini_atscale_tss_subset

.. autofunction:: perturbo.preprocessing.write_gasperini_atscale_tss_subset_h5mu

Inspect and transform counts
----------------------------

.. autofunction:: perturbo.preprocessing.to_dense_array

.. autofunction:: perturbo.preprocessing.count_gene_outliers_per_cell

.. autofunction:: perturbo.preprocessing.winsorize_counts_to_gene_thresholds

.. autofunction:: perturbo.preprocessing.compute_gene_clip_thresholds
