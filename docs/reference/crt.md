# Conditional randomization tests

`perturbo.crt` contains the supported building blocks for adding conditional
randomization tests (CRTs) to a fitted PerTurbo analysis. Most users should let
the command-line workflow manage these objects; see [Running analyses](../running_analyses.md).
This page is for callers that need to inspect a baseline, process custom element
chunks, or stream test results into their own output layer. Background and
interpretation are covered in [Method](../method.md) and [Results guide](../results_guide.md).

The control-anchored API operates on `PerTurboData` objects with a common gene
axis. The control data contain $C$ cells and the chunk data contain $N$ cells
assigned to $E$ elements. Counts are shaped `(cells, genes)`. Dense assignment
matrices are `(cells, elements)`; sparse indexed designs carry the same logical
shape. A chunk result is always `(E, G)`, with rows in `target_names` order and
columns in `gene_names` order.

The all-cells API is intended for screens in which cells can carry multiple
guides. It fits the gene-independent assignment model once with
`prepare_all_cells_propensity`, then reuses that state across gene blocks.
Its Bernoulli cumulant-generating function is evaluated directly, while the
reported tail probability uses a saddlepoint approximation. It therefore
should not be described as an exact p-value. CRT robustness to an outcome-model
misspecification also does not remove the need for a suitable fitted propensity
model.

## Main call contracts

| Call | Required state | Important result |
| --- | --- | --- |
| `prepare_crt_baseline` | matched data and stage-one `ControlFit` | `CRTBaseline`; `null_check` records the distance from the null mode, and `pre_polish_check` records the supplied fit when polishing is enabled |
| `run_crt_for_chunk` | baseline, unpadded element chunk, and its control pool | `ChunkCRTResult`; empirical and requested parametric tails have shape `(targets, genes)` |
| `prepare_all_cells_propensity` | a baseline prepared on the same analyzed cells | reusable `AllCellsPropensityFit`; sparse `cell_index` and `element_index` describe observed cell-element pairs in element-major order |
| `run_crt_all_cells` | all-cells baseline and data; optionally the prepared propensity state | `ChunkCRTResult` with no resamples; `p_value` is missing and the saddlepoint fields carry the test tail |
| `CRTAccumulator.absorb` | any compatible chunk result | scatters by element and gene names, so input chunks need not follow global row order |
| `CRTAccumulator.finalize` | all desired chunks absorbed | dictionary of `(elements, genes)` arrays; Benjamini-Hochberg correction is applied over all finite pairs for each tail family |

`gene_chunk_size` and `gene_block_size` partition independent gene calculations;
they do not change the hypothesis family. Multi-assignment cells in a
control-anchored chunk are counted in
`num_multi_assignment_cells_dropped`. The all-cells pool retains them.

## Control-anchored recipe

This example assumes the same loaders used for model fitting have produced
`control_data`, `control_fit`, and an **unpadded** `chunk_data`.

```python
from perturbo.crt import CRTAccumulator, prepare_crt_baseline, run_crt_for_chunk

baseline = prepare_crt_baseline(
    control_data,
    control_fit,
    polish=True,
)
chunk_result = run_crt_for_chunk(
    baseline,
    chunk_data,
    control_data=control_data,
    num_resamples=999,
    gene_chunk_size=500,
    resampling_mechanism="permutation",
)

collector = CRTAccumulator(
    element_names=tuple(all_element_names),
    gene_names=baseline.gene_names,
)
collector.absorb(chunk_result)
crt_columns = collector.finalize()
```

## All-cells recipe

Here `analysis_data` contains every analyzed cell, and the baseline is prepared
from a fit over those same cells.

```python
from perturbo.crt import (
    prepare_all_cells_propensity,
    prepare_crt_baseline,
    run_crt_all_cells,
)

baseline = prepare_crt_baseline(
    analysis_data,
    all_cells_control_fit,
    polish=True,
)
propensity = prepare_all_cells_propensity(
    baseline,
    analysis_data,
    include_guide_count=True,
)
result = run_crt_all_cells(
    baseline,
    analysis_data,
    propensity_fit=propensity,
    gene_chunk_size=500,
)
```

## Baseline and design objects

```{eval-rst}
.. autoclass:: perturbo.crt.ControlNuisance
```
```{eval-rst}
.. autofunction:: perturbo.crt.assemble_control_nuisance
```
```{eval-rst}
.. autoclass:: perturbo.crt.BaselineNullCheck
      :members:
```
```{eval-rst}
.. autofunction:: perturbo.crt.check_baseline_is_null_mode
```
```{eval-rst}
.. autofunction:: perturbo.crt.polish_baseline_to_null_mode
```
```{eval-rst}
.. autoclass:: perturbo.crt.CRTBaseline
      :members:
```
```{eval-rst}
.. autofunction:: perturbo.crt.prepare_crt_baseline
```
```{eval-rst}
.. autoclass:: perturbo.crt.ControlBlock
```
```{eval-rst}
.. autofunction:: perturbo.crt.control_block_for_genes
```
```{eval-rst}
.. autofunction:: perturbo.crt.build_chunk_design
```
```{eval-rst}
.. autofunction:: perturbo.crt.exclude_targets
```

## Running and collecting CRTs

```{eval-rst}
.. autoclass:: perturbo.crt.ChunkCRTResult
```
```{eval-rst}
.. autofunction:: perturbo.crt.run_crt_for_chunk
```
```{eval-rst}
.. autoclass:: perturbo.crt.AllCellsPropensityFit
```
```{eval-rst}
.. autofunction:: perturbo.crt.prepare_all_cells_propensity
```
```{eval-rst}
.. autofunction:: perturbo.crt.run_crt_all_cells
```
```{eval-rst}
.. autoclass:: perturbo.crt.CRTAccumulator
      :members: absorb, finalize
```
```{eval-rst}
.. autofunction:: perturbo.crt.fit_tail_families
```

## Validation and supported values

```{eval-rst}
.. autofunction:: perturbo.crt.validate_crt_config
```
```{eval-rst}
.. autofunction:: perturbo.crt.validate_offset_compatibility
```
```{eval-rst}
.. autodata:: perturbo.crt.CRT_CONTROL_NAME
```
```{eval-rst}
.. autodata:: perturbo.crt.CRT_TAIL_FAMILIES
```
```{eval-rst}
.. autodata:: perturbo.crt.CRT_SADDLEPOINT_FAMILY
```
```{eval-rst}
.. autodata:: perturbo.crt.CRT_ALL_TAIL_FAMILIES
```
```{eval-rst}
.. autodata:: perturbo.crt.CRT_MECHANISMS
```
```{eval-rst}
.. autodata:: perturbo.crt.CRT_POOLS
```
```{eval-rst}
.. autodata:: perturbo.crt.DEFAULT_NEWTON_STEP_TOLERANCE
```
```{eval-rst}
.. autodata:: perturbo.crt.SUPPORTED_LIKELIHOODS
```
```{eval-rst}
.. autodata:: perturbo.crt.SUPPORTED_SIZE_FACTOR_MODES
```
