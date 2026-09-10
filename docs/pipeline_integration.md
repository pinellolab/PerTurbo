# PerTurbo in the IGVF CRISPR pipeline

One PerTurbo run produces every table the pipeline publishes. The separate cis
invocation is no longer needed.

## What the pipeline used to do

Two fits per dataset, distinguished only by the guide-to-element map: one grouping
guides into elements, one treating each guide as its own unit. Each was given a
`--pairs-to-test` file that restricted the fit to a cis pair set so its discoveries
could be compared one to one with SCEPTRE, and the adapter published four tables.

## What changed

The restricted and transcriptome-wide analyses differ in one thing: the
multiple-testing family. The estimates and the p-values are identical, because a
conditional randomization test is independent across pairs given the stage-one
baseline, and a pair's Bayesian effect does not depend on which other pairs were
requested. So `--pairs-to-test` no longer restricts the fit. It selects the rows of
a second table:

| file | rows | Benjamini-Hochberg family |
|---|---|---|
| `element_effects.parquet` | every pair | all tested pairs |
| `element_effects_requested_pairs.parquet` | the requested pairs present in the grid | the requested pairs |

The command line says this at startup, because the flag meant something else on the
`codex/jax-cis-high-moi` branch and a caller who has not read these notes would
otherwise wait out a transcriptome-wide run expecting a small one.

## Matching the SCEPTRE module's correction

The restricted table exists to be compared with SCEPTRE, so its q-values are
computed the way `bin/merge_sceptre_chunk_results.py` computes SCEPTRE's:
`scipy.stats.false_discovery_control` with the BH method over the finite p-values of
the table, leaving the rest missing. `tests/test_pairs_to_test_tables.py` asserts
the two agree to 1e-12, missing-value handling included.

## What the pipeline should change

1. Delete the cis PerTurbo process. One `inference_perturbo` process remains.
2. Point the container at the new image.
3. In `perturbo_v2_pipeline_adapter.py`, keep writing the pairs file and keep passing
   it, but read the restricted table from the run's output instead of launching a
   second fit. The two fits that remain are the element fit and the guide fit, which
   differ in their guide map and are both still required: the per-guide table comes
   from a fit in which each guide is its own unit, not from a derived summary.
4. `--test-all-pairs` becomes the only behaviour of the fit itself.

## Effect sizes and their uncertainty

Both tables carry `posterior_mean` and `posterior_scale`, the stage-two posterior
mean and its standard deviation, alongside `z_value` and `posterior_prob`. No
CRT-based interval is needed for the pipeline's outputs.

## Which cells a perturbation is tested against

The CRT has two pools and the choice is a flag, `--crt-pool`:

- `control-anchored` tests each perturbation inside the control cells plus its own
  cells, with the null fit on controls. It drops any cell carrying two
  perturbations. This is the low-MOI design.
- `all-cells` tests each element as a marginal association over every analysed
  cell, with the null fit on all cells. It needs the guide-to-element map and no
  control cells. This is the high-MOI design.
- `auto` measures the design from the data rather than from how the file was
  written: the screen is tested against a control pool when the median number of
  guides per cell is below `--crt-auto-moi-threshold` (default 3, so a dual-guide
  construct still reads as one perturbation) *and* at least
  `--crt-auto-min-control-cells` cells (default 100) carry nothing but control
  guides. Otherwise every cell is used. An AnnData input carries one label per
  cell and is low MOI by construction. The command line prints the measurement
  and the decision.

The pipeline today runs everything in the high-MOI design, by passing the element
map, and its own `Multiplicity_of_infection` setting (stored in
`guide.uns["moi"]`) never reaches PerTurbo. Preserve that behaviour explicitly
rather than by accident: the adapter should pass `--crt-pool all-cells` when the
pipeline's setting is `high` and `--crt-pool control-anchored` when it is `low`,
so the decision lives in IGVF's configuration and changes there if it changes.
`auto` is the right default for a user running PerTurbo by hand, not for a
pipeline that has already decided.
