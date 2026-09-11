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

## What changed in the pipeline (branch `perturbo-v2-single-run`)

1. The cis-only `inference_perturbo` run and the separate `inference_perturbo_global`
   process are replaced by one `inference_perturbo` process on `concat_mudata`. It
   takes the prepared inference input as a second file, for its `pairs_to_test`, and
   emits the local and the global per-element and per-guide tables together.
2. `perturbo_v2_pipeline_adapter.py` makes one invocation (still an element fit and a
   guide-identity fit, in parallel when configured), passes the requested pairs
   through `--pairs-to-test`, and writes the global tables from
   `element_effects.parquet` and the local ones from
   `element_effects_requested_pairs.parquet`.
3. The conditional randomization test runs beside the effect estimates
   (`INFERENCE_PERTURBO_CRT`, default true). Its saddlepoint p-value becomes
   `perturbo_p_value`; the posterior probability is kept as
   `perturbo_posterior_prob`.
4. Which cells a perturbation is tested against follows the pipeline's own
   `Multiplicity_of_infection` setting (`INFERENCE_PERTURBO_CRT_POOL = from-moi`:
   `high` is every cell, `low` the pure control cells plus the perturbation's own),
   so the decision lives in IGVF's configuration and changes there if it changes.
   `auto` lets PerTurbo measure the design; a pool can also be named outright.
5. The container is `ghcr.io/pinellolab/perturbo:v2.0.0rc1`.
6. Stage two runs at Adam step size 0.01 for 500 steps (`INFERENCE_PERTURBO_STEP_SIZE`,
   `INFERENCE_PERTURBO_NUM_STEPS_BETAS`), the setting a ground-truth sweep showed to
   match 2,500 steps at 0.003; the previous 300 steps at 0.003 under-converged.

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
  written: every cell is used when the median number of guides per cell exceeds
  `--crt-auto-moi-threshold` (default 3, so a dual-guide construct still reads
  as one perturbation), the control pool otherwise. An AnnData input carries one
  label per cell and is low MOI by construction. The command line prints the
  measurement and the decision, reports how many cells carry only control
  guides, and warns when they are fewer than `--crt-min-control-cells` (1,000)
  or under 1% of cells.

With an element map on the control-anchored pool, the assignment is collapsed to
elements, the pool is the cells that carry nothing but control guides, and a cell
carrying two or more elements is set aside and counted. On the Hon lab WTC11 TF
screen as the pipeline processes it (69,647 cells, median 1-2 guides per cell,
labelled `moi = high` by the pipeline's default), that is half the cells; the
all-cells pool keeps them at the cost of testing each element over every cell.
Measured on one A100 40 GB with `--crt-only`: all-cells 6:01 wall and 37 GB of
host memory for 703,888 pairs (1,907 at q<0.05); control-anchored 2:57 and 10 GB
for 656,328 pairs (670 at q<0.05, 31,619 cells set aside). The two agree on 580
pairs at q<0.05 and recover the element's own TF gene as the top hit equally
often (51% vs 53% of elements). Neither pool is free: one discards cells, the
other tests a marginal question over every cell. Choose on the design, and let
the run say which it chose.

The pipeline today runs everything in the high-MOI design, by passing the element
map, and its own `Multiplicity_of_infection` setting (stored in
`guide.uns["moi"]`) never reaches PerTurbo. Preserve that behaviour explicitly
rather than by accident: the adapter should pass `--crt-pool all-cells` when the
pipeline's setting is `high` and `--crt-pool control-anchored` when it is `low`,
so the decision lives in IGVF's configuration and changes there if it changes.
`auto` is the right default for a user running PerTurbo by hand, not for a
pipeline that has already decided.
