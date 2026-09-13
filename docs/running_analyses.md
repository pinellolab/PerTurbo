# Running analyses

This guide explains the choices in a PerTurbo command. For file layout and
validation before a run, see [Data preparation](data_preparation.md). For a
short first run, see the [Quickstart](quickstart.md). The statistical model is
described in [Method](method.md), and the output columns are explained in the
[Results guide](results_guide.md).

PerTurbo fits a control model first, then estimates an effect for every
element–gene pair. By default it also runs a conditional randomization test
(CRT) when the selected model supports it.

The defaults in this guide are the command-line defaults. The Python
`fit_from_path` convenience function has its own defaults; set its arguments
explicitly rather than assuming that an omitted Python argument behaves like
an omitted CLI flag. See the [API reference](api.md).

## Identify the expression and perturbation data

For MuData, name the RNA modality with `--modality-key`. A high-MOI or
guide-level experiment also needs `--perturbation-modality-key`. If guide rows
map to larger perturbation elements, provide the `.varm` mapping and, when the
mapping has no column labels, the `.uns` array of element names.

```bash
perturbo \
  --input screen.h5mu \
  --out-dir results \
  --modality-key rna \
  --perturbation-modality-key grna \
  --perturbation-element-varm-key element_targeted \
  --perturbation-element-names-uns-key element_names \
  --control-substring non-targeting
```

Here `grna.X` is the cell-by-guide assignment matrix,
`grna.varm["element_targeted"]` maps guides to elements, and
`grna.uns["element_names"]` names the element columns. `--perturbation-layer`
selects an assignment layer instead of `grna.X`. Use validated guide calls
with the assignment values expected by the fitted model.

For low-MOI data stored as one label per cell in RNA `.obs`, use
`--perturbation-key` and a control substring:

```bash
perturbo \
  --input screen.h5ad \
  --out-dir results \
  --perturbation-key perturbation \
  --control-substring non-targeting
```

The substring must reliably identify negative controls in the element names.
Stage one uses at most 10,000 control cells by default; change this cap with
`--max-control-cells`.

## Choose the CRT pool

`--crt-pool auto` is the default. PerTurbo measures the realized multiplicity
of infection from the data. It chooses the all-cells pool when the median is
strictly greater than 3 guides per cell, and the control-anchored pool
otherwise. The log records the measured median and the selected pool. Use
`--crt-auto-moi-threshold` only when your experimental design justifies a
different boundary.

The pools answer related but different questions:

- `control-anchored` is the low-MOI analysis. Each target is compared within
  its cells plus control cells, and the null outcome model is fit to controls.
- `all-cells` is the high-MOI analysis. Each element is tested as a marginal
  association across all analyzed cells. It requires a guide-to-element map
  and uses the propensity saddlepoint calculation without resampling.

The auto decision depends on guide count, not the number of controls. PerTurbo
warns when fewer than 1,000 cells, or less than 1% of cells, carry only control
guides. Treat that warning as an experimental-design concern: the low-MOI null
may be noisy, while high-MOI control elements provide weak calibration. The
threshold for the count warning is `--crt-min-control-cells`.

The production CRT configuration uses the negative-binomial likelihood, fixed
observed or zero size factors, zero latent factors, no guide random effects,
and a propensity saddlepoint tail. If an unsupported option was chosen and
`--crt` was not explicit, PerTurbo fits effects and logs that it skipped the
CRT. An explicit `--crt` turns the same incompatibility into an error.

## Control memory with gene blocks and disk-backed reads

Use `--backed` for a large `.h5ad` or `.h5mu` file so expression blocks are
read from disk. Backed mode limits host memory; the current working block still
has to fit on the selected JAX device.

`--gene-chunk-size N` fits `N` genes at a time in stage two while retaining
every cell and every perturbation predictor. This is the appropriate
memory-bounded path when perturbations co-occur. It currently requires:

- `--likelihood negbin`;
- observed or fixed-zero size factors;
- `--num-factors 0`;
- shared, always-on guide effects;
- no guide random effects, baseline-uncertainty propagation, or fitted
  perturbation dispersion.

The default `--gene-chunk-size 0` leaves explicit gene blocking off. With mutually
exclusive assignments it permits perturbation chunking. If assignments
co-occur and would otherwise be split, PerTurbo automatically switches to
256-gene blocks so the joint perturbation model remains intact. Set an
explicit positive gene chunk size when a smaller or larger block is needed.

`--perturbation-chunk-size` caps the number of perturbations in a low-MOI
chunk. At its default of `0`, PerTurbo chooses chunks using
`--max-chunk-size`, whose default is 50,000 cells. Perturbation chunks are not
used to split co-occurring predictors because doing so would change the fitted
model. `--crt-gene-chunk-size` is a separate CRT memory control and defaults
to 2,000 genes.

PerTurbo uses full-batch SVI by default. Leave `--minibatch-size`,
`--minibatch-size-control`, and `--minibatch-size-betas` at `0`. Gene blocks
already bound the large cells-by-genes arrays without changing which cells
contribute to an update.

For example, a large high-MOI screen can be run as:

```bash
perturbo \
  --input screen.h5mu \
  --out-dir results \
  --modality-key rna \
  --perturbation-modality-key grna \
  --perturbation-element-varm-key element_targeted \
  --perturbation-element-names-uns-key element_names \
  --control-substring non-targeting \
  --backed \
  --gene-chunk-size 256 \
  --device gpu
```

## Account for library size and measured covariates

The default `--size-factor-mode observed` conditions on a fixed per-cell
offset. With `--library-size-key total_counts`, PerTurbo transforms raw library
sizes to centered `log1p` offsets. With no library-size key, it uses each
cell's total over the analyzed genes. `--size-factor-key` instead expects an
already transformed size factor, usually centered near zero. The two keys are
mutually exclusive.

`--size-factor-mode none` fixes all offsets to zero. `infer` estimates latent
size factors, but it is incompatible with the CRT and with gene chunking.

Use `--continuous-covariates percent_mito guide_count` for numeric `.obs`
fields. PerTurbo applies `log1p` plus standardization to count-like fields and
standardization to other continuous fields. Use `--batch-covariate batch` for
a categorical `.obs` field; PerTurbo uses one-hot columns and drops the most
frequent level as the reference. A field cannot be both continuous and
categorical. The fitted transformation is saved in
`covariate_metadata.json`.

Covariates should be recorded before treatment and should represent nuisance
variation you want to adjust for. See [Method](method.md) for how they enter
the model.

## Choose training length

The defaults are 500 full-batch updates for the control fit and 500 for the
effect fit, with Adam step size `0.01`. Under full-batch training, one step is
one pass through the cells used by that stage, so one epoch also resolves to
one step. Each gene block receives the complete stage-two schedule.

Use one scheduling style per run:

- `--num-steps N` sets both stages, or provide both
  `--num-steps-control` and `--num-steps-betas`.
- `--num-epochs N` sets both stages, or provide both
  `--num-epochs-control` and `--num-epochs-betas`.

Step-based and epoch-based flags cannot be mixed. Shared and stage-specific
flags of the same type cannot be mixed either. Compare a longer run before
interpreting unexpectedly small effects; the learning rate and number of
updates work together. Inspect `control_loss_curve.png` and
`beta_loss_curve.png` rather than assuming the defaults converged for every
dataset.

## Run only the test

`--crt-only` runs stage one and the CRT but skips stage-two effect fitting.
This is useful for power or calibration work. The element table is still
written, with missing posterior effect estimates and populated CRT columns.
It cannot be combined with `--no-crt`.

## Define a focused multiple-testing family

`--pairs-to-test pairs.csv` accepts CSV, TSV, or Parquet with columns
`element` and `gene`. It does not shorten fitting or testing: the full grid is
still analyzed. PerTurbo writes both the transcriptome-wide table and
`element_effects_requested_pairs.parquet`, with Benjamini–Hochberg q-values
recomputed among the requested pairs that exist in the analyzed grid. This is
appropriate when the focused family, such as prespecified cis pairs, was
defined before looking at the results.

## Know what is saved

Every successful run writes `control_fit.npz`, `control_loss_curve.png`, and
`element_effects.parquet`. A completed stage-two fit also writes
`beta_loss_curve.png`. CRT runs add `crt_metadata.json` and CRT columns to the
effect table. Covariates add `covariate_metadata.json`; relative guide effects
add `guide_efficiency.parquet`.

By default, `--save-model-params` also writes a light, simulation-ready bundle
in the output directory that can be loaded with `PERTURBO.load(...)` while the
original input remains available. Use `--no-save-model-params` when that
bundle is not needed. See the [Results guide](results_guide.md) for biological
interpretation and the [API reference](api.md) for programmatic access.
