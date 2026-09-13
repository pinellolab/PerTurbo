# Troubleshooting

Start with the exact message in the terminal. PerTurbo validates many choices
before fitting so that a key mismatch or unsupported model does not consume a
long compute job. For expected input layouts, see
[Data preparation](data_preparation.md); for command options, see
[Running analyses](running_analyses.md).

## Input keys and controls

### “`--perturbation-key is required unless --perturbation-modality-key is provided`”

Choose the representation your file actually uses. For one label per cell,
pass the RNA `.obs` column with `--perturbation-key`. For a cell-by-guide
matrix in MuData, pass its modality with `--perturbation-modality-key`.

### “`--control-substring is required for low-MOI inputs`”

Pass a substring that occurs in all intended negative-control labels and not
in targeting labels, for example `--control-substring non-targeting`. Check
the labels in `.obs` or the element names before rerunning.

### No controls are found, or very few control cells are reported

Confirm capitalization, punctuation, and spelling in the actual element
names. In guide-level data, verify the guide-to-element map and element-name
array. PerTurbo counts a high-MOI control cell only when it carries nothing
but control guides. A warning below 1,000 control-only cells or below 1% of the
screen is not removed by changing the warning threshold; it indicates limited
negative-control information.

### “`--perturbation-element-varm-key requires --perturbation-modality-key`”

A `.varm` mapping belongs to the perturbation modality. Supply both flags. If
the mapping has no labeled columns, also pass
`--perturbation-element-names-uns-key` for the corresponding `.uns` names.

## Counts and offsets

### Counts are rejected as invalid

The expression matrix must contain finite, non-negative integer counts. Do not
provide log-normalized, scaled, residualized, or imputed expression values.
Return to the raw count layer described in [Data preparation](data_preparation.md).

### “`--size-factor-key and --library-size-key are mutually exclusive`”

Choose one. `--library-size-key` accepts raw totals and transforms them to
centered `log1p` offsets. `--size-factor-key` accepts offsets that have already
been transformed. Raw integer totals passed as size factors are rejected
because they would be used on the wrong scale.

### Size factors are centered differently between controls and analysis cells

Run both stages through the same PerTurbo command and use the same library-size
source. The command carries the control centering constant into every analysis
block. This error commonly indicates separately prepared objects or offsets
computed with different cell or gene sets.

## CRT pool and calibration

### “`--crt-pool auto chose the all-cells pool ... but there is no guide-to-element map`”

The measured median exceeded the high-MOI threshold, so the all-cells test
needs to know which guides belong to each element. Pass
`--perturbation-element-varm-key` and, when needed, the element-name `.uns`
key. If the experiment should instead use a low-MOI comparison, inspect the
assignment matrix and then choose `--crt-pool control-anchored` explicitly.

### “`--crt-pool all-cells runs the propensity saddlepoint with no resamples`”

Use the supported combination:

```bash
--crt-pool all-cells \
--crt-mechanism propensity \
--crt-tail-families saddlepoint \
--crt-saddlepoint-only
```

This path does not use `--crt-num-resamples` to produce an empirical p-value.

### The log says the CRT was skipped

The CRT is enabled by default only when its assumptions are supported. It
requires the plain negative-binomial likelihood, observed or zero fixed size
factors, zero latent factors, and no guide random effects. Read the reason
printed after “Skipping the conditional randomization test.” If testing is
required, correct that configuration and add explicit `--crt`; an unsupported
explicit request fails rather than silently proceeding.

### The empirical CRT finds no discoveries

With resampling, the smallest empirical p-value is
`1 / (number of resamples + 1)`. The log reports this floor and how many pairs
tie there. On a large testing grid the floor can exceed the
Benjamini–Hochberg cutoff. Inspect the saddlepoint p- and q-value columns and
their validity flags as described in the [Results guide](results_guide.md).
Do not interpret a resampling floor as evidence that all biological effects
are absent.

### Control-element p-values look conservative

Controls are skipped as targets by default. With
`--crt-test-control-elements`, a control element is tested against a pool that
contains its own cells, which is conservative by construction. For a cleaner
calibration check, prespecify controls held out from the pool.

### The CRT baseline reports poor convergence

Baseline polishing is on by default and moves the nuisance coefficients toward
the control-cell null mode. The default
`--crt-allow-unconverged-baseline` warns after polishing instead of stopping,
because genes with almost no control counts can dominate the diagnostic. Read
the reported pre-polish and post-polish checks. Confirm that controls contain
counts for the genes of interest and inspect the control loss curve. Use
`--no-crt-allow-unconverged-baseline` only when your workflow requires the
diagnostic to be fatal; turning the warning off does not improve the fit.

## Memory, backed mode, and GPUs

### The process runs out of host memory while reading the file

Add `--backed` so AnnData or MuData expression values stay on disk and are
read in bounded blocks. Backed mode does not reduce the device memory needed
for the active block.

### The GPU runs out of memory

For co-occurring or high-MOI assignments, lower `--gene-chunk-size`; every
block still retains all cells and perturbation predictors. For CRT-specific
memory pressure, lower `--crt-gene-chunk-size`. For mutually exclusive
low-MOI assignments, reduce `--perturbation-chunk-size` or
`--max-chunk-size`. Keep full-batch training enabled.

### PerTurbo uses the CPU or selects the wrong GPU

Pass `--device gpu` or an indexed device such as `--device gpu:1`. If no GPU is
visible, verify that the installed JAX build matches the cluster's accelerator
and driver, and that the scheduler assigned a GPU to the job. The
[Installation guide](installation.md) covers environment setup.

### Gene chunking is rejected

Gene chunking currently supports plain `--likelihood negbin`, observed or
zero size factors, `--num-factors 0`, shared always-on guide effects, fixed
perturbation dispersion, and no guide random effects or propagated baseline
uncertainty. Change the incompatible model option or run without gene
chunking if the unblocked fit fits in memory.

## Training behavior

### Effects are unexpectedly close to zero

Compare with a longer full-batch fit before drawing a biological conclusion.
The defaults are 500 steps per stage at step size `0.01`, and large effects
may require more optimization. Keep the learning rate and training length
together as a tuning decision. Inspect both loss curves and record the exact
settings used.

### The loss curve is noisy, rises, or is not finite

First validate the raw count matrix and all size-factor and covariate fields
for missing or non-finite values. Confirm that `--size-factor-key` is already
transformed and that `--library-size-key` contains raw totals. If losses are
finite but unstable, compare longer runs and review whether the selected step
size is appropriate. If they become non-finite, preserve the log and command
and reduce the problem to a small representative subset for a bug report.

### A step/epoch scheduling error appears

Use exactly one schedule family. Set `--num-steps`, or set both stage-specific
step flags. Alternatively set `--num-epochs`, or set both stage-specific epoch
flags. Do not mix steps with epochs or shared with stage-specific scheduling.
Under the default full-batch training, one epoch resolves to one update.

## Output questions

### Posterior effect columns are missing after `--crt-only`

This is expected: `--crt-only` deliberately skips stage two. The effect table
contains CRT results and missing posterior estimates. Remove `--crt-only` to
fit effect sizes as well.

### A requested pair is absent from the focused table

The terminal reports how many requested rows were present. Names must match
the analyzed element and gene names exactly. `--pairs-to-test` does not add
elements or genes and does not restrict the fit; it creates a second table and
recomputes q-values within the matched requested family.

### There is no `beta_loss_curve.png`

The file is omitted when stage two did not run, including `--crt-only`.
Otherwise, preserve the command and terminal log and check whether the run
finished after writing `element_effects.parquet`.

For unresolved problems, include the PerTurbo version, complete command,
terminal output, input container type and shapes, JAX device information, and
whether `--backed` and gene chunking were used. Avoid sharing cell-level
metadata that could identify participants.
