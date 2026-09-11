# The saddlepoint CRT: a quickstart

This is the user-facing guide to perturbo's conditional randomization test
(CRT). It covers what the test is, which of its two designs applies to a
screen, the commands, the outputs, and the limits worth knowing before
reading a p-value. The research notes under `docs/research/` in the PerTurbo research repository hold the
derivations and the calibration record - its `README.md` indexes
them, `14_saddlepoint_crt_exposition.md` is the method write-up and
`15_run_ledger.md` the list of runs; `AGENTS.md` holds the implementation map.

## What it is

For every (perturbation, gene) pair the CRT asks whether the cells carrying
the perturbation express the gene differently from what a null model of the
unperturbed baseline predicts. The statistic is the negative-binomial score
for a perturbation effect, projected free of the nuisance terms (intercept,
size factor, covariates, batch), and its null distribution comes from
re-assigning perturbation labels according to a fitted *selection model*: each
cell's probability of carrying the perturbation given its covariates. Because
that distribution is a sum of independent Bernoulli terms, its cumulant
generating function is exact and the tail probability is evaluated with a
saddlepoint approximation instead of resampling. No resamples are drawn, the
p-values reach far below what any permutation count could resolve, and a
transcriptome-wide screen finishes in minutes on one GPU.

Two properties matter for interpretation:

- The null is fit on the baseline and reused, never refit per pair. Stage one
  of perturbo's two-stage model supplies it; the CRT then moves the nuisance
  coefficients onto the exact null mode by Fisher scoring
  (`--crt-polish-baseline`, on by default; always on for the all-cells design). The polish is what makes the test insensitive to how long stage one trained: measured on the simulation, the null false-positive rate and the power are unchanged from 100 stage-one steps to 2,500.
- The statistic is a score test at the null. It is calibrated and matches
  SCEPTRE's power, but a Wald test that refits the alternative (an NB GLM) can
  be two to four points more powerful when many guides of mixed efficacy share
  an element. That is the price of an exact null.

## Which design

Nothing in the package measures MOI; you choose the pool.

| | Control-anchored (low MOI) | All cells (high MOI) |
|---|---|---|
| Requires | at most one perturbation per cell, and a pool of unperturbed control cells | a guide-to-element map; nothing else |
| Null fit on | control cells | every analysed cell |
| Each pair tested in | controls plus the target's own cells | all cells, as a marginal association |
| Flag | `--crt-pool control-anchored` | `--crt-pool all-cells` |

A cell with two perturbations in a low-MOI screen is dropped, not
reinterpreted. The all-cells test does not adjust for other elements'
effects on the same gene; in a real screen where any one element sits in a few
percent of cells this costs little, but a simulated design that packs one
gene's elements into most cells will mislead it.

### Letting the data choose: `--crt-pool auto`

The default, `auto`, measures the design instead of inferring it from how the
file was written: the screen is high MOI, and every cell is used, when the
median number of guides per cell exceeds `--crt-auto-moi-threshold` (default
3, so a screen whose constructs carry two guides each still reads as one
perturbation per cell); otherwise it is low MOI and each perturbation is
tested against the control pool. An AnnData input has one label per cell and
is low MOI by construction. The command line prints the measurement and the
decision, and an explicit `--crt-pool` always wins, so a pipeline that has
already decided its design should say so rather than let the measurement
decide.

Whichever pool is chosen, the run reports how many cells carry nothing but
control guides and warns when they are fewer than `--crt-min-control-cells`
(default 1,000) or under 1% of all cells: a thin control population makes
the control-anchored null noisy and leaves the all-cells test few
calibration negatives.

Measured on the screens this package has been run on
(`docs/crt_pool_auto_litmus.csv`):

| screen | median guides per cell | control-only cells | `auto` |
|---|---|---|---|
| Replogle K562 essential, IGVF pipeline reprocessing | 1 | 10,176 | control-anchored |
| Replogle K562 essential, released matrix | 1 | 10,691 | control-anchored |
| Hon lab WTC11 TF screen, IGVF pipeline processing | 1-2 | 2,302 | control-anchored |
| Gasperini pilot | 15 | 462 | all-cells |
| Gasperini at-scale, released matrix | 28 | 1,527 | all-cells |
| Gasperini at-scale, IGVF pipeline processing | 14 | 415 | all-cells |

### Element maps on the control-anchored pool

A low-MOI screen usually arrives with a guide-to-element map, since that is
how the pipeline groups guides. The control-anchored test collapses each
cell's assignment to elements, takes as its pool the cells that carry
nothing but control guides (a cell with a control guide beside a targeting
one is perturbed, and is analysed as such), and sets aside any cell that
carries two or more elements rather than reinterpret it: the null resamples
one label per cell. The run prints the count set aside. On a screen at a
realised MOI well above one this can be a large share of the cells, and
`--crt-pool all-cells` is the test that keeps them, at the cost of testing
each element as a marginal association over every cell. The Bayesian effect
estimates use every analysed cell in either case.


## Commands

Low MOI (Replogle-style), saddlepoint only, with a batch covariate:

```bash
perturbo --input screen.h5mu --out-dir out \
  --modality-key gene --perturbation-key perturbation --control-substring non-targeting \
  --batch-covariate gem_group --library-size-key total_umis --size-factor-mode observed \
  --crt --crt-mechanism propensity --crt-tail-families saddlepoint --crt-saddlepoint-only \
  --crt-polish-baseline --crt-allow-unconverged-baseline \
  --num-steps-control 2500 --num-steps-betas 300
```

High MOI (Gasperini-style), CRT only, no stage-two fit:

```bash
perturbo --input screen.h5mu --out-dir out \
  --modality-key gene --perturbation-modality-key guide \
  --perturbation-element-varm-key guide_intended_target_pairs \
  --perturbation-element-names-uns-key intended_targets \
  --library-size-key total_umis --size-factor-mode observed \
  --continuous-covariates total_umis percent_mito --batch-covariate prep_batch \
  --crt --crt-only --crt-pool all-cells --crt-mechanism propensity \
  --crt-tail-families saddlepoint --crt-saddlepoint-only --crt-allow-unconverged-baseline \
  --num-steps-control 2500 --num-steps-betas 300
```

The covariates are not decoration. On the real at-scale screen the same CLI
run with the size-factor offset alone called 35,499 pairs at q<0.1, with a
log-depth covariate 28,227, and the research driver with depth, percent
mitochondrial and the preparation batch 19,513; non-targeting calibration at
p<0.05 was nominal in every case, so the excess lives in the far tail of
highly expressed genes, where low-quality and unusually deep cells shift
many genes at once. Simulated screens carry none of that structure, which is
why the calculator's grids agree with the research driver to 0.99 without any
covariate. A real screen needs the covariates it needs: depth beyond the
offset, mitochondrial fraction, and whatever batch structure the preparation
had. In Python the same flags are `continuous_covariates=["total_umis",
"percent_mito"]` and `batch_covariate="prep_batch"`.

The same from Python, with snake_case keywords for every flag:

```python
from perturbo import fit_from_path

fit_from_path(
    "screen.h5mu", "out",
    modality_key="gene", perturbation_modality_key="guide",
    perturbation_element_varm_key="guide_intended_target_pairs",
    perturbation_element_names_uns_key="intended_targets",
    library_size_key="total_umis", size_factor_mode="observed",
    continuous_covariates=["total_umis", "percent_mito"], batch_covariate="prep_batch",
    crt=True, crt_only=True, crt_pool="all-cells", crt_mechanism="propensity",
    crt_tail_families=("saddlepoint",), crt_saddlepoint_only=True,
    crt_allow_unconverged_baseline=True,
    num_steps_control=2500, num_steps_betas=300,
)
```

`--crt-only` stops after stage one and the CRT: the path for power
calculations and calibration checks, where the stage-two posterior is not
needed. Without it, the stage-two effect estimates and the CRT p-values land
in the same table.

Flags worth knowing:

- `--crt-polish-baseline` moves the stage-one coefficients onto the control
  null mode. With a batch covariate the SVI baseline sits far enough off the
  mode that the null-mode guard fails without it.
- `--crt-allow-unconverged-baseline` lets a run proceed when a few degenerate
  genes (all-zero or near-zero in the controls) never reach the mode. The guard
  is the only check on baseline quality, so read the reported counts.
- `--crt-two-sided {equal-tail,symmetric}` picks the two-sided convention and
  defaults to `equal-tail`: twice the tail on the observed side, each tail at
  its own saddlepoint. `symmetric` is P(|S| >= |observed|).
  The score's null is right-skewed, so the symmetric form over-rejects
  up-regulation and under-rejects knockdowns (on the essential screen's non-targeting cells the
  right tail beat the left 6-fold at p<0.001); equal-tail rejects each tail
  equally and, on the benchmark, matches SCEPTRE on up-regulation and beats it
  by 1-6 points on knockdowns. Results recorded before 7 September 2026 used
  the symmetric convention; the run ledger
  (`docs/research/15_run_ledger.md` in the research repository) marks each run either way.
- A batch covariate on its own (`--batch-covariate`, with no continuous
  covariates) takes a categorical kernel rather than the dense nuisance
  design: the batch codes are kept, the kernel fits its per-batch intercepts at
  stage one's dispersion, and the test is stratified by batch as the closed-form
  null requires. Results match the dense path (Replogle essential, to the
  digits reported) and the CRT chunks run 4.3x faster. Adding a continuous
  covariate beside the batch returns to the dense path. There is no flag: the
  choice follows the nuisance design.
- `--crt-gene-chunk-size` bounds memory on wide panels; results do not depend
  on it. Perturbation chunking (`--max-chunk-size`) never changes a target's
  p-value either, because resamples are keyed on the target's name.
- Supported configuration is deliberately narrow: plain negative-binomial
  likelihood, observed or fixed size factors, no latent factors, no guide
  random effects. Latent size factors are refused because a per-cell offset fit
  jointly with the effect breaks the exchangeability the test relies on.

## Outputs

The element-by-gene table (`element_effects.parquet`) gains:

| column | meaning |
|---|---|
| `crt_saddlepoint_p_value` | two-sided saddlepoint p-value |
| `crt_saddlepoint_q_value` | Benjamini-Hochberg over every tested pair in the run |
| `crt_saddlepoint_used_screen` | the pair kept the fast three-cumulant screen value (it was nowhere near significance) |
| `crt_z_value` | standardized score, (observed - null mean) / null sd, the sign of the effect |
| `crt_p_value` | NaN in saddlepoint-only runs; the resampling p-value otherwise |

Rank by `crt_saddlepoint_q_value` for discovery; for effect sizes use the
stage-two posterior (or a GLM), not the score. Score magnitude saturates for
strong effects.

## What to expect

- Calibration: on real non-targeting labels the rejection rate at p<0.05 is
  0.045-0.056 on Replogle essential and the Gasperini at-scale screen, and the
  realised FDR in simulation sits at the target. The far tail on real
  non-targeting labels is inflated (five to six times nominal at 1e-5 on
  Replogle) and disappears when the labels are shuffled: that is structure
  among the real NTC cells, not the test.
- Power: within half a point of SCEPTRE on every simulated design; two to
  four points under an NB GLM Wald test with four guides of mixed efficacy at
  100-200 cells per guide; far above rank tests and the t-test at equal FDR.
- Speed: Replogle essential (310k cells, 2,273 targets, 8,563 genes, 47 gem
  groups) in about 19 minutes end to end through the CLI on one A100, of
  which 17 are the CRT chunks and 8 the stage-one fit; a Figure-6-style simulated
  screen of 700 to 35,000 cells and 2,000 genes in about a minute end to end
  including the stage-one fit.
