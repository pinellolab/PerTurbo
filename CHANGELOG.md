# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

## [2.0.0rc9] - 2026-09-16

### Fixed

-   Batch covariate levels are enumerated over the analysed cells, not over the
    stage-one control cells. A level the control set never sampled used to be
    absent from the design entirely: its cells got an all-zero indicator row and
    therefore the reference level's coefficient, with nothing in
    `covariate_metadata.json` to say a level had gone missing. On a 126,154-cell
    TAP-seq chr8 screen whose non-targeting guides sit almost entirely in one
    lane, one of fourteen lanes had no control cell and its 10,570 cells were
    silently merged into the reference lane, which is also the lane every
    non-targeting element lives in. The all-cells CRT pool refits the nuisance
    coefficients over every analysed cell, so such a level is identified there
    and now keeps its design column, with the reference set to the most frequent
    analysed level. Every other path estimates the nuisance coefficients from the
    control cells alone, where the level is genuinely unidentifiable: it is
    dropped from the design, named in a loud warning, and the reference stays the
    most frequent control level, which is the well-conditioned choice for that
    fit. `covariate_metadata.json` gains `batch_all_levels`,
    `batch_level_counts`, `batch_levels_source` and
    `unidentifiable_batch_levels`, so a level present in the data is never
    unlisted.
-   The all-cells propensity CRT resamples each element only within the batch
    levels where it has cells. With a categorical batch in the design, an element
    confined to some of its levels is *separated* by the level indicators: no
    finite coefficient reproduces the zeros, the unpenalized logistic MLE does not
    exist, and the IRLS - which has no step control - simply walks the
    coefficients outward on every iteration. By the twenty-fifth step they are
    large enough that forming the linear predictor in float32 cancels
    catastrophically, and the fit can collapse onto a degenerate zero/one
    assignment whose Bernoulli null has no variance at all. The saddlepoint then
    reports its floor, about 1e-12, for an element that was never perturbed.
    Restricting the support takes the limit the MLE was already walking towards -
    zero selection probability in an unoccupied level - and the separated
    directions leave the likelihood, so what remains is an ordinary identified
    logistic regression.

    On a 126,154-cell TAP-seq chr8 screen with 14 sequencing lanes, 30 candidate
    enhancers cut in silico to a single lane went from 7.7% / 4.8% / 4.18% of
    far-gene pairs at p<0.05 / 0.01 / 0.001 to 3.9% / 1.1% / 0.43%, against 4.7% /
    1.1% / 0.14% for the untouched enhancers in the same run. Untouched pairs on
    genes whose baseline reached the null mode are unchanged (Spearman 1.000000,
    largest p-value difference 5.4e-3). Elements present in every level are
    unaffected by construction, and a screen where no element misses a level takes
    the original path. On the unrestricted dataset the cis calls at q<0.05 are
    identical; the ten screen-wide calls that disappear are eight pairs whose
    z-value was 5e-5 beside a null variance of 1e-7, plus two whose p-values are
    unchanged and whose q-values moved only because those eight left the
    multiplicity pool. The same run also took 172s against the previous 1,109s;
    the screen promotes the same number of pairs either way (4,246 against
    4,426), so that is the masked fit and not less tail work, and the two runs
    were not measured under matched GPU contention.

    `--no-crt-all-cells-batch-support` restores the previous behaviour.

-   A separated propensity fit no longer needs to be diagnosed from its output:
    `AllCellsPropensityFit` carries `batch_codes` and `element_support`, and the
    CLI reports how many elements miss a level.
-   The structured nuisance algebra reduced over batch levels with a scatter-add,
    whose accelerator cost grows as the segment count *falls* because colliding
    rows serialize on one output address. A batch covariate is the worst case:
    tens of levels over hundreds of thousands of cells. `_segment_sum` now
    contracts against the indicator matrix instead - the same sum written as a
    matrix product - for designs below 256 groups, keeping the scatter beyond
    that, where it wins. The indicator is materialized as `(cells, groups)` and
    never as the full nuisance design, so the arrow structure of `Z'WZ` is
    untouched; only the reduction changes. On a 126,154-cell TAP-seq screen with
    a 14-level batch covariate, the all-cells propensity CRT over 1,041 elements
    and 68 genes went from 2,630 s to 76 s measured back to back on one V100;
    its batched logistic IRLS from 10.3 s to 0.12 s per 64 elements, and
    `fisher_nb_null` from 1.12 s to 0.05 s. Without a batch covariate the same
    run took 15 s, so the structured route was costing 35x the design it was
    meant to make cheap. Saddlepoint p-values move by less than the run-to-run
    spread the scatter itself produced: against a stored reference run, the
    median relative change is 6.0e-7 where an unmodified rerun gives 1.1e-6, and
    no call changes at p < 0.05, 1e-3 or 1e-5.
### Added

-   A run with `--batch-covariate` now reports how its control cells sit across the
    batch levels, and warns when they are confined to one. Measured on a
    126,154-cell TAP-seq screen with 14 sequencing lanes: all 30 non-targeting
    guides had been prepared in one lane, so 2,033 of the 2,049 control-only cells
    sat there and the other 13 lanes held between zero and four each. The batch
    covariate was supplied and did not help, because the control pool carries no
    information about the levels it has no cells in. Control-anchored, every lane
    effect read as a knockdown - 1,241 enhancer guides over 50 Mb "knocked down"
    MRPL13. All-cells, the non-targeting guides themselves returned 21.9% of tests
    at p < 0.05 while the targeting guides, spread over every lane, were
    calibrated. Within the one lane both pools were calibrated at 4.8%.

    Every run with a batch covariate and an identifiable control pool prints one
    `[perturbo] controls: ...` line with the control cell count, the number of
    batch levels, the busiest level's share of the control cells, that level's
    share of the analysed cells and its control fraction, and how many levels hold
    at least 20 control cells. A `[perturbo] WARNING:` line and a `RuntimeWarning`
    follow when the controls are confined: the busiest level holds more than 90% of
    the control cells while holding less than 50% of the analysed cells, or exactly
    one level holds at least 20 control cells while the screen has more than one
    level. The warning names the consequence for the pool in use - that the
    control-anchored pool is effectively a single batch and cannot identify the
    other levels' effects from controls alone, or that all-cells non-targeting
    calibration checks are confounded with batch and should be read within the
    level. The numbers are recorded in `crt_metadata.json` and
    `covariate_metadata.json` as `control_batch_levels`,
    `control_top_batch_level`, `control_top_batch_share`,
    `control_top_batch_screen_share`, `control_batch_represented_levels` and
    `control_batch_confined`, among others. No numerical result changes.

## [2.0.0rc8] - 2026-09-15

### Fixed

-   Perturbation targets with no assigned cell no longer abort a control-anchored
    CRT run. The low-MOI design keeps only cells carrying exactly one
    perturbation, so a sparse guide can lose every cell - 37 of 4,120 guides on a
    real TAP-seq screen had none to begin with. Those targets are dropped from the
    design, reported on stdout with a count and a truncated name list, and counted
    in `crt_metadata.json` as `targets_without_assigned_cells`. Their
    `element_effects` rows remain present with missing CRT statistics, so a fixed
    target set still resolves and an untested target stays distinguishable from one
    that tested null. An input where *every* target is empty is still fatal.
-   Grouping guides into elements is a sparse product. The dense int8 product had
    no BLAS kernel and ran for hours on one core on a 233,000-cell screen before
    anything reached the GPU.
-   High-MOI CLI chunking retains co-occurring predictors by fitting gene blocks
    instead of dropping other perturbation columns. Sparse assignments remain
    compact through loading and cell minibatching.
-   Observed size factors preserve zero-count cells and full-panel centering.
    Simulation bundles retain the fitted offsets and all chunked guide posterior
    fields; fixed-zero offsets remain distinct from counts-derived offsets.
-   Shared element means no longer count duplicate guides twice when perturbation
    dispersion is enabled. Mixture-NB simulation leaves the outlier component
    independent of the perturbation mean, matching the fitted likelihood.
-   Log-normal NB quadrature uses log weights, and sampling applies the requested
    sample shape once.
-   Release tests and build checks run for `v2-port`; package validation supports
    the build backend's metadata version.
-   The float32 pin read the variational parameters through their constraints and
    handed them back to the optimizer unconstrained, so every positive scale
    restarted at exp of its value and stage one began 40% above its reference
    loss. rc1 and rc2 carry this; the pin now works in the optimizer's own space.

### Added

-   PerTurbo 2.0 replaces the default implementation with NumPyro/JAX.
-   The former PyTorch/Pyro/scvi implementation is isolated under the deprecated
    `perturbo.legacy` optional extra.
-   Cortado-format MuData registrations and fit bundles are read compatibly and
    migrated to the PerTurbo v2 format when saved.

### Changed

-   Python 3.11 is now the minimum supported version.
-   The command-line entry point is `perturbo`.
-   Perturbation codes are 32-bit. A chunk holding a single perturbation received
    8-bit categorical codes from pandas, and stage two then failed where it
    combined them with the first chunk's perturbation count.
-   Guide-level summaries are derived only where a guide's effect differs from its
    element's. Under the shared strategy it does not, so nothing is derived and
    consumers join the element table through the guide map; under the offset
    strategy the moments are closed-form; only the relative strategy samples, and
    it does so in bounded element and guide blocks. The previous implementation
    drew 64 samples of the whole effect matrix whatever the strategy, tens of
    gigabytes on a screen-scale fit.

### Fixed

-   A perturbation with more cells than `--max-chunk-size` no longer aborts the
    run. It takes a chunk of its own, and the run reports which perturbations did
    so and which one sets peak memory. Screens exist with tens of thousands of
    cells behind one perturbation, and a perturbation's cells cannot be split
    across chunks without breaking its estimate.
