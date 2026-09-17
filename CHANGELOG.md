# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

-   The CLI's first log line names the code that is running:
    `[perturbo] perturbo <version> running from <import path>`, with a warning
    when a source tree on `PYTHONPATH` shadows the installed package, because
    `importlib.metadata` then still reports the installed version. The same
    three fields (`perturbo_version`, `perturbo_source`,
    `perturbo_source_shadows_installed`) are written to `crt_metadata.json`
    and `covariate_metadata.json`. Motivated by a TAP-seq run that was believed
    to be exercising rc10 while the container's rc8 was executing.

## [Unreleased]

### Fixed

-   Equal-tail Bernoulli-propensity CRT tails now check root residuals and the
    raw Lugannani–Rice correction before accepting a saddlepoint approximation.
    Valid log probabilities are retained even when their linear probabilities
    underflow. Failed approximations use a guarded Chernoff bound; a finite,
    nonnegative tilt may supply that bound when root residual is the only
    failed check. If the bound cannot be used, the result is conservatively
    p=1. This is a numerical/statistical correction and can change discoveries.
    Exact support-boundary probabilities and the symmetric convention are
    unchanged.

### Added

-   CRT tail failure codes, Chernoff/conservative-one flags, and root residuals
    are exported separately from the existing screening and low-information
    annotations. See `docs/propensity_tail_diagnostics.md`.

## [2.0.0rc10] - 2026-09-17

### Changed

-   The low-MOI control-anchored CRT runs the Replogle genome-wide screen at
    its pre-port speed again. Per 300-target chunk over 8,248 genes and a
    267-level `gem_group` covariate: rc9 took 1,013-1,274 s, this release
    247-311 s, against roughly 257 s for the 8 September cortado CLI run of the
    same design (confirmed from that run's own `covariate_metadata.json`:
    `gem_group`, 266 levels plus reference, no continuous covariates, saddlepoint
    CRT over 92,270,376 rows). rc8 and rc9 were equally slow, so the version-to-
    version regression suite could not see it; the cost arrived with the port.
    Correctness is unchanged: against the rc9 run on Replogle essential
    (19,463,699 pairs) no pair is lost or gained, log10 p agrees to Pearson
    1.000000000 with a maximum absolute difference of 2.0e-5, q<0.05 calls move
    901,737 -> 901,736 (Jaccard 0.999999), q<0.1 calls are identical, all 1,783
    of 1,970 on-target pairs are recovered on both sides, and both make zero
    null false discoveries. The dominant cost was not arithmetic but the
    device->host synchronisations and host-side stages sitting between GPU
    kernels inside each chunk, which left the GPU at 0-56% utilisation:

    -   The screen-wide shared-slope selection model reaches the saddlepoint
        kernel as one cell-level logit vector plus a per-target intercept - the
        *shared-logit* form - instead of as coefficients in a compact basis that
        the kernel rebuilt into logits with a `(controls, basis) @ (basis, 64)`
        float64 product and a `(64, width, basis)` gather for every promoted
        64-pair block. With a batch covariate the basis is one column per level,
        and those logits do not depend on the gene. `precompute_low_moi_
        permutations` returns both forms and the CRT prefers the shared-logit
        one; the screen-wide fitting policy is untouched. Given the same logits
        the two forms agree in the null moments to 1e-14 and in log p to about
        2e-7 relative on a minority of pairs, which is the fixed-iteration
        saddlepoint root solve amplifying summation-order noise - perturbing the
        compact form's own inputs by 1e-15 moves it further - and production
        stored the compact form in float32, so the switch is a precision gain.
    -   Under shared slopes the per-chunk compact basis - a rank-revealing SVD
        of the chunk's own nuisance design - is no longer built; nothing read
        it. Target-specific fits still build theirs.
    -   Each target's selection intercept is solved for a whole target batch at
        once on the device, by the same 60-step bisection on the same bracket,
        instead of one target at a time in NumPy with a host exponential over
        the pool per step.
    -   The per-pair detected-cell counts behind `crt_low_information` are
        computed on the device as segment sums over the membership COO, which
        serves a low-MOI partition and a high-MOI cell-in-many-elements design
        alike, in 256 MB row chunks; observed counts agree exactly with the
        host form and expected counts to the float32 elementwise chain.
    -   Every promoted block's saddlepoint result stays on the device until the
        chunk's loop is over and is pulled once, instead of once per block:
        on a genome-wide gene slice that was thousands of device syncs with the
        host idle between them.

-   The rank-revealing propensity basis behind the screen-wide selection fit is
    built on the device, in row blocks. It was a host `np.linalg.svd` of the
    whole column-normalised design; on the genome-wide screen that is a
    (1,989,578 x 269) float64 matrix, and it held one CPU core for about twenty
    minutes once per run while the GPU sat idle (Xaira HEK293T: 1,599 s). A
    single device SVD is not the replacement - XLA wanted ~16 GB for the 4.3 GB
    input and ran out of memory on a 16 GB card - so the basis is a tall-skinny
    QR in 262,144-row blocks streamed from the host, one QR of the stacked
    triangles, and one SVD of the single (cols x cols) R, whose singular values
    are the design's to rounding; the rank tolerance is unchanged and still
    stated in the original dimensions and source dtype. Measured in the
    production container on a V100-PCIE-16GB at the genome-wide shape: 38.7 s
    cold, 30.9 s compiled, peak device 5.1 GB, rank 268 of 269, column space
    agreeing with the host SVD to 5e-10. The decomposition runs under
    `jax.enable_x64()` regardless of the caller's precision mode, restoring the
    mode on exit, and the float32 result is assembled on the host block by
    block so device memory is bounded by one block.

-   The two selection-model representations are named for what they hold:
    `shared_logits_given` in the kernel and `shared_logit_propensity` in the
    caller replace the former "legacy" labels, and the `TargetPermutations`
    docstring says what `shared_logits` and `pool_intercepts` are rather than
    who used to read them.

## [2.0.0rc9] - 2026-09-16

### Fixed

-   The compact low-MOI propensity saddlepoint forms its own-cell logits in
    bounded row chunks. Building the vector as
    `sum(basis[own_rows] * coefficients[own_codes], axis=1)` asks XLA for two
    `(own cells, basis width)` float64 operands and their product on the way to
    a `(own cells,)` result; at X-Atlas/Orion scale that is 3,243,392 rows over
    a 109-column basis, 2.63 GiB apiece to produce 26 MB, and it drove six runs
    into GPU OOM. The chunked form accumulates the same float64 dot products on
    the row budget the pool projection already uses, so the peak scales with the
    chunk rather than the cell count. The arithmetic is untouched and the fit is
    identical to the last bit: parity is pinned at three forced chunk budgets
    and at one row per chunk, because the production budget leaves a test-sized
    screen in a single chunk and left this path covered only in that form.

-   Stage one returns its control counts to the host once the control fit is
    written, instead of leaving the full device panel resident for the CRT to
    work around.

-   `crt_low_information` is reported only for element rows that were tested.
    The count arrays keep zero as their sentinel wherever no chunk absorbed a
    row, so an element dropped for having no assigned cells, or simply outside
    this run, used to read as the most information-poor pair in the screen on
    the strength of a placeholder; its CRT statistics were already missing.
    `low_information_genes_entirely_flagged` is quantified over the tested rows
    for the same reason: asking for every row reported zero such genes on every
    control-anchored screen, whose control elements are never tested.

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
    and now keeps its design column. Every other path estimates the nuisance
    coefficients from the control cells alone, where the level is genuinely
    unidentifiable: it is dropped from the design and named in a loud warning.
    The reference level stays the most frequent level among the cells being fit
    on either path, which is the well-conditioned choice for that fit.
    `covariate_metadata.json` gains `batch_all_levels`,
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

-   The all-cells refit no longer anchors the batch reference on a level the
    stage-one cells never occupy. The reference is folded into the intercept, so
    taking the most frequent *analysed* level could pick one with no control
    cell: every control row then carried exactly one indicator, which is the
    intercept written a second time, and the stage-one design lost a rank (3 of
    4 on the toy that found it). The level itself, now the reference, had no
    column for the refit to identify either, the zero-variance exemption had
    nothing to exempt, and the warning claimed a column that did not exist.
    Stage two reuses stage one's covariate coefficients, so the aliasing
    propagated. The reference is now the most frequent level among the cells
    being fit on every path; the all-cells refit is unpenalized, so its fitted
    means are invariant to that choice, and an analysed-only level keeps its
    column through the exemption rather than through the reference.

-   `--crt-all-cells-batch-support` now protects a two-level batch covariate.
    The support was read off the bordered factorization, which declines a design
    with fewer than two indicator columns because a one-column diagonal block is
    not worth splitting off. A two-level factor is coded by exactly one column,
    so the support came back empty and the flag was a silent no-op - though an
    element confined to one of two lanes is separated by that column just as it
    is by thirteen. The level codes are now derived from the design's mutually
    exclusive binary indicator block directly, with the dropped reference level
    recovered as its own code. Designs with three or more levels get the same
    codes as before and are unaffected.

-   The new indicator-matrix contraction in `_segment_sum` asks for
    `Precision.HIGHEST`. The scatter-add it replaces summed float32 terms
    exactly, while a matrix product is free to run in TF32 on Ampere and later -
    ten mantissa bits where the reduction feeding `weighted_information`,
    `transpose_dot`, `fisher_nb_null` and the low-MOI score path had twenty-four.
    No CPU result changes.

### Changed

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

-   Every CRT pair now says how much data its p-value rests on. The element table
    gains `crt_observed_nonzero` (how many of the element's cells detected the
    gene, SCEPTRE's low-MOI effective sample size), `crt_expected_nonzero` (how
    many the stage-one null expected to, summing `1 - (theta/(theta+mu))^theta`
    over the same cells at the baseline's own fitted mean) and
    `crt_low_information`, set when neither reaches
    `--crt-min-informative-cells` (default 5; 0 keeps the counts and flags
    nothing). `crt_metadata.json` records the threshold, the flagged-pair count,
    how many of those are otherwise significant at q<0.05, and how many genes are
    flagged for every element; the run prints the same counts.

    It is a flag and never a gate: p-values, q-values and every validity field
    are bit-identical with the threshold set to any value. The two counts are
    both needed because each is wrong in one direction. On a 26,432-cell Replogle
    chunk, 403 pairs move by more than 0.1 log10 p between two runs of the
    identical binary; `max(observed, expected) < 5` catches 386 of them and
    removes 1 of 116 on-target calls, where observed alone at 7 removes 40 of
    them (a real knockdown pushes observed to zero) and expected alone at 5
    removes 164 of 245 induction-like calls (an induction from an undetected
    baseline expects nothing). Both CRT pools compute the counts, over each
    element's own cells, as one sparse indicator product per gene block.

-   A separated propensity fit no longer needs to be diagnosed from its output:
    `AllCellsPropensityFit` carries `batch_codes` and `element_support`, and the
    CLI reports how many elements miss a level.

-   `crt_metadata.json` records `all_cells_batch_support`, together with
    `all_cells_elements_missing_a_batch_level`, `all_cells_elements` and
    `all_cells_batch_levels`, so a stored result says whether its all-cells null
    was restricted to each element's own batch levels and how many elements that
    could have applied to. `cli_args` in the model-parameter bundle records the
    flag, and the bundle carries the measured counts beside it as
    `crt_all_cells_batch_support_summary`. The element-support summary line is
    also printed by an unchunked run, which used to compute the support and say
    nothing about it.

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
