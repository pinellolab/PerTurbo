# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

-   PerTurbo 2.0 replaces the default implementation with NumPyro/JAX.
-   A conditional randomization test (`--crt`), evaluated in closed form rather
    than by resampling, so every perturbation-gene pair of a genome-scale screen
    can be tested. The null distribution of the negative-binomial score statistic
    under random reassignment of the guide is a weighted sum of independent
    Bernoulli variables; its cumulant generating function is exact and a
    saddlepoint approximation gives tail probabilities to 1e-12 without drawing a
    resample. One baseline fit is amortised over every pair. `--crt-only` stops
    after the test and skips the effect estimates.
-   Both screen designs are served by the same test: one perturbation per cell
    tested inside a pool of control cells, and many perturbations per cell tested
    as marginal associations over all cells (`--crt-pool`). The default `auto`
    measures the design from the data: all cells when the median guides per cell
    exceeds 3 (`--crt-auto-moi-threshold`), the control pool otherwise. It prints
    the measurement and the decision, reports how many cells carry only control
    guides, and warns below `--crt-min-control-cells` (1,000) or 1%; an explicit
    `--crt-pool` always wins.
-   The control-anchored test accepts a guide-to-element map. The assignment is
    collapsed to elements, the pool is the cells carrying nothing but control
    guides, and cells carrying two or more elements are set aside and counted
    rather than reinterpreted. Previously such designs were refused, which left a
    low-MOI screen analysed through the pipeline with only the all-cells pool.
-   `--pairs-to-test` writes `element_effects_requested_pairs.parquet` beside the
    transcriptome-wide table, holding the requested pairs with Benjamini-Hochberg
    recomputed within that family, so one run yields both a preselected-pair
    comparison and the full analysis.
-   The former PyTorch/Pyro/scvi implementation is isolated under the deprecated
    `perturbo.legacy` optional extra.
-   Cortado-format MuData registrations and fit bundles are read compatibly and
    migrated to the PerTurbo v2 format when saved.

### Changed

-   Python 3.11 is now the minimum supported version.
-   The Adam step size defaults to 0.01 (was 0.003) and stage two to 500 steps
    (was 2,500). On simulated screens with known effects, 0.003 needed all 2,500
    steps to converge and 300 steps at that rate left effects 27% shrunk; 0.01
    with 500 steps matches the converged fit within 2-3%. Stage one keeps 2,500.
-   The command-line entry point is `perturbo`.
-   **`--pairs-to-test` no longer restricts the fit.** It previously sampled effects
    only for the requested pairs; it now selects the rows of a second output table
    while the fit and the test still cover every pair. Callers that relied on the
    old behaviour for speed should expect a transcriptome-wide run; the command
    line states this at startup.
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
