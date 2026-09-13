# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Fixed

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
