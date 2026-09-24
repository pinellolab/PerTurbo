# rc11 follow-up changes (September 24, 2026)

These changes were validated on `v2-port` after rc11 and are included in
`2.0.0rc12`. The release also includes the guide-efficacy/API compatibility
changes documented in the changelog.

## API/CLI parity (#61)

`fit_from_path` now defaults to the CLI learning rate, 0.01 (formerly 0.003).
This intentionally changes fits for API callers that omit `step_size`; pass
`step_size=0.003` to retain that setting. Explicit values are unchanged.
Both interfaces default to 500 CRT genes per block and expose the informative
cell annotation threshold and all-cells batch-support switch. The annotation
does not remove hypotheses from BH correction.

For 16 GiB GPUs, smaller CRT blocks, `max_chunk_size=25000`, and
`perturbation_chunk_size=270` are starting settings, not a memory guarantee.
Cell count, guide multiplicity, covariates, and other resident buffers matter.

## Separate all-cells CRT gene scheduling (#62)

In gene-chunked runs, CRT uses its own outer gene schedule. Stage two keeps its
configured width. The CRT planner reserves the gather budget and estimates
40 bytes per cell/gene within 80% of device capacity. It reports its effective
width; if device capacity is unavailable it does not widen past the existing
stage-two width. This is conservative planning, not a measured peak-memory cap.
The screen is never densified across all genes by this new path.

Design metadata and propensity fits are reused. Results accumulate into the
same screen-wide BH family. CRT-only skips stage-two data loading/fitting;
combined runs finish the CRT pass before traversing the stage-two blocks.

## Root convergence and fallback (#60)

All three root solvers require sufficient Newton progress relative to the
previous step, otherwise bisect, then perform three bracket-clipped Newton
polish steps. This addresses interior Newton two-cycles without changing the
likelihood or target equation. It can intentionally change p-values for formerly
incomplete solves. It is a correctness change, separate from runtime parity.

A finite correctly signed tilt still provides a Chernoff bound when the only
failure flags are root residual (16), raw LR probability above one (128), or
both (144). Other support, sign, and finiteness checks remain in force. The
strict policy remains unchanged.

The source identity logger also distinguishes a source checkout from a different
editable installation: an installed rc5 distribution must not label imported
rc11 source as rc5.

## Validation

CPU checks used the existing JAX 0.11.0 runtime without altering its environment.

- API/CLI and gene scheduling: 40 focused tests passed before the solver port.
- Combined solver, packed saddlepoint, scheduling, API-default and source-identity
  tests: 46 passed after integration.
- A real all-cells CRT fixture (500 cells, seven genes, three schedules including
  short final blocks): p/log-p/q agreed within rtol 1e-5; failure flags, detection
  counts, low-information flags and q<0.05 discoveries matched exactly.
- Cycling-root tests compare to independent bisection and the LR approximation
  at that root, not to exact finite-sample tail probabilities.

No full-scale GPU timing or peak-memory guarantee is inferred from these checks.

## rc12 release validation

The merged release candidate passed 88 API, output-contract, saved-model,
source-provenance, scheduling, detection-count and saddlepoint tests, plus
28 Newton-cycle, Chernoff-fallback and packed-saddlepoint tests on the existing
CPU runtime. `uv lock --check --offline` passed after updating only the package
version. GitHub's full Test workflow was disabled for inactivity at release
preparation; the build and container smoke-test workflows remained active.
