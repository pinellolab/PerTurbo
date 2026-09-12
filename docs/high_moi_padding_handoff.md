# High-MOI padding-gradient test handoff

Branch: `codex/high-moi-padding-gradient`, based on `v2-port` at `a448511`.
Implementation and regression-test commit: `080a559` (can be cherry-picked alone).

This branch changes only padding handling in `design_matrix_product`. Padding
previously gathered coefficient row zero and masked its contribution to zero;
autodiff still scattered the resulting zero updates into that real row. The
candidate uses an out-of-bounds sentinel with a zero-filled gather, so its
transpose drops those updates. The model, optimizer, step schedule, and
full-batch fitting remain the same. There is no experimental sparse dependency.

For 200k cells, mean 30 active elements and padded width 104, approximately
14.8 million padding slots target element zero per gene per update in the
original code. The actual **element** width may differ from the guide width.
Repeated destinations can cause contention in GPU atomic scatter updates.
This mechanism is visible in the compiled graph, but its runtime impact on
the cluster is **unverified**. The candidate retains the padded array dimensions.

## What has been checked

The pre-branch prototype passed float32/float64 forward and gradient checks.
In a synthetic backed CLI run with 6k cells, 1,024 elements, and three 32-gene
blocks (40 full-batch effect updates each), the candidate produced bit-identical
means, scales, z-values, and posterior probabilities for all 98,304 output rows.
CPU execution time was essentially unchanged, around 1.1 s per 20 updates.
This is neither GPU performance evidence nor real-data calibration evidence.

On this branch, all 47 targeted tests passed (padding references, gene-block
SVI/CLI, model, indexed CRT, and SVI precision), and changed Python files passed
Ruff. Reproduce the tests with the existing environment:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m pytest -q \
  tests/test_sparse_design_padding.py tests/test_gene_chunked_svi.py \
  tests/test_gene_chunk_cli.py tests/test_svi_stays_float32.py \
  tests/test_model.py tests/test_crt_indexed_design.py
```

Generated benchmark outputs should be written
outside the checkout. The standalone runner is included specifically for this
branch's testing handoff; it adds no package dependency or installed command.

## First test: isolate the GPU backward contraction

In a GPU allocation, use the existing PerTurbo environment and run from the
branch checkout. `PYTHONPATH=src` selects this checkout rather than another
installed version. `JAX_PLATFORMS=cuda` fails if CUDA is unavailable.

```bash
PYTHONPATH=src JAX_PLATFORMS=cuda python benchmarks/high_moi_padding.py \
  --cells 200000 --elements 5000 --active 30 --width 104 --genes 32 \
  --repeats 5 --out-dir /tmp/perturbo-padding-104
```

The same process compares the original implementation with this branch's
production function. The fixture retains all 200k synthetic cells and needs no
Gasperini file. It records synchronized timings, device and software versions,
reference errors, compiled HLO, and temporary-memory estimates in `result.json`
and adjacent files. Start with 32 genes before testing the real block width.

Compare the **adjoint** timing first: it isolates the backward multiplication,
without the forward product or likelihood. Forward timing is recorded separately.
Repeat with width 30 to check whether any improvement depends on padding:

```bash
PYTHONPATH=src JAX_PLATFORMS=cuda python benchmarks/high_moi_padding.py \
  --cells 200000 --elements 5000 --active 30 --width 30 --genes 32 \
  --repeats 5 --out-dir /tmp/perturbo-padding-30
```

The no-padding fixture differs slightly at the first row. Each original/candidate
comparison uses exactly the same inputs. Random duplicates are deliberately
summed in both the dense-reference calculation and the tested products.

## Next test: a short real-data SVI comparison

Compare the base commit `a448511` and this branch in separate checkouts, using
the same data, seeds, cell selection, gene-block width, and optimizer settings.
Keep every cell and predictor. Use a small equal number of effect updates and
`--no-crt` to isolate fitting; use separate output directories. Distinguish first
compilation from warmed updates, then compare effect estimates and loss curves
within float32 tolerances. GPU reductions can differ in rounding order, so
bitwise equality on CPU is not a GPU requirement.

Return both probe JSON files, GPU model, JAX/jaxlib versions, actual mean/max
active elements per cell, gene-block width, and synchronized SVI update times.
If the isolated adjoint improves but SVI does not, profile the actual SVI GPU
kernels before proceeding. If padding removal has little effect, that result
rules against this specific contention hypothesis; it does not justify changing
the fitting schedule or assuming the 24-hour estimate is unavoidable.

The implementation/tests and testing handoff are separate commits so the small
production change can be cherry-picked independently.
