# Contributing

Use Python 3.11+ and `uv` for local development.

```bash
uv sync --group test --group dev
uv run pytest
uv run ruff check src tests
uv build
```

Keep production code under `src/perturbo`. Benchmark runners, applications,
notebooks, and manuscript analysis belong in the Cortado research repository.
The default test environment must not acquire Torch, Pyro, or scvi-tools; test
the deprecated implementation only with `uv sync --extra legacy --group test`.

## Where the source of truth lives

`src/perturbo` is a rename of the analysis package developed in the Cortado
research repository, and changes usually start there. When you edit shared
implementation files, make the same edit in both places; a change that lands
only here will be overwritten by the
next port. What this repository owns outright is the packaging: the Dockerfile,
the workflows, the legacy extra and its isolation test.

## Two constraints that are easy to break

The package enables float64 at import, but **the SVI parameters must stay
float32**. Under float64 the array
constructors and `AutoNormal`'s scale initialisation produce float64, the model
casts its design matrices to the parameter dtype, and every cells-by-genes
intermediate of the likelihood doubles; a screen that fitted on a 40 GB card
then does not. `_pin_svi_params_float32` enforces it after `svi.init` and
`tests/test_svi_stays_float32.py` guards it, so keep new initial values and
cell-axis arrays in float32.

**The step size and the step count are a pair.** Adam moves a coefficient by
roughly the step size per step, so their product must exceed the largest effect
in nats with margin. The defaults, 0.01 and 500 steps per stage, were chosen
against simulated screens with known effects; lowering one without raising the
other shrinks large effects toward zero.
