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
research repository, and changes usually start there. When you edit the
conditional randomization test, the command line, or `core.py`, make the same
edit in both places; a change that lands only here will be overwritten by the
next port. What this repository owns outright is the packaging: the Dockerfile,
the workflows, the legacy extra and its isolation test.

## Two constraints that are easy to break

The package enables float64 at import, for the conditional randomization test's
tails, but **the SVI parameters must stay float32**. Under float64 the array
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

## Building the documentation

Install the package and its documentation dependencies from the locked
environment, then run the same strict Sphinx build used in CI:

```bash
uv sync --locked --extra docs
JAX_PLATFORM_NAME=cpu MPLBACKEND=Agg \
  uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

`-W` turns warnings, including unresolved cross-references and malformed
directives, into build failures. `--keep-going` reports all warnings in one
run. Notebook execution is disabled in `docs/conf.py`, so this check imports
the current checkout for API documentation but does not execute notebooks.

Write public Python docstrings in NumPy style, with explicit `Parameters`,
`Returns`, and `Yields` sections where applicable. Document each public object
on its canonical page under `docs/reference/` so Sphinx has one primary target
for cross-references. MyST pages can wrap standard reStructuredText autodoc
directives in an `eval-rst` fence when directive options are clearer in that
form:

````markdown
```{eval-rst}
.. autofunction:: perturbo.results.build_standard_element_effects_df
```
````

Keep explanatory shape, ordering, and state contracts beside the canonical
directive rather than duplicating the generated signature on several pages.
