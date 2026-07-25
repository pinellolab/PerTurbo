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
