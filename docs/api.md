# API reference

The reference is generated from the current Python signatures and NumPy-style
docstrings using Sphinx autodoc. Begin with the model and I/O page for notebook
workflows, or the file facade for CLI-style analyses from Python.

| Module | Responsibility |
| --- | --- |
| `perturbo.inference` | `PerTurboModel`: in-memory fitting, model accessors, persistence |
| `perturbo.io` | Register MuData fields and read/write fit bundles |
| `perturbo.api` | `fit_from_path`: run the file-oriented workflow |
| `perturbo.core` | Data containers, covariate transforms, baseline and effect fitting |
| `perturbo.crt` | Conditional randomization tests, score state, result accumulation |
| `perturbo.results` | Posterior summaries, effect tables, result serialization |
| `perturbo.simulation` | Generate counts from a fitted model |
| `perturbo.model` | NumPyro model callables and likelihood selection |
| `perturbo.preprocessing` | Count thresholds and dataset-specific preparation |

All names exported by `perturbo.__all__` are documented below. `PERTURBO` is an
alias of `PerTurboModel`. Lower-level module APIs expose the ordering and shape
contracts required to compose custom workflows; they do not supply every check
performed by the CLI. Underscore-prefixed helpers are implementation details.
The deprecated `perturbo.legacy` implementation is outside this v2 reference.

```{toctree}
:maxdepth: 1

reference/model_and_io.md
reference/data_and_fitting.md
reference/crt.md
reference/results_and_simulation.md
reference/models.md
reference/preprocessing.md
```

## Conventions

Counts have shape `(n_cells, n_genes)`, guide assignments have logical shape
`(n_cells, n_guides)`, and guide-to-element mappings have shape
`(n_guides, n_elements)`. Effect arrays use `(n_elements, n_genes)`.
Every paired array must use the same cell, gene, guide, and element order.
Sparse storage does not change those logical axes.

Size factors passed to low-level model calls are **log offsets**, not raw library
sizes. User-facing setup and file loaders perform their documented transforms.
Effect coefficients are on the natural-log scale; `exp(beta)` is the expression
ratio under the fitted model. The [results guide](results_guide.md) distinguishes
variational uncertainty, historical tail-score column names, and CRT p-values.

Importing `perturbo` enables JAX float64 globally. The likelihood fit retains its
explicit float32 arrays, but unrelated JAX code in the same process can observe
the changed default dtype behavior. Use a separate process when another library
requires a different JAX configuration.

For complete examples, see the [Python workflow](python_api.md) and
[small CPU analysis](quickstart.md). Python facade defaults differ from CLI
defaults; the relevant signatures and reference pages state those differences.
