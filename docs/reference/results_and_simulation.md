# Results and simulation

`perturbo.results` converts posterior arrays into stable long-form tables, and
`perturbo.simulation` draws a new MuData object from a trained or loaded model.
For guidance on choosing columns and significance measures, see the
[Results guide](../results_guide.md). The generative model is described in
[Method](../method.md).

## Array and row-order contracts

Element effect arrays use shape `(n_elements, n_genes)`. Guide effects use `(n_guides, n_genes)`; guide efficacy uses `(n_guides,)`,
one value per guide. Effect-table builders flatten the effect arrays
in C row-major order: all genes for the first element or guide, then all genes
for the second. Thus row `i * n_genes + j` represents pair `(i, j)`.
`extra_columns` passed to a standard element builder must have exactly the same
shape and follow the same ordering.

The compact `build_element_effects_df` and `build_guide_effects_df` functions use
the columns `loc`, `scale`, `z_value`, and `q_value`. In these accessor tables,
`q_value` is the two-sided standard-Normal tail probability; it is not a
Benjamini-Hochberg adjusted value. Prefer the standard table functions for new
code.

The standard element table has this base schema:

| Column | Meaning |
| --- | --- |
| `method` | caller-supplied method label |
| `element`, `gene` | tested pair |
| `posterior_mean`, `posterior_scale` | effect summary on the model's log fold-change scale |
| `z_value` | posterior mean divided by posterior scale |
| `posterior_prob` | two-sided standard-Normal tail probability from `z_value` |
| `empirical_p_value` | two-sided fitted zero-centered t-null tail; missing when no finite `null_z_values` are supplied |

Additional matrices, including CRT p-values and diagnostics, are appended under
their dictionary keys. Tail columns remain float64 so very small probabilities
are preserved. `iter_standard_element_effects_frames` yields bounded row blocks
in exactly the same order as the in-memory builder.

## Building and streaming result tables

```python
from perturbo.results import build_standard_element_effects_df

effects = build_standard_element_effects_df(
    method="perturbo",
    effect_loc=beta_loc,       # (elements, genes)
    effect_scale=beta_scale,   # (elements, genes)
    element_names=element_names,
    gene_names=gene_names,
    null_z_values=control_element_z,
    extra_columns=crt_columns,
)
```

For a large grid, write blocks directly. Without `requested_pairs` the writer
returns `None`; with it, the file still contains the full grid and the return
value is the restricted in-memory table. Any q-value columns in that restricted
table are recalculated for the restricted hypothesis family.

```python
from perturbo.results import write_standard_element_effects_parquet

selected = write_standard_element_effects_parquet(
    "element_effects.parquet",
    method="perturbo",
    effect_loc=beta_loc,
    effect_scale=beta_scale,
    element_names=element_names,
    gene_names=gene_names,
    extra_columns=crt_columns,
    row_block_size=250_000,
    requested_pairs=requested_pairs,
)
```

## Simulating from a trained model

`simulate_data_from_trained_model` requires a model with both control and beta
fits. `guide_obs` is `(cells, guides)`, `guide_by_element` is
`(guides, elements)`, `element_by_gene_lfc` is `(elements, model_genes)`, and
`guide_efficacy` has one value per guide. Optional `cell_indices` choose source
cell covariates and size factors; optional `gene_indices` choose model genes.
The returned MuData contains RNA counts and the supplied guide observations,
then records a valid PerTurbo setup using the trained model's modality and key
configuration. Vary `seed` between replicates.

```python
from perturbo.simulation import save_simulated_mudata, simulate_data_from_trained_model

simulated = simulate_data_from_trained_model(
    model,
    guide_obs=guide_matrix,
    guide_by_element=guide_to_element,
    element_by_gene_lfc=lfc,
    guide_efficacy=efficacy,
    seed=17,
)
path = save_simulated_mudata(simulated, "simulated.h5mu")
```

## Posterior accessors

```{eval-rst}
.. autoclass:: perturbo.results.PosteriorMedians
      :members:
```
```{eval-rst}
.. autoclass:: perturbo.results.PosteriorParameter
```
```{eval-rst}
.. autofunction:: perturbo.results.extract_parameter_table
```

## Result tables

```{eval-rst}
.. autofunction:: perturbo.results.build_element_effects_df
```
```{eval-rst}
.. autofunction:: perturbo.results.build_standard_element_effects_df
```
```{eval-rst}
.. autofunction:: perturbo.results.iter_standard_element_effects_frames
```
```{eval-rst}
.. autofunction:: perturbo.results.write_standard_element_effects_parquet
```
```{eval-rst}
.. autofunction:: perturbo.results.build_guide_effects_df
```
```{eval-rst}
.. autofunction:: perturbo.results.build_guide_efficiency_df
```

## Simulation functions

```{eval-rst}
.. autofunction:: perturbo.simulate_data_from_trained_model
```
```{eval-rst}
.. autofunction:: perturbo.save_simulated_mudata
```
