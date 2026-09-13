# Data loading and two-stage fitting

`perturbo.core` contains the low-level NumPyro/JAX data loaders and the two
fitting stages. Most analyses should use `PerTurboModel` or `fit_from_path`,
which coordinate validation, chunking, persistence, and result tables. Use the
functions on this page when integrating PerTurbo into a custom in-memory
pipeline and when you can preserve every array's identity and order yourself.

See the [Python API guide](../python_api.md) for the higher-level workflow,
[data preparation](../data_preparation.md) for AnnData/MuData layout, and the
[method](../method.md) for the statistical model.

## Two-stage contract

PerTurbo first fits baseline expression and nuisance parameters on control
cells. It then conditions on that `ControlFit` while estimating perturbation
effects from analysis cells. The following must agree between stages:

- gene identity and order;
- likelihood and latent-factor specification;
- covariate columns, transformations, and order;
- guide random-effect configuration; and
- the meaning and centering of size-factor offsets.

Fit covariate preprocessing on controls and reuse it for analysis cells. When
library-size offsets are computed, also carry the control-derived log-mean
center into the analysis loader.

```python
from perturbo.core import (
    fit_control,
    fit_perturbation_effects,
    load_analysis_cells,
    load_controls,
)

control_data, covariate_state = load_controls(
    mdata,
    modality_key="rna",
    perturbation_modality_key="grna",
    control_selector="non-targeting",
    continuous_covariates=["n_counts"],
    batch_covariate="batch",
    return_covariate_transform_state=True,
)

analysis_data = load_analysis_cells(
    mdata,
    modality_key="rna",
    perturbation_modality_key="grna",
    perturbation_element_varm_key="element_targeted",
    perturbation_element_names_uns_key="element_names",
    covariate_transform_state=covariate_state,
    library_size_center_log_mean=control_data.library_size_center_log_mean,
)

control_fit = fit_control(
    control_data,
    use_observed_size_factors=True,
)
beta_fit = fit_perturbation_effects(
    analysis_data,
    control_fit,
    use_observed_size_factors=True,
)
```

Both fitting functions use full-batch SVI by default (`minibatch_size=None`). A
positive minibatch size changes the optimization regime; it is not equivalent
to merely splitting an otherwise identical update. The low-level default is
1,000 optimizer steps with an Adam step size of `0.01`.

## Array contracts

Let $N$ be cells, $G$ genes, $P$ perturbation elements, $Q$ guides, $C$
covariates, and $K$ latent factors.

| Value | Shape | Ordering and meaning |
| --- | --- | --- |
| `PerTurboData.counts` | `(N, G)` | Raw finite, non-negative integer counts; columns match `gene_names` |
| one-label `pert_id` | `(N,)` | Integer codes indexing `pert_names` |
| matrix `pert_id` | `(N, P)` | Dense or indexed element membership; columns match `pert_names` and may contain multiple active elements per cell |
| `size_factors` | `(N, 1)` | Centered log-library offset aligned to cells |
| `covariates` | `(N, C)` | Transformed continuous and batch design in `covariate_names` order |
| `cell_mask` | `(N,)` | Boolean validity mask; false rows are padding and do not enter the likelihood |
| `guide_matrix` | `(N, Q)` | Optional dense or indexed guide membership in `guide_names` order |
| `guide_to_element` | `(Q, P)` | Optional guide-to-parent map; guide-sharing fits require one parent per non-padding guide |
| `ControlFit.beta_0`, `theta` | `(G,)` | Stage-one baseline and negative-binomial dispersion |
| `ControlFit.covariate_coef` | `(C, G)` | Stage-one adjustment coefficients |
| `ControlFit.factor_loadings` | `(K, 1, G)` | Optional stage-one latent-factor loadings |
| `ControlFit.factor_scores` | `(K, N, 1)` | Optional scores for the stage-one control cells |
| `BetaFit.posterior_mean`, `posterior_scale`, `z_values` | `(P, G)` | Stage-two element effects in perturbation-major order |
| guide posterior arrays | `(Q, G)` | Present only for guide-specific strategies or summaries |

Name lists are part of the data contract rather than display-only metadata.
Reordering a matrix without applying the same permutation to its names changes
which biological element or gene an estimate represents.

## Counts and offsets

Loaders reject normalized, fractional, negative, empty, or non-finite count
matrices. Keep raw counts in the selected expression modality's `.X`; store
normalized expression elsewhere.

There are three offset inputs:

1. `size_factor_key` reads an already transformed and centered log offset and
   does not center it again.
2. `library_size_key` reads positive integer library sizes, applies `log1p`,
   and subtracts the control-derived mean.
3. With neither key, loaders derive library sizes from row totals over the full
   expression panel, then apply the same transform.

Gene-chunked loading must not calculate offsets from the current gene slice.
Pass `full_panel_library_sizes`, or let backed loading read full-panel totals in
bounded row blocks, and reuse `library_size_center_log_mean` from controls.

The loaders prepare `size_factors`, but the low-level fitting functions condition
on them only when `use_observed_size_factors=True`. With the default `False`,
the model infers latent size factors and uses the prepared values as
initialization.

## Covariates

`fit_covariate_transform` imputes continuous values from control medians,
optionally applies `log1p` to count-like columns, z-scores with control moments,
and encodes batch levels relative to the most frequent control level. It removes
zero-variance features. `apply_covariate_transform` reproduces the retained
feature order without learning from analysis cells. New batch levels map to all
zero indicators rather than changing the reference design.

## Perturbation and guide models

An `obs` perturbation column represents one label per cell. A perturbation
modality supports multiple guides or elements per cell and can remain compact as
an indexed design. If `selected_perturbations` would retain a cell while dropping
another co-occurring active predictor, loading fails: fitting that incomplete
design would confound the retained effect. Chunk high-MOI fits over genes while
keeping every perturbation predictor.

The default guide strategy is `shared`. `relative` learns a guide-by-gene
efficiency multiplying the parent element effect and requires
`retain_guide_structure=True` plus a valid guide-to-element map. The
`absolute` activity mode is reserved but currently rejected by stage-two SVI.

Guide random effects must be fitted in stage one if stage two requests them.
With retained guide structure, the learned scale enters the guide-level model;
without a guide map, stage two uses it to inflate element posterior scales.
Perturbation-specific dispersion is limited to negative-binomial likelihoods.
For cells with multiple active perturbations, retain guide structure so the
dispersion model has the required guide-level assignment.

Latent factors are disabled with `num_factors=None` in the core functions. A
positive value initializes factors by PCA in stage one; pass the same value and
the matching `ControlFit` to stage two. The integer `0` is not the no-factor
sentinel for these low-level calls.

## Data and fit objects

```{eval-rst}
.. autoclass:: perturbo.core.SVIConfig

.. autoclass:: perturbo.core.PerTurboData

.. autoclass:: perturbo.core.CovariateTransformState

.. autoclass:: perturbo.core.BaselinePosteriorSummary

.. autoclass:: perturbo.core.ControlFit

.. autoclass:: perturbo.core.BetaFit
```

## Covariate preprocessing

```{eval-rst}
.. autofunction:: perturbo.core.fit_covariate_transform

.. autofunction:: perturbo.core.apply_covariate_transform
```

## Loading

```{eval-rst}
.. autofunction:: perturbo.core.load_controls

.. autofunction:: perturbo.core.load_analysis_cells
```

## Fitting and summaries

```{eval-rst}
.. autofunction:: perturbo.core.fit_control

.. autofunction:: perturbo.core.fit_perturbation_effects

.. autofunction:: perturbo.core.summarize_betas
```
