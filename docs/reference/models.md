# NumPyro models and distributions

`perturbo.model` defines probabilistic model callables for custom NumPyro
workflows. A model invocation creates sample sites and returns observations
(or `None` when observation sampling is skipped); it does not return a fitted
`PerTurboModel`. For ordinary fitting, use the
[model or core interfaces](data_and_fitting.md).

## Likelihood-specific entry points

The following public functions bind the likelihood argument of `BaseModel`.
Their remaining parameters follow the shared contract below.

| Callable | Bound likelihood | Observation model |
| --- | --- | --- |
| `NegBinModel` | `nb` | Negative binomial |
| `CensoredNegativeBinomialModel` | `censored_nb` | Negative binomial with upper-tail censoring |
| `LogNormalNegativeBinomialModel` | `lnnb` | Log-normal mixture of negative binomials |
| `MixtureNegativeBinomialModel` | `mixture_nb` | Baseline/outlier negative-binomial mixture |

```{eval-rst}
.. autofunction:: perturbo.NegBinModel

.. autofunction:: perturbo.CensoredNegativeBinomialModel

.. autofunction:: perturbo.LogNormalNegativeBinomialModel

.. autofunction:: perturbo.MixtureNegativeBinomialModel
```

### Shared argument contract

| Parameter | Meaning |
| --- | --- |
| `counts` | Nonnegative integer-valued count array, `(n_cells, n_genes)`, or `None` for generative use with explicit dimensions. |
| `pert_id` | Element indices `(n_cells,)` for mutually exclusive assignments, or a design with logical shape `(n_cells, n_elements)` for additive co-occurring predictors. Internal indexed sparse designs are supported by the design operations. |
| `size_factors` | Observed log offsets, conventionally `(n_cells, 1)`; `None` creates a cell-local latent size factor. |
| `covariates` | Numeric design `(n_cells, n_covariates)` after preprocessing, or `None`. |
| `guide_matrix` | Guide design with logical shape `(n_cells, n_guides)`; used for optional guide random effects in these base-model callables. |
| `guide_to_element`, `guide_effect_strategy` | Accepted for a common calling convention but ignored by `BaseModel`. Guide-aware fitting uses `GuideSharedEffectModel` instead. |
| `cell_mask` | Boolean valid-cell mask, `(n_cells,)` or `(n_cells, 1)`, including padded rows. Missing means all rows are valid. |
| `num_cells`, `num_genes`, `num_perts`, `num_guides` | Explicit full data dimensions. Inferable dimensions may be omitted; an index-form `pert_id` needs an explicit `num_perts`. |
| `num_factors` | Optional number of latent factors. `None` disables the factor sites at this level; higher-level callers normalize their zero-factor setting. |
| `prior` | Effect prior: `normal` or `cauchy`. |
| `subsample_size`, `cell_idx` | Optional cell subsampling controls for advanced NumPyro use. `None` uses full data. Subsampling changes the optimization regime. |
| `likelihood` | Bound to the value above by each public callable. |
| `skip_obs_sampling` | Create latent sites but return `None` before the observation site. |
| `guide_random_effects` | Add guide-by-gene random deviations when guide data are available. |
| `fit_perturbation_dispersion` | Learn nonnegative perturbation contributions to inverse dispersion. |
| `perturbation_dispersion_prior_rate` | Exponential prior rate for those inverse-dispersion contributions. |
| `count_censoring_threshold` | Per-gene upper cutoff for the censored likelihood; supplied by the high-level preprocessing path. |

The guide-aware implementation retains the same cell/gene axes and additionally
requires both guide assignments and their element mapping. Prefer the high-level
fitter to select this implementation and build its sparse design. Do not replace
a multi-guide cell by one categorical perturbation index.

## Sparse design representation

The core loaders construct this representation when appropriate. Logical axes
remain cells by predictors; callers should use the loaders to preserve masks
and mappings rather than construct it from unvalidated guide data.

```{eval-rst}
.. autoclass:: perturbo.sparse_design.IndexedDesignMatrix
```

## Distribution interfaces

These are NumPyro distributions with scalar events and broadcast batch shapes.
They expose the usual `log_prob`, `sample`, `expand`, `mean`, and `variance`
interfaces. Their parameters use NumPyro's negative-binomial logits convention:
without the log-normal mixture, the mean is `total_count * exp(logits)`.

```{eval-rst}
.. autoclass:: perturbo.CensoredNegativeBinomial
   :members: log_prob, sample, expand, mean, variance

.. autoclass:: perturbo.LogNormalNegativeBinomial
   :members: log_prob, sample, expand, mean, variance
```

For censored observations, `log_prob` evaluates an NB point mass at values at or
below the cutoff, and `log P(Y > cutoff)` above it. `sample`, `mean`, and `variance`
refer to the underlying uncensored distribution. The upper-tail calculation
stops the gradient through the NB concentration in its CDF; it does not implement
the full concentration derivative of that tail term.

For the log-normal mixture, `log_prob` integrates with Gauss–Hermite quadrature,
whereas `sample` draws a continuous normal perturbation of the logits. Increasing
`num_quad_points` controls the density approximation; it does not change the
sampling distribution. These likelihood variants are not interchangeable with
the NB configuration supported by the default CRT.
