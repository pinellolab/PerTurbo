# Python API

`PerTurboModel` works with an in-memory `MuData` object and exposes fitted-model
accessors. `fit_from_path` runs the file-oriented workflow used by the command
line. See the [quickstart](quickstart.md), [data preparation](data_preparation.md),
[running analyses](running_analyses.md), [method](method.md), [results](results_guide.md),
and [installation](installation.md) guides for the surrounding workflow.

## Register MuData

`setup_mudata` records where PerTurbo should find RNA counts, guides, element
mappings, covariates, and controls. It mutates the object. When named library
size, size factor, or gene-mean fields are absent, it computes them from the RNA
matrix; this is more than simply registering paths and column names.

```python
import perturbo

setup = perturbo.setup_mudata(
    mdata,
    modalities={"rna_layer": "rna", "perturbation_layer": "grna"},
    guide_by_element_key="element_targeted",
    guide_element_uns_key="element_names",
    control_substring="non-targeting",
)
```

The `modalities` keys are fixed; their values name entries in `mdata.mod`.
Registration is written to `mdata.uns["_perturbo_setup"]`.
`get_mudata_setup(mdata)` reads it. A legacy
`_cortado_setup` payload is upgraded in memory when read.

Key parameters are:

- `guide_by_element_key`: perturbation `.varm` key whose rows are guides and
  columns are elements.
- `guide_element_uns_key`: optional `.uns` element-name key when the mapping has
  no column labels.
- `control_substring`: substring identifying control guides or elements. Setup
  attempts limited inference when omitted; inspect the returned setup.
- `batch_key` and `continuous_covariates_keys`: RNA `.obs` adjustment columns.
- `library_size_key` and `size_factor_key`: RNA `.obs` columns to reuse. Setup
  creates `_library_size` and `_size_factor` when they are absent.
- `perturbation_layer`: optional perturbation layer; otherwise `.X` is used.
- `gene_name_key`: optional RNA `.var` column used instead of `.var_names`.

## Fit an in-memory model

```python
model = perturbo.PerTurboModel(
    mdata,
    likelihood="nb",
    effect_prior_dist="normal",
    guide_effect_strategy="shared",
)
model.train(steps=500, lr=0.01, batch_size=None, accelerator="cpu")
```

`batch_size=None` and `batch_size=0` request full-batch fitting. A positive value
enables cell minibatching for both stages. Full batch is the clearest baseline
for convergence comparisons; minibatching changes the optimization regime and
is not a transparent memory switch.

`train` fits the control baseline and perturbation effects. It does **not** run
the CRT or add `crt_*` columns. Use `fit_from_path(..., crt=True)` or the CLI for
the CRT.

`steps` sets a raw SVI step count for both stages; `control_steps` and
`beta_steps` override individual stages. `max_epochs` requests dataset passes;
`control_epochs` and `beta_epochs` override individual stages. `lr` replaces the
configured SVI step size. `accelerator` selects a device family and `device` can
select a specific device. Likelihood, prior, guide strategy, latent factors,
censoring, guide random effects, and perturbation dispersion belong on
`PerTurboModel(...)`, not `train`.

```python
effects = model.get_element_effects()
guide_effects = model.get_guide_effects()  # requires a guide-to-element map
medians = model.posterior_medians()
losses = model.history["elbo_train"]

model.save("perturbo_bundle", save_anndata=True, overwrite=False)
restored = perturbo.PerTurboModel.load("perturbo_bundle")
```

Consult the [results guide](results_guide.md) before interpreting historically
named probability or q-value fields in accessor tables.

## Run the file workflow from Python

`fit_from_path` translates keyword arguments to the CLI implementation, sharing
its validation, chunking, outputs, and light-bundle behavior. This example
explicitly selects the all-cells CRT pool for a high-MOI analysis. For a low-MOI
screen like the synthetic quickstart, select `crt_pool="control-anchored"`
instead; see [CRT pools](running_analyses.md) for the different hypotheses.

```python
perturbo.fit_from_path(
    "screen.h5mu",
    "perturbo_outputs/run",
    modality_key="rna",
    perturbation_modality_key="grna",
    perturbation_element_varm_key="element_targeted",
    perturbation_element_names_uns_key="element_names",
    control_substring="non-targeting",
    likelihood="negbin",
    size_factor_mode="observed",
    num_steps_control=500,
    num_steps_betas=500,
    step_size=0.01,
    minibatch_size_betas=0,
    crt=True,
    crt_pool="all-cells",
    crt_mechanism="propensity",
    crt_tail_families=("saddlepoint",),
    crt_saddlepoint_only=True,
)
```

Defaults that are easy to confuse with other interfaces:

- `crt=False` does **not** force testing off in this release: it omits the
  CRT arguments, so the CLI can still select CRT automatically. Set `crt=True`
  to forward an explicit CRT configuration; use CLI `--no-crt` to disable testing.
- `step_size=0.003` and `size_factor_mode="infer"`.
- minibatch sizes are `0`, meaning full batch.
- `return_model=False`; the function normally writes files and returns `None`.
- this release has no `gene_chunk_size` keyword on `fit_from_path`. The CLI has
  that option and can select gene blocks automatically.

These Python defaults are defined by the function signature and are not all the
same as invoking the CLI without corresponding flags. False values for
`crt_saddlepoint_only`, `crt_allow_unconverged_baseline`, and
`crt_polish_baseline` likewise omit the flag and leave the CLI default active. Set important values
explicitly in a reproducible analysis.

Parameter groups include:

- Input mapping: `modality_key`, `perturbation_modality_key`,
  `perturbation_layer`, `perturbation_element_varm_key`, and element-name key.
- Adjustment: `continuous_covariates`, `batch_covariate`, `size_factor_mode`,
  `size_factor_key`, and `library_size_key`.
- Training: `num_steps*` or `num_epochs*`, `step_size`, `num_particles`,
  `minibatch_size*`, `prior`, `likelihood`, and `num_factors`.
- Scale: `backed`, `max_control_cells`, `perturbation_chunk_size`, and
  `max_chunk_size`.
- Testing: `crt`, pool, mechanism, resample and tail options, baseline polishing,
  and `crt_only`.
- Reporting: `pairs_to_test`, which recomputes q-values within the requested
  family without changing estimates or p-values.

Set `return_model=True` to receive a loaded model after files are written. For
large analyses, prefer the CLI in [running analyses](running_analyses.md), which
exposes the complete execution surface and makes the invocation easy to record.

## Low-level functions

Advanced workflows can call `load_controls`, `load_analysis_cells`,
`fit_control`, and `fit_perturbation_effects`. The caller must keep cell, gene,
covariate, guide, and element ordering aligned. `fit_control` returns a
`ControlFit`; pass that baseline with compatible `PerTurboData` to
`fit_perturbation_effects`. See the generated [API reference](api.md).
