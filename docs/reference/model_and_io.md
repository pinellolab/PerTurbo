# Model and data registration reference

This page is the canonical generated reference for the in-memory model and its
MuData registration and persistence functions. The [Python API guide](../python_api.md)
shows how these calls fit together.

## Registration metadata

`setup_mudata` stores modality and column names in
`mdata.uns["_perturbo_setup"]`. It also computes missing RNA library sizes,
centered log size factors, and gene means. The stored size factor is a centered
log offset: it is added to the log expected count, rather than multiplied into
the count directly.

```{eval-rst}
.. autoclass:: perturbo.MuDataSetup
   :members: to_json_dict, from_json_dict
```

```{eval-rst}
.. autofunction:: perturbo.setup_mudata
```

```{eval-rst}
.. autofunction:: perturbo.get_mudata_setup
```

## Fitted model

`PerTurboModel` loads controls and analysis cells from the registered modalities.
`train` runs the baseline and effect stages and retains their arrays on the
object. It does not run the CRT. Array axes use cells by genes for counts,
elements by genes for element effects, and guides by genes for guide-specific
effects where present.

```{eval-rst}
.. autoclass:: perturbo.PerTurboModel
   :members: setup_mudata, train, posterior_medians, posterior_parameter_table, get_element_effects, get_guide_effects, guide_efficacy, view_anndata_setup, save, load
```

`perturbo.PERTURBO` is an alias of this class.

## File facade

`fit_from_path` writes the CLI-style output directory. Its defaults come from
the Python signature: step size is `0.003`, normalization mode is `infer`, and
minibatch values of zero mean full batch. Despite its name, `crt=False` omits
CRT arguments and leaves CLI automatic CRT selection active; it does not force
testing off. Use CLI `--no-crt` when testing must be disabled. It does not expose
the CLI's `gene_chunk_size` option in this release.

```{eval-rst}
.. autofunction:: perturbo.fit_from_path
```

## Bundle persistence

A complete bundle contains registered MuData, JSON metadata, and control and
effect array archives. A light bundle instead refers to source data and may
carry saved cell indices and normalized offsets. Loading a light bundle checks
the source gene order and restores saved offsets when available.

```{eval-rst}
.. autofunction:: perturbo.save_fit_bundle
```

```{eval-rst}
.. autofunction:: perturbo.load_fit_bundle
```
