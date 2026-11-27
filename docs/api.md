# Python API

## Fit a dataset from disk

From the repository root, fit the bundled low-MOI screen and inspect discoveries:

```python
from pathlib import Path
import mudata
from perturbo import PerTurboModel, fit_from_path

out = Path("runs/replogle-python")
fit_from_path("docs/tutorials/inputs/replogle.h5ad", out,
              perturbation_key="gene_transcript", control_substring="non-targeting",
              gene_name_key="gene_name", library_size_key="UMI_count", batch_covariate="gem_group")
path = out / "fitted.h5mu"
effects = mudata.read_h5mu(path).uns["_perturbo"]["effects"]
hits = effects.loc[effects.q_global < 0.1, ["element", "gene", "posterior_mean", "q_global"]]
model = PerTurboModel.load(path)
```

The command-line interface and `fit_from_path` run the same analysis. Each row of `effects` identifies an element and a response gene. A negative `posterior_mean` indicates reduced expression. Choose the q-value column for the set you want to interpret: `q_global` across the screen, `q_element` within one element, or `q_cis` across a supplied candidate list.

## Work with an existing fit

`PerTurboModel.load` restores the fitted baseline and perturbation effects. Use `model.get_element_effects()` to inspect estimates and uncertainty. To filter discoveries, use the saved `effects` table shown above. For a relative fit, `model.guide_efficacy` has one value for each guide and response gene. In a shared fit the guide multipliers are fixed at one.

For a worked example, start with the [enhancer analysis](tutorials/gasperini.ipynb) or the [single-construct analysis](tutorials/replogle.ipynb). See [fitted outputs](fitted_export_pilot.md) for the file layout.

## Callable reference

The reference pages use signatures and docstrings from the installed package. `PERTURBO` is an alias for `PerTurboModel`.

```{toctree}
:maxdepth: 1

reference/fitting
reference/components
reference/preprocessing
reference/simulation
```
