# API

## Primary workflow

- `perturbo.PERTURBO` / `perturbo.PerTurboModel`: notebook-friendly two-stage fitting, posterior access, and bundles.
- `perturbo.setup_mudata` and `perturbo.get_mudata_setup`: register and inspect MuData inputs.
- `perturbo.fit_from_path`: Python equivalent of the `perturbo` CLI.

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

## Functional API

- `perturbo.load_controls` and `perturbo.load_analysis_cells`
- `perturbo.fit_control` and `perturbo.fit_perturbation_effects`
- `perturbo.summarize_betas`

## Results and persistence

- `perturbo.build_guide_effects_df`
- `perturbo.load_fit_bundle` and `perturbo.save_fit_bundle`
- `perturbo.simulate_data_from_trained_model` and `perturbo.save_simulated_mudata`

The deprecated PyTorch interface is isolated under `perturbo.legacy` and
requires the `legacy` extra.
