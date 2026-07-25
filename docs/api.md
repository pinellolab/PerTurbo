# API

## Primary workflow

- `perturbo.PERTURBO` / `perturbo.PerTurboModel`: notebook-friendly two-stage fitting, posterior access, and bundles.
- `perturbo.setup_mudata` and `perturbo.get_mudata_setup`: register and inspect MuData inputs.
- `perturbo.fit_from_path`: Python equivalent of the `perturbo` CLI.

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
