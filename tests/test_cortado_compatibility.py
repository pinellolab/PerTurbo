"""Compatibility tests for data and bundles written by the Cortado prototype."""

from __future__ import annotations

import json

import anndata as ad
import mudata as md
import numpy as np
import pytest

from perturbo.io import get_mudata_setup, load_fit_bundle, save_fit_bundle


def _legacy_mdata() -> md.MuData:
    rna = ad.AnnData(X=np.asarray([[1, 0], [0, 2]], dtype=np.int32))
    rna.obs_names = ["cell_a", "cell_b"]
    pert = ad.AnnData(X=np.asarray([[1], [0]], dtype=np.int8))
    pert.obs_names = rna.obs_names.copy()
    result = md.MuData({"rna": rna, "grna": pert})
    result.uns["_cortado_setup"] = {
        "rna_modality": "rna",
        "perturbation_modality": "grna",
        "perturbation_layer": None,
        "batch_key": None,
        "library_size_key": "_library_size",
        "size_factor_key": "_size_factor",
        "continuous_covariates_keys": [],
        "gene_by_element_key": None,
        "guide_by_element_key": None,
        "rna_element_uns_key": None,
        "guide_element_uns_key": None,
        "gene_name_key": None,
        "control_substring": None,
    }
    return result


def test_legacy_registration_is_upgraded_in_memory_and_on_save(tmp_path) -> None:
    mdata = _legacy_mdata()

    with pytest.warns(DeprecationWarning, match="_cortado_setup"):
        setup = get_mudata_setup(mdata)

    assert setup.rna_modality == "rna"
    assert "_perturbo_setup" in mdata.uns
    assert "_cortado_setup" not in mdata.uns

    save_fit_bundle(
        tmp_path / "bundle",
        metadata={"likelihood": "nb"},
        mdata=mdata,
        control_arrays={},
        beta_arrays={},
    )
    loaded = load_fit_bundle(tmp_path / "bundle")
    assert loaded["metadata"]["producer"] == "perturbo"
    assert loaded["metadata"]["bundle_version"] == 2
    assert "_perturbo_setup" in loaded["mdata"].uns
    assert "_cortado_setup" not in loaded["mdata"].uns


def test_legacy_light_bundle_metadata_loads_from_source_data(tmp_path) -> None:
    mdata = _legacy_mdata()
    root = tmp_path / "light"
    root.mkdir()
    metadata = {
        "bundle_format": "light_model_params",
        "bundle_version": 1,
        "likelihood": "nb",
        "effect_prior_dist": "normal",
        "setup": mdata.uns["_cortado_setup"],
        "source": {"path": "unused-with-source-data.h5mu"},
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    np.savez(root / "control_fit.npz")
    np.savez(root / "beta_fit.npz")

    loaded = load_fit_bundle(root, source_data=mdata)

    assert loaded["metadata"]["bundle_version"] == 1
    assert "_perturbo_setup" in loaded["mdata"].uns
