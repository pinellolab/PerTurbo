"""Tests for high-MOI perturbation matrix inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
import mudata as md
import pytest
import scipy.sparse as sp
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from perturbo.api import fit_control, load_analysis_cells, load_controls
from perturbo.model import NegBinModel


def _make_mudata() -> md.MuData:
    cell_names = ["c0", "c1", "c2", "c3"]
    counts = np.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
        ],
        dtype=np.int32,
    )
    obs = pd.DataFrame(
        {
            "label": ["ctrl", "pertA", "pertA+pertB", "ctrl"],
        },
        index=cell_names,
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    adata_rna = ad.AnnData(X=counts, obs=obs, var=var)

    pert_matrix = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.int8,
    )
    pert_var = pd.DataFrame(index=["pertA", "pertB", "pertC"])
    adata_pert = ad.AnnData(
        X=pert_matrix,
        obs=pd.DataFrame(index=cell_names),
        var=pert_var,
    )
    return md.MuData({"rna": adata_rna, "pert": adata_pert})


def _make_mudata_with_control() -> md.MuData:
    cell_names = ["c0", "c1", "c2", "c3"]
    counts = np.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
        ],
        dtype=np.int32,
    )
    obs = pd.DataFrame(
        {
            "label": ["ctrl", "pertA", "pertA+pertB", "ctrl"],
        },
        index=cell_names,
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    adata_rna = ad.AnnData(X=counts, obs=obs, var=var)

    pert_matrix = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
        ],
        dtype=np.int8,
    )
    pert_var = pd.DataFrame(index=["ctrl", "pertA", "pertB"])
    adata_pert = ad.AnnData(
        X=pert_matrix,
        obs=pd.DataFrame(index=cell_names),
        var=pert_var,
    )
    return md.MuData({"rna": adata_rna, "pert": adata_pert})


def _make_mudata_with_control_element_columns() -> md.MuData:
    cell_names = ["c0", "c1", "c2", "c3"]
    counts = np.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
        ],
        dtype=np.int32,
    )
    adata_rna = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )

    pert_matrix = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.int8,
    )
    guide_names = ["GGGGGGGGGGGGGGGGGGGG", "AAAAAAAAAAAAAAAAAAAA", "CCCCCCCCCCCCCCCCCCCC"]
    pert_var = pd.DataFrame(index=guide_names)
    adata_pert = ad.AnnData(
        X=pert_matrix,
        obs=pd.DataFrame(index=cell_names),
        var=pert_var,
    )
    adata_pert.varm["guide_intended_target_pairs"] = pd.DataFrame(
        np.eye(3, dtype=np.int8),
        index=guide_names,
        columns=["random_1", "scrambled_1", "GENEA"],
    )
    return md.MuData({"rna": adata_rna, "pert": adata_pert})


def _make_mudata_with_control_element_uns_names() -> md.MuData:
    cell_names = ["c0", "c1", "c2", "c3"]
    counts = np.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
        ],
        dtype=np.int32,
    )
    adata_rna = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )

    pert_matrix = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.int8,
    )
    guide_names = ["GGGGGGGGGGGGGGGGGGGG", "AAAAAAAAAAAAAAAAAAAA", "CCCCCCCCCCCCCCCCCCCC"]
    pert_var = pd.DataFrame(index=guide_names)
    adata_pert = ad.AnnData(
        X=pert_matrix,
        obs=pd.DataFrame(index=cell_names),
        var=pert_var,
    )
    adata_pert.varm["guide_intended_target_pairs"] = np.eye(3, dtype=np.int8)
    adata_pert.uns["intended_targets"] = np.array(["random_1", "non-targeting_2", "GENEA"], dtype=object)
    return md.MuData({"rna": adata_rna, "pert": adata_pert})


def test_negbin_model_high_moi_runs_one_step() -> None:
    counts = jnp.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 0, 3],
            [1, 0, 0],
        ],
        dtype=jnp.int32,
    )
    pert_id = jnp.array(
        [
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0],
        ],
        dtype=jnp.int32,
    )
    svi = SVI(
        NegBinModel,
        AutoNormal(NegBinModel),
        numpyro.optim.Adam(step_size=0.01),
        Trace_ELBO(),
    )
    init_key = jax.random.PRNGKey(0)
    state = svi.init(
        init_key,
        counts,
        pert_id,
        num_cells=counts.shape[0],
        num_genes=counts.shape[1],
    )
    svi.update(
        state,
        counts,
        pert_id,
        num_cells=counts.shape[0],
        num_genes=counts.shape[1],
    )


def test_load_controls_high_moi_defaults_to_all_rows() -> None:
    data = _make_mudata()
    controls = load_controls(
        data,
        perturbation_key=None,
        control_selector=None,
        modality_key="rna",
        perturbation_modality_key="pert",
    )
    assert controls.counts.shape == (4, 3)
    assert controls.pert_id.shape == (4, 3)
    assert controls.pert_names == ["pertA", "pertB", "pertC"]


def test_load_analysis_cells_high_moi_subset_filters_cells() -> None:
    data = _make_mudata()
    analysis_data = load_analysis_cells(
        data,
        perturbation_key=None,
        modality_key="rna",
        perturbation_modality_key="pert",
        selected_perturbations=["pertA", "pertB"],
    )
    assert analysis_data.pert_names == ["pertA", "pertB"]
    assert analysis_data.counts.shape == (2, 3)
    assert analysis_data.pert_id.shape == (2, 2)
    assert np.array_equal(
        np.asarray(analysis_data.pert_id),
        np.array(
            [
                [1, 0],
                [1, 1],
            ],
            dtype=np.int8,
        ),
    )
    assert np.array_equal(
        np.asarray(analysis_data.counts),
        np.array(
            [
                [2, 0, 1],
                [0, 0, 3],
            ],
            dtype=np.int32,
        ),
    )


def test_load_controls_high_moi_uses_control_substring_from_var_names() -> None:
    data = _make_mudata_with_control()
    controls = load_controls(
        data,
        perturbation_key=None,
        control_selector="ctrl",
        modality_key="rna",
        perturbation_modality_key="pert",
    )
    assert controls.counts.shape == (2, 3)
    assert controls.pert_id.shape == (2, 1)
    assert controls.pert_names == ["ctrl"]
    assert np.array_equal(
        np.asarray(controls.pert_id),
        np.array(
            [
                [1],
                [1],
            ],
            dtype=np.int8,
        ),
    )


def test_load_controls_high_moi_infers_control_guides_from_element_columns() -> None:
    data = _make_mudata_with_control_element_columns()
    controls = load_controls(
        data,
        perturbation_key=None,
        control_selector=None,
        modality_key="rna",
        perturbation_modality_key="pert",
        perturbation_element_varm_key="guide_intended_target_pairs",
        infer_control_guides=True,
    )
    assert controls.counts.shape == (3, 3)
    assert controls.pert_id.shape == (3, 2)
    assert controls.pert_names == ["GGGGGGGGGGGGGGGGGGGG", "AAAAAAAAAAAAAAAAAAAA"]
    assert np.array_equal(
        np.asarray(controls.pert_id),
        np.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
            ],
            dtype=np.int8,
        ),
    )


def test_load_controls_high_moi_infers_control_guides_from_uns_element_names() -> None:
    data = _make_mudata_with_control_element_uns_names()
    controls = load_controls(
        data,
        perturbation_key=None,
        control_selector=None,
        modality_key="rna",
        perturbation_modality_key="pert",
        perturbation_element_varm_key="guide_intended_target_pairs",
        perturbation_element_names_uns_key="intended_targets",
        infer_control_guides=True,
    )
    assert controls.counts.shape == (3, 3)
    assert controls.pert_id.shape == (3, 2)
    assert controls.pert_names == ["GGGGGGGGGGGGGGGGGGGG", "AAAAAAAAAAAAAAAAAAAA"]
    assert np.array_equal(
        np.asarray(controls.pert_id),
        np.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
            ],
            dtype=np.int8,
        ),
    )


def test_load_controls_high_moi_matches_control_substring_against_element_names() -> None:
    data = _make_mudata_with_control_element_uns_names()
    controls = load_controls(
        data,
        perturbation_key=None,
        control_selector="non-targeting",
        modality_key="rna",
        perturbation_modality_key="pert",
        perturbation_element_varm_key="guide_intended_target_pairs",
        perturbation_element_names_uns_key="intended_targets",
    )
    assert controls.counts.shape == (2, 3)
    assert controls.pert_id.shape == (2, 1)
    assert controls.pert_names == ["AAAAAAAAAAAAAAAAAAAA"]
    assert np.array_equal(
        np.asarray(controls.pert_id),
        np.array(
            [
                [1],
                [1],
            ],
            dtype=np.int8,
        ),
    )


def test_load_controls_high_moi_raises_when_control_guides_cannot_be_inferred() -> None:
    data = _make_mudata()
    with pytest.raises(ValueError, match="requires identifiable control guides"):
        load_controls(
            data,
            perturbation_key=None,
            control_selector=None,
            modality_key="rna",
            perturbation_modality_key="pert",
            infer_control_guides=True,
        )


def test_fit_control_guide_random_effects_requires_multiple_control_guides() -> None:
    data = _make_mudata_with_control()
    controls = load_controls(
        data,
        perturbation_key=None,
        control_selector="ctrl",
        modality_key="rna",
        perturbation_modality_key="pert",
    )

    with pytest.raises(ValueError, match="at least two guide/control-guide labels"):
        fit_control(
            controls,
            num_steps=1,
            model_name="negbin",
            guide_random_effects=True,
        )


def test_load_controls_high_moi_raises_when_control_substring_not_in_grna_names() -> None:
    data = _make_mudata_with_control()
    with pytest.raises(ValueError, match="did not match any perturbation names"):
        load_controls(
            data,
            perturbation_key=None,
            control_selector="nonexistent_control",
            modality_key="rna",
            perturbation_modality_key="pert",
        )


def test_load_analysis_cells_high_moi_can_group_by_element_varm_dataframe() -> None:
    data = _make_mudata()
    data["pert"].varm["element_targeted"] = pd.DataFrame(
        np.array(
            [
                [1, 0],
                [0, 1],
                [0, 1],
            ],
            dtype=np.int8,
        ),
        index=data["pert"].var_names,
        columns=["elem1", "elem2"],
    )

    analysis_data = load_analysis_cells(
        data,
        perturbation_key=None,
        modality_key="rna",
        perturbation_modality_key="pert",
        perturbation_element_varm_key="element_targeted",
    )

    assert analysis_data.pert_names == ["elem1", "elem2"]
    assert analysis_data.pert_id.shape == (4, 2)
    assert np.array_equal(
        np.asarray(analysis_data.pert_id),
        np.array(
            [
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 0],
            ],
            dtype=np.int8,
        ),
    )


def test_load_analysis_cells_high_moi_grouped_varm_requires_names_or_uns_key() -> None:
    data = _make_mudata()
    data["pert"].varm["element_targeted"] = np.array(
        [
            [1, 0],
            [0, 1],
            [0, 1],
        ],
        dtype=np.int8,
    )
    with pytest.raises(ValueError, match="--perturbation-element-names-uns-key"):
        load_analysis_cells(
            data,
            perturbation_key=None,
            modality_key="rna",
            perturbation_modality_key="pert",
            perturbation_element_varm_key="element_targeted",
        )


def test_load_analysis_cells_high_moi_grouped_varm_uses_uns_names_for_subset() -> None:
    data = _make_mudata()
    data["pert"].varm["element_targeted"] = np.array(
        [
            [1, 0],
            [0, 1],
            [0, 1],
        ],
        dtype=np.int8,
    )
    data["pert"].uns["element_names"] = ["elem1", "elem2"]

    analysis_data = load_analysis_cells(
        data,
        perturbation_key=None,
        modality_key="rna",
        perturbation_modality_key="pert",
        perturbation_element_varm_key="element_targeted",
        perturbation_element_names_uns_key="element_names",
        selected_perturbations=["elem2"],
    )

    assert analysis_data.pert_names == ["elem2"]
    assert analysis_data.pert_id.shape == (1, 1)
    assert np.array_equal(np.asarray(analysis_data.pert_id), np.array([[1]], dtype=np.int8))
    assert np.array_equal(
        np.asarray(analysis_data.counts),
        np.array([[0, 0, 3]], dtype=np.int32),
    )


def test_load_analysis_cells_high_moi_grouped_varm_accepts_sparse_mapping() -> None:
    data = _make_mudata()
    data["pert"].varm["element_targeted"] = sp.csr_matrix(
        np.array(
            [
                [1, 0],
                [0, 1],
                [0, 1],
            ],
            dtype=np.int8,
        )
    )
    data["pert"].uns["element_names"] = ["elem1", "elem2"]

    analysis_data = load_analysis_cells(
        data,
        perturbation_key=None,
        modality_key="rna",
        perturbation_modality_key="pert",
        perturbation_element_varm_key="element_targeted",
        perturbation_element_names_uns_key="element_names",
    )

    assert analysis_data.pert_names == ["elem1", "elem2"]
    assert analysis_data.pert_id.shape == (4, 2)
