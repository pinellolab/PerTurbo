from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import anndata as ad
import jax.numpy as jnp
import mudata as md
import numpy as np
import numpyro
import pandas as pd
import scipy.sparse as sp

import perturbo.core as core
from perturbo.core import ControlFit, SVIConfig, fit_perturbation_effects, load_analysis_cells, load_controls
from perturbo.model import GuideSharedNegativeBinomialModel, NegBinModel
from perturbo.sparse_design import IndexedDesignMatrix, indexed_design_to_dense


def _control_fit(num_genes: int) -> ControlFit:
    return ControlFit(
        beta_0=jnp.zeros(num_genes, dtype=jnp.float32),
        theta=jnp.full(num_genes, 10.0, dtype=jnp.float32),
        noise_scale=jnp.ones(num_genes, dtype=jnp.float32),
        factor_loadings=None,
        factor_scores=None,
        factor_center=None,
        pca_loadings=None,
        size_factors=jnp.zeros((2, 1), dtype=jnp.float32),
        losses=jnp.array([]),
        svi_result=None,
    )


def _high_moi_data() -> tuple[md.MuData, np.ndarray, np.ndarray]:
    counts = np.array(
        [[2, 1, 9], [4, 0, 1], [3, 2, 5], [2, 1, 4], [8, 1, 2], [7, 0, 3], [9, 1, 0], [6, 2, 1]],
        dtype=np.int32,
    )
    guides = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1], [1, 0], [0, 1], [1, 1], [0, 0]],
        dtype=np.int8,
    )
    obs = pd.DataFrame(index=[f"c{i}" for i in range(counts.shape[0])])
    rna = ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=["g0", "g1", "g2"]))
    perturbations = ad.AnnData(
        X=sp.csr_matrix(guides),
        obs=obs.copy(),
        var=pd.DataFrame(index=["guide_A", "guide_B"]),
    )
    perturbations.varm["targets"] = pd.DataFrame(
        np.eye(2, dtype=np.float32),
        index=perturbations.var_names,
        columns=["target_A", "target_B"],
    )
    return md.MuData({"rna": rna, "pert": perturbations}), counts, guides


def test_gene_chunk_loads_only_selected_counts_and_uses_full_panel_offsets(monkeypatch) -> None:
    data, counts, guides = _high_moi_data()
    dense_shapes: list[tuple[int, ...]] = []
    original_to_dense = core._to_dense

    def recording_to_dense(matrix):
        dense_shapes.append(tuple(matrix.shape))
        return original_to_dense(matrix)

    monkeypatch.setattr(core, "_to_dense", recording_to_dense)
    loaded = load_analysis_cells(
        data,
        modality_key="rna",
        perturbation_modality_key="pert",
        perturbation_element_varm_key="targets",
        retain_guide_structure=True,
        selected_gene_indices=slice(0, 2),
        indexed_perturbation_design=True,
    )

    np.testing.assert_array_equal(np.asarray(loaded.counts), counts[:, :2])
    assert loaded.gene_names == ["g0", "g1"]
    assert (counts.shape[0], counts.shape[1]) not in dense_shapes
    expected_offsets = np.log1p(counts.sum(axis=1)) - np.mean(np.log1p(counts.sum(axis=1)))
    np.testing.assert_allclose(np.asarray(loaded.size_factors).reshape(-1), expected_offsets, rtol=1e-6)
    assert isinstance(loaded.pert_id, IndexedDesignMatrix)
    assert isinstance(loaded.guide_matrix, IndexedDesignMatrix)
    np.testing.assert_array_equal(np.asarray(indexed_design_to_dense(loaded.pert_id)), guides)


def test_gene_chunk_design_cache_reuses_sparse_assignment_and_covariates(monkeypatch) -> None:
    data, counts, _ = _high_moi_data()
    data.mod["rna"].obs["depth"] = np.linspace(0.0, 1.0, counts.shape[0])
    data.mod["pert"].varm["targets"] = sp.eye(2, dtype=np.int8, format="csr")
    data.mod["pert"].uns["target_names"] = ["target_A", "target_B"]
    grouped_calls = 0
    original_group = core._group_perturbation_matrix_by_element

    def counting_group(*args, **kwargs):
        nonlocal grouped_calls
        grouped_calls += 1
        return original_group(*args, **kwargs)

    monkeypatch.setattr(core, "_group_perturbation_matrix_by_element", counting_group)
    library_sizes = counts.sum(axis=1)
    first, cache = load_analysis_cells(
        data,
        modality_key="rna",
        perturbation_modality_key="pert",
        perturbation_element_varm_key="targets",
        perturbation_element_names_uns_key="target_names",
        retain_guide_structure=True,
        continuous_covariates=["depth"],
        selected_gene_indices=slice(0, 1),
        full_panel_library_sizes=library_sizes,
        indexed_perturbation_design=True,
        _return_design_cache=True,
    )
    second = load_analysis_cells(
        data,
        modality_key="rna",
        perturbation_modality_key="pert",
        perturbation_element_varm_key="targets",
        perturbation_element_names_uns_key="target_names",
        retain_guide_structure=True,
        continuous_covariates=["depth"],
        selected_gene_indices=slice(1, 3),
        full_panel_library_sizes=library_sizes,
        indexed_perturbation_design=True,
        _design_cache=cache,
    )

    assert grouped_calls == 1
    assert second.pert_id is first.pert_id
    assert second.covariates is first.covariates
    assert second.guide_matrix is first.guide_matrix
    assert sp.issparse(first.guide_to_element)
    assert second.guide_to_element is first.guide_to_element
    np.testing.assert_array_equal(np.asarray(first.counts), counts[:, :1])
    np.testing.assert_array_equal(np.asarray(second.counts), counts[:, 1:3])
    np.testing.assert_allclose(np.asarray(first.size_factors), np.asarray(second.size_factors))


def test_gene_chunk_design_cache_rejects_changed_selection_or_options() -> None:
    data, counts, _ = _high_moi_data()
    _, cache = load_analysis_cells(
        data,
        modality_key="rna",
        perturbation_modality_key="pert",
        perturbation_element_varm_key="targets",
        selected_gene_indices=slice(0, 1),
        full_panel_library_sizes=counts.sum(axis=1),
        indexed_perturbation_design=True,
        _return_design_cache=True,
    )
    with np.testing.assert_raises_regex(ValueError, "options changed"):
        load_analysis_cells(
            data,
            modality_key="rna",
            perturbation_modality_key="pert",
            perturbation_element_varm_key="targets",
            selected_gene_indices=slice(1, 2),
            full_panel_library_sizes=counts.sum(axis=1),
            indexed_perturbation_design=False,
            _design_cache=cache,
        )
    with np.testing.assert_raises_regex(ValueError, "cell selection/order changed"):
        load_analysis_cells(
            data,
            modality_key="rna",
            perturbation_modality_key="pert",
            perturbation_element_varm_key="targets",
            selected_gene_indices=slice(1, 2),
            full_panel_library_sizes=counts.sum(axis=1),
            indexed_perturbation_design=True,
            cell_keep_mask=np.arange(counts.shape[0]) != 0,
            _design_cache=cache,
        )


def test_control_loader_skips_unused_element_mapping(monkeypatch) -> None:
    data, counts, _ = _high_moi_data()

    def fail_mapping(*args, **kwargs):
        raise AssertionError("unused guide-to-element mapping should not be loaded")

    monkeypatch.setattr(core, "_load_perturbation_element_mapping", fail_mapping)
    controls = load_controls(
        data,
        modality_key="rna",
        perturbation_modality_key="pert",
        perturbation_element_varm_key="targets",
        control_selector=None,
        infer_control_guides=False,
        max_control_cells=None,
    )
    np.testing.assert_array_equal(np.asarray(controls.counts), counts)


def test_backed_gene_chunk_does_not_materialize_raw_or_layers(tmp_path, monkeypatch) -> None:
    cell_names = ["c0", "c1", "c2", "c3"]
    counts = np.arange(24, dtype=np.int32).reshape(4, 6)
    obs = pd.DataFrame(index=cell_names)
    rna = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(6)]))
    rna.raw = rna.copy()
    rna.layers["unused_full_panel"] = sp.csr_matrix(counts + 1)
    rna_path = tmp_path / "rna-with-raw.h5ad"
    rna.write_h5ad(rna_path)
    backed_rna = ad.read_h5ad(rna_path, backed="r")

    guide_assignments = sp.csr_matrix(
        np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=np.int8)
    )
    perturbations = ad.AnnData(
        X=guide_assignments,
        obs=obs.copy(),
        var=pd.DataFrame(index=["ctrl", "guide_A", "guide_B"]),
    )
    perturbations.varm["targets"] = sp.eye(3, dtype=np.int8, format="csr")
    perturbations.uns["target_names"] = ["ctrl", "target_A", "target_B"]
    data = SimpleNamespace(mod={"rna": backed_rna, "pert": perturbations})

    def fail_to_memory(self, *args, **kwargs):
        raise AssertionError("loaders must not materialize the whole backed AnnData object")

    monkeypatch.setattr(ad.AnnData, "to_memory", fail_to_memory)
    try:
        loaded = load_analysis_cells(
            data,
            modality_key="rna",
            perturbation_modality_key="pert",
            perturbation_element_varm_key="targets",
            perturbation_element_names_uns_key="target_names",
            selected_gene_indices=slice(2, 4),
            indexed_perturbation_design=True,
        )
        controls = load_controls(
            data,
            modality_key="rna",
            perturbation_modality_key="pert",
            perturbation_element_varm_key="targets",
            perturbation_element_names_uns_key="target_names",
            control_selector="ctrl",
            max_control_cells=None,
        )
    finally:
        backed_rna.file.close()

    np.testing.assert_array_equal(np.asarray(loaded.counts), counts[:, 2:4])
    np.testing.assert_array_equal(np.asarray(controls.counts), counts[[0]])


def test_counts_derived_offsets_keep_zero_rows_and_control_center() -> None:
    center = float(np.log1p(9.0))
    offsets, loaded_center = core._load_observed_size_factors(
        pd.DataFrame(index=["zero", "positive"]),
        np.array([[0], [3]], dtype=np.int32),
        library_size_center_log_mean=center,
        counts_library_sizes=np.array([0, 9], dtype=np.float64),
    )

    assert loaded_center == center
    np.testing.assert_allclose(np.asarray(offsets).reshape(-1), [-center, 0.0], atol=1e-6)


def test_control_loader_densifies_sparse_guides_only_after_filter_and_cap(monkeypatch) -> None:
    num_cells, num_guides = 40, 200
    obs = pd.DataFrame(index=[f"c{i}" for i in range(num_cells)])
    rna = ad.AnnData(
        X=np.ones((num_cells, 2), dtype=np.int32),
        obs=obs,
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    guide_columns = np.concatenate([np.zeros(20, dtype=np.int64), np.arange(1, 21, dtype=np.int64)])
    assignments = sp.csr_matrix(
        (np.ones(num_cells), (np.arange(num_cells), guide_columns)),
        shape=(num_cells, num_guides),
    )
    perturbations = ad.AnnData(
        X=assignments,
        obs=obs.copy(),
        var=pd.DataFrame(index=["ctrl"] + [f"guide_{i}" for i in range(1, num_guides)]),
    )
    data = md.MuData({"rna": rna, "pert": perturbations})
    dense_shapes: list[tuple[int, ...]] = []
    original_to_dense = core._to_dense

    def recording_to_dense(matrix):
        dense_shapes.append(tuple(matrix.shape))
        return original_to_dense(matrix)

    monkeypatch.setattr(core, "_to_dense", recording_to_dense)
    controls = load_controls(
        data,
        modality_key="rna",
        perturbation_modality_key="pert",
        control_selector="ctrl",
        max_control_cells=5,
    )

    assert controls.counts.shape[0] == 5
    assert (num_cells, num_guides) not in dense_shapes
    assert (5, 1) in dense_shapes


def test_low_moi_target_subset_stays_sparse_until_rows_and_columns_are_selected(monkeypatch) -> None:
    num_cells, num_guides = 40, 200
    obs = pd.DataFrame(index=[f"c{i}" for i in range(num_cells)])
    rna = ad.AnnData(
        X=np.ones((num_cells, 2), dtype=np.int32),
        obs=obs,
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    columns = np.arange(num_cells, dtype=np.int64)
    perturbations = ad.AnnData(
        X=sp.csr_matrix(
            (np.ones(num_cells), (np.arange(num_cells), columns)),
            shape=(num_cells, num_guides),
        ),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"guide_{i}" for i in range(num_guides)]),
    )
    perturbations.varm["targets"] = sp.eye(num_guides, dtype=np.int8, format="csr")
    perturbations.uns["target_names"] = [f"target_{i}" for i in range(num_guides)]
    data = md.MuData({"rna": rna, "pert": perturbations})
    dense_shapes: list[tuple[int, ...]] = []
    original_to_dense = core._to_dense

    def recording_to_dense(matrix):
        dense_shapes.append(tuple(matrix.shape))
        return original_to_dense(matrix)

    monkeypatch.setattr(core, "_to_dense", recording_to_dense)
    loaded = load_analysis_cells(
        data,
        modality_key="rna",
        perturbation_modality_key="pert",
        perturbation_element_varm_key="targets",
        perturbation_element_names_uns_key="target_names",
        selected_perturbations=["target_0", "target_1"],
        retain_guide_structure=True,
    )

    assert loaded.counts.shape == (2, 2)
    assert loaded.pert_id.shape == (2, 2)
    assert loaded.guide_matrix.shape == (2, 2)
    assert (num_cells, num_guides) not in dense_shapes


def test_sparse_grouping_uses_positive_presence_without_int8_overflow() -> None:
    num_guides = 130
    assignments = sp.csr_matrix(np.full((1, num_guides), 0.5, dtype=np.float32))
    mapping = sp.csr_matrix(np.ones((num_guides, 1), dtype=np.int8))

    grouped = core._group_perturbation_matrix_by_element(assignments, mapping, preserve_sparse=True)

    np.testing.assert_array_equal(grouped.toarray(), [[1]])


def test_indexed_high_moi_design_runs_minibatched_svi() -> None:
    data, _, _ = _high_moi_data()
    loaded = load_analysis_cells(
        data,
        modality_key="rna",
        perturbation_modality_key="pert",
        perturbation_element_varm_key="targets",
        retain_guide_structure=True,
        selected_gene_indices=slice(0, 2),
        indexed_perturbation_design=True,
    )

    fit = fit_perturbation_effects(
        loaded,
        _control_fit(2),
        num_steps=1,
        svi_config=SVIConfig(),
        model_name="nb",
        use_observed_size_factors=True,
        minibatch_size=4,
    )

    assert fit.posterior_mean.shape == (2, 2)
    assert fit.losses.shape == (1,)


def test_subset_control_fit_genes_slices_every_gene_axis() -> None:
    control = _control_fit(4)
    control = replace(
        control,
        beta_0=jnp.arange(4, dtype=jnp.float32),
        covariate_coef=jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
    )

    chunk = core.subset_control_fit_genes(control, slice(1, 3))

    np.testing.assert_array_equal(np.asarray(chunk.beta_0), [1, 2])
    np.testing.assert_array_equal(np.asarray(chunk.covariate_coef), [[1, 2], [5, 6]])


def _observation_log_prob(counts, design, beta, beta_0, theta, offsets) -> float:
    conditioned = numpyro.handlers.condition(
        NegBinModel,
        data={"beta": beta, "beta_0": beta_0, "theta": theta},
    )
    trace = numpyro.handlers.trace(numpyro.handlers.seed(conditioned, rng_seed=4)).get_trace(
        counts,
        design,
        size_factors=offsets,
        num_cells=counts.shape[0],
        num_genes=counts.shape[1],
        num_perts=beta.shape[0],
    )
    return float(jnp.sum(trace["obs"]["fn"].log_prob(counts)))


def test_indexed_joint_likelihood_equals_dense_and_gene_blocks() -> None:
    counts = jnp.array([[3, 1, 2], [5, 0, 4], [2, 6, 1]], dtype=jnp.int32)
    dense = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float32)
    indexed = core.indexed_design_from_matrix(dense)
    beta = jnp.array([[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]], dtype=jnp.float32)
    beta_0 = jnp.array([1.0, 0.7, 1.2], dtype=jnp.float32)
    theta = jnp.array([10.0, 8.0, 12.0], dtype=jnp.float32)
    offsets = jnp.array([[0.0], [0.1], [-0.2]], dtype=jnp.float32)

    joint_dense = _observation_log_prob(counts, jnp.asarray(dense), beta, beta_0, theta, offsets)
    joint_indexed = _observation_log_prob(counts, indexed, beta, beta_0, theta, offsets)
    split = sum(
        _observation_log_prob(
            counts[:, gene_slice],
            indexed,
            beta[:, gene_slice],
            beta_0[gene_slice],
            theta[gene_slice],
            offsets,
        )
        for gene_slice in (slice(0, 2), slice(2, 3))
    )

    np.testing.assert_allclose(joint_indexed, joint_dense, rtol=1e-6)
    np.testing.assert_allclose(split, joint_dense, rtol=1e-6)


def test_shared_mean_counts_an_element_once_when_two_guides_target_it() -> None:
    beta = jnp.array([[np.log(2.0)]], dtype=jnp.float32)
    conditioned = numpyro.handlers.condition(
        GuideSharedNegativeBinomialModel,
        data={
            "beta": beta,
            "beta_0": jnp.array([np.log(10.0)], dtype=jnp.float32),
            "theta": jnp.array([50.0], dtype=jnp.float32),
            "guide_dispersion_excess_inverse": jnp.array([[0.1], [0.1]], dtype=jnp.float32),
        },
    )
    trace = numpyro.handlers.trace(numpyro.handlers.seed(conditioned, rng_seed=9)).get_trace(
        jnp.array([[0]], dtype=jnp.int32),
        jnp.array([[1]], dtype=jnp.bool_),
        size_factors=jnp.zeros((1, 1), dtype=jnp.float32),
        guide_matrix=jnp.array([[1.0, 1.0]], dtype=jnp.float32),
        guide_to_element=jnp.array([[1.0], [1.0]], dtype=jnp.float32),
        num_cells=1,
        num_genes=1,
        num_perts=1,
        num_guides=2,
        fit_perturbation_dispersion=True,
    )

    np.testing.assert_allclose(np.asarray(trace["obs"]["fn"].mean), [[20.0]], rtol=1e-5)
