import json

import anndata as ad
import h5py
import mudata as md
import numpy as np
import pandas as pd
import pytest

from perturbo.fitted_export import export
from perturbo.io import control_fit_arrays, save_light_fit_bundle


def test_writer_records_exact_baseline_order_and_elides_identity(tmp_path):
    for rows in ([2, 0], [0, 1, 2]):
        save_light_fit_bundle(
            tmp_path, metadata={}, control_arrays={}, beta_arrays={},
            baseline_cell_indices=np.array(rows), source_cell_count=3,
        )
        metadata = json.loads((tmp_path / "metadata.json").read_text())
        assert metadata["n_baseline_cells"] == len(rows)
        path = tmp_path / "baseline_cell_indices.npy"
        if len(rows) == 2:
            assert metadata["baseline_cell_mapping"] == "verified_indices"
            np.testing.assert_array_equal(np.load(path), rows)
        else:
            assert metadata["baseline_cell_mapping"] == "all_cells"
            assert not path.exists()


def bundle(path, strategy):
    path.mkdir()
    genes, guides, elems = ["g2", "g1"], ["a2", "b", "a1"], ["A", "B"]
    cells = ["z", "x", "y"]
    rna = ad.AnnData(
        np.array([[2, 1], [4, 3], [6, 5]], dtype=np.uint16),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=genes),
    )
    grna = ad.AnnData(np.eye(3, dtype=np.uint8), obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=guides))
    grna.varm["mapping"] = np.array([[1, 0], [0, 1], [1, 0]])
    grna.uns["elements"] = elems
    md.MuData({"rna": rna, "grna": grna}).write_h5mu(path / "mdata.h5mu")
    meta = dict(
        setup={
            "rna_modality": "rna",
            "perturbation_modality": "grna",
            "guide_by_element_key": "mapping",
            "guide_element_uns_key": "elements",
        },
        likelihood="negbin",
        effect_prior_dist="normal",
        gene_names=genes,
        guide_names=guides,
        perturbation_names=elems,
        guide_effect_strategy=strategy,
        cli_args={"none": None, "seed": 2},
    )
    (path / "metadata.json").write_text(json.dumps(meta))
    df = pd.DataFrame(
        {
            "element": ["A", "B", "A", "B"],
            "gene": ["g1", "g2", "g2", "g1"],
            "posterior_mean": np.array([1, 2, 3, 4], dtype="f4"),
            "posterior_scale": np.array([0.1, 0.2, 0.3, 0.4], dtype="f4"),
            "method": "perturbo",
            "crt_saddlepoint_p_value": [1e-200, 0.12, 0.31, np.nan],
            "crt_saddlepoint_q_value": [3e-200, 0.18, 0.31, np.nan],
            "crt_p_value": [0.4] * 4,
        }
    )
    df.to_parquet(path / "element_effects.parquet")
    cis = df.iloc[[2, 0]].copy()
    cis["crt_saddlepoint_q_value"] = [0.31, 2e-200]
    cis.to_parquet(path / "element_effects_requested_pairs.parquet")
    np.savez(
        path / "control_fit.npz",
        beta_0=np.ones(2, dtype="f4"),
        theta=np.ones(2, dtype="f4"),
        noise_scale=np.ones(2, dtype="f4"),
        size_factors=np.array([4.0, 5.0]),
    )
    np.save(path / "baseline_cell_indices.npy", [2, 0])
    np.save(path / "cell_keep_indices.npy", [0, 2])
    np.save(path / "size_factors.npy", [5.0, 4.0])
    cov = pd.DataFrame({"batch": [0.0, 1.0]}, index=["z", "y"])
    cov.to_parquet(path / "cell_covariates.parquet")
    if strategy == "relative":
        values = np.arange(6, dtype="f4").reshape(3, 2) / 10
        g = pd.DataFrame(
            {
                "guide": np.repeat(guides, 2),
                "gene": np.tile(genes, 3),
                "element": np.repeat(["A", "B", "A"], 2),
                "posterior_mean": values.ravel(),
                "method": "perturbo",
            }
        )
        g.iloc[::-1].to_parquet(path / "guide_efficiency.parquet", index=False)
        np.savez(
            path / "guide_posteriors.npz",
            guide_relative_efficiency_mean=values,
            guide_relative_efficiency_scale=values + 1,
            guide_effect_mean=values + 2,
            guide_effect_scale=values + 3,
        )
    return path


@pytest.mark.parametrize("strategy", ["relative", "shared"])
def test_native_export(tmp_path, strategy):
    root = bundle(tmp_path / "bundle", strategy)
    dest = export(root, tmp_path / "new", primary="saddlepoint", cis=True)
    m = md.read_h5mu(dest / "fitted.h5mu")
    pt = m.uns["_perturbo"]
    assert pt["metadata"]["method"] == "perturbo"
    np.testing.assert_array_equal(pt["baseline_cell_indices"], [2, 0])
    np.testing.assert_array_equal(pt["cell_keep_indices"], [0, 2])
    np.testing.assert_array_equal(m["rna"].X, [[2, 1], [4, 3], [6, 5]])
    np.testing.assert_array_equal(m["grna"].X, np.eye(3))
    fx = pt["effects"]
    np.testing.assert_equal(fx.q_cis.to_numpy(), [2e-200, np.nan, 0.31, np.nan])
    assert fx.q_global.dtype == np.float64
    assert pt["crt_metadata"]["result_fields"]["q_global"]["finite_p"] == 3
    if strategy == "relative":
        assert np.isclose(pt["efficiency"].iloc[0]["guide_effect_mean"], 2.5)
    else:
        assert "efficiency" not in pt
    assert (dest / "guide_efficiency.parquet").exists() == (strategy == "relative")


@pytest.mark.parametrize("prefix", ["", "crt_"])
def test_detection_annotations_round_trip(tmp_path, prefix):
    root = bundle(tmp_path / "bundle", "shared")
    path = root / "element_effects.parquet"
    table = pd.read_parquet(path)
    annotations = pd.DataFrame({
        "observed_nonzero": [0, 3, 5, 0],
        "expected_nonzero": [1.5, 4.0, 8.0, 0.0],
        "low_information": [True, True, False, False],
    })
    pd.concat([table, annotations.add_prefix(prefix)], axis=1).to_parquet(path, index=False)
    dest = export(root, tmp_path / "export", primary="saddlepoint")
    effects = md.read_h5mu(dest / "fitted.h5mu").uns["_perturbo"]["effects"]
    pd.testing.assert_frame_equal(effects[annotations.columns].reset_index(drop=True), annotations)
    assert not set(annotations.add_prefix("crt_").columns).intersection(effects.columns)


def test_unverified_legacy_baseline_and_undeclared_family(tmp_path):
    root = bundle(tmp_path / "old", "shared")
    (root / "baseline_cell_indices.npy").unlink()
    with pytest.raises(ValueError, match="primary"):
        export(root, tmp_path / "export")
    dest = export(root, tmp_path / "export", primary="saddlepoint")
    pt = md.read_h5mu(dest / "fitted.h5mu").uns["_perturbo"]
    assert "q_cis" not in pt["effects"]
    assert pt["metadata"]["baseline_cell_mapping"] == "unknown"


def test_mismatch_rejects_export(tmp_path):
    root = bundle(tmp_path / "old", "relative")
    g = pd.read_parquet(root / "guide_efficiency.parquet")
    g.loc[0, "posterior_mean"] = 100
    g.to_parquet(root / "guide_efficiency.parquet")
    with pytest.raises(ValueError, match="Values differ"):
        export(root, tmp_path / "new", primary="saddlepoint")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("strategy", ["shared", "relative"])
def test_restore_model(tmp_path, strategy):
    from perturbo.inference import PerTurboModel

    root = bundle(tmp_path / "old", strategy)
    name = export(root, tmp_path / "new", primary="saddlepoint") / "fitted.h5mu"
    restored = PerTurboModel.load(name)
    assert restored.element_names == ["A", "B"] and restored.gene_names == ["g2", "g1"]
    np.testing.assert_equal(restored.beta_fit.posterior_mean, [[3, 1], [2, 4]])
    np.testing.assert_array_equal(restored.control_fit.size_factors, [4, 5])
    np.testing.assert_array_equal(restored.adata["rna"].obs["_perturbo_saved_size_factor"], [5, 4])
    assert list(restored.adata["rna"].obs_names) == ["z", "y"]
    if strategy == "relative":
        with np.load(root / "guide_posteriors.npz") as guide:
            for field in guide.files:
                np.testing.assert_array_equal(getattr(restored.beta_fit, field), guide[field])
    else:
        np.testing.assert_array_equal(restored.guide_efficacy, np.ones(3))


@pytest.mark.parametrize("missing", [False, True])
def test_restore_nullable_string_ids_checks_mask(tmp_path, missing):
    from anndata.io import write_elem
    from perturbo.inference import PerTurboModel

    root = bundle(tmp_path / "old", "shared")
    path = export(root, tmp_path / "new", primary="saddlepoint") / "fitted.h5mu"
    with h5py.File(path, "r+") as handle, ad.settings.override(allow_write_nullable_strings=True):
        table = handle["uns/_perturbo/effects"]
        del table["element"]
        values = ["A", pd.NA if missing else "B", "A", "B"]
        write_elem(table, "element", pd.array(values, dtype="string"))
        var = handle["mod/grna/var"]
        key = var.attrs["_index"]
        del var[key]
        write_elem(var, key, pd.array(["a2", "b", "a1"], dtype="string"))
    if missing:
        with pytest.raises(ValueError, match="Missing fit identifier"):
            PerTurboModel.load(path)
    else:
        model = PerTurboModel.load(path)
        np.testing.assert_array_equal(model.beta_fit.posterior_mean, [[3, 1], [2, 4]])


def test_low_moi_construct_design_and_categorical_fit_ids(tmp_path):
    from perturbo.inference import PerTurboModel

    root = bundle(tmp_path / "old", "shared")
    native = export(root, tmp_path / "new", primary="saddlepoint") / "fitted.h5mu"
    screen = md.read_h5mu(native)
    pt = screen.uns["_perturbo"]
    pt["metadata"].update(input_mode="low_moi", perturbation_key="gene_transcript", guide_names=["A", "B"])
    del pt["metadata"]["setup"]["guide_by_element_key"]
    del pt["metadata"]["setup"]["guide_element_uns_key"]
    screen["rna"].obs["gene_transcript"] = ["B", "none", "A"]
    for column in ["gene", "element"]:
        pt["effects"][column] = pd.Categorical(pt["effects"][column])
    native = tmp_path / "low_moi.h5mu"
    screen.write_h5mu(native)
    model = PerTurboModel.load(native)
    np.testing.assert_array_equal(model.adata["grna"].X.toarray(), [[0, 1], [1, 0]])
    np.testing.assert_array_equal(model.beta_fit.posterior_mean, [[3, 1], [2, 4]])
    pt["effects"].iloc[0] = pt["effects"].iloc[-1]
    corrupt = tmp_path / "duplicate.h5mu"
    screen.write_h5mu(corrupt)
    with pytest.raises(ValueError, match="Duplicate"):
        PerTurboModel.load(corrupt)


def test_export_preserves_declared_fdr_families(tmp_path):
    root = bundle(tmp_path / "bundle", "shared")
    path = root / "element_effects.parquet"
    fx = pd.read_parquet(path)
    fx["p_value"] = fx["crt_saddlepoint_p_value"]
    fx["q_global"] = fx["crt_saddlepoint_q_value"]
    fx["q_element"] = [2e-200, 0.12, 0.31, np.nan]
    fx["q_cis"] = [1e-200, np.nan, np.nan, np.nan]
    fx.to_parquet(path, index=False)
    fields = {name: {"column": name} for name in ["p_value", "q_global", "q_element", "q_cis"]}
    meta = {"primary_test": "saddlepoint", "result_fields": fields, "cis_family": {"finite_p": 1}}
    (root / "crt_metadata.json").write_text(json.dumps(meta))
    dest = export(root, tmp_path / "new")
    pt = md.read_h5mu(dest / "fitted.h5mu").uns["_perturbo"]
    pd.testing.assert_frame_equal(pt["effects"][list(fields)].reset_index(drop=True), fx[list(fields)], check_exact=True)
    assert pt["crt_metadata"]["cis_family"]["finite_p"] == 1
    with pytest.raises(ValueError, match="Primary test"):
        export(root, tmp_path / "mismatch", primary="resampling")


@pytest.mark.parametrize("strategy", ["shared", "relative"])
def test_rc13_artifacts_after_relocation(tmp_path, strategy):
    from perturbo.fitted_output import publish
    from perturbo.inference import PerTurboModel

    root = bundle(tmp_path / "work", strategy)
    source = root / "mdata.h5mu"
    screen = md.read_h5mu(source)
    screen["grna"].layers["guide_assignment"] = screen["grna"].X.astype(bool)
    screen.write_h5mu(source)
    meta = json.loads((root / "metadata.json").read_text())
    meta["source"] = {"path": str(source)}
    effects = pd.read_parquet(root / "element_effects.parquet")
    effects["p_value"] = effects["crt_saddlepoint_p_value"]
    effects["q_global"] = effects["crt_saddlepoint_q_value"]
    effects.to_parquet(root / "element_effects.parquet")
    with np.load(root / "control_fit.npz") as z:
        controls = dict(z)
    guides = {}
    if strategy == "relative":
        with np.load(root / "guide_posteriors.npz") as z:
            guides = dict(z)
    native, compact = tmp_path / "fitted.h5mu", tmp_path / "compact"
    for fmt, dest in [("h5mu", native), ("bundle", compact)]:
        publish(
            root,
            dest,
            fmt,
            metadata=meta,
            control=controls,
            beta={"losses": np.array([2.0, 1.0])},
            guides=guides,
            offsets=np.array([5.0, 4.0]),
            keep=np.array([0, 2]),
            baseline=np.array([2, 0]),
            n_cells=3,
            obs=screen["rna"].obs,
        )
    moved = source.rename(tmp_path / "relocated.h5mu")
    standalone = PerTurboModel.load(native)
    with pytest.raises(FileNotFoundError, match="source_path"):
        PerTurboModel.load(compact)
    linked = PerTurboModel.load(compact, source_path=moved)
    result = md.read_h5mu(native)
    np.testing.assert_array_equal(result["grna"].layers["guide_assignment"], screen["grna"].layers["guide_assignment"])
    for name, file in [("effects", "element_effects.parquet"), ("efficiency", "guide_efficiency.parquet")]:
        if name == "efficiency" and strategy == "shared":
            assert not (compact / file).exists()
            continue
        pd.testing.assert_frame_equal(
            result.uns["_perturbo"][name].reset_index(drop=True), pd.read_parquet(compact / file), check_exact=True
        )
    for model in (standalone, linked):
        np.testing.assert_array_equal(model.beta_fit.posterior_mean, [[3, 1], [2, 4]])
        for name, value in controls.items():
            np.testing.assert_array_equal(control_fit_arrays(model.control_fit)[name], value)
        for field, values in guides.items():
            np.testing.assert_array_equal(getattr(model.beta_fit, field), values)
    # Re-saving a restored analysis must preserve tests and the original unfiltered assays.
    written = linked.save(tmp_path / "resaved")
    checked = md.read_h5mu(written)
    pd.testing.assert_frame_equal(
        checked.uns["_perturbo"]["effects"], result.uns["_perturbo"]["effects"], check_exact=True
    )
    assert checked["rna"].n_obs == 3
    from_native = standalone.save(tmp_path / "linked_again", output_format="bundle")
    np.testing.assert_array_equal(
        PerTurboModel.load(from_native).beta_fit.posterior_mean, linked.beta_fit.posterior_mean
    )
    with open(moved, "ab") as out:
        out.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        PerTurboModel.load(compact, source_path=moved)


def test_writer_preserves_large_assay_attributes(tmp_path):
    from perturbo.fitted_output import _assay_file

    source = tmp_path / "source.h5mu"
    dest = tmp_path / "copy.h5mu"
    attribute = np.arange(20_000, dtype=np.int64)
    with h5py.File(source, "w", track_order=True) as h:
        group = h.create_group("mod").create_group("rna", track_order=True)
        array = group.create_dataset("X", data=np.arange(3), track_order=True)
        for obj in (h, group, array):
            obj.attrs["large"] = attribute
    _assay_file(source, dest, "rna")
    with h5py.File(dest) as h:
        for obj in (h, h["mod/rna"], h["mod/rna/X"]):
            np.testing.assert_array_equal(obj.attrs["large"], attribute)
        np.testing.assert_array_equal(h["mod/rna/X"][:], np.arange(3))
