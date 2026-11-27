"""Opt-in small-screen pilot: python -m perturbo.fitted_export BUNDLE NEW_DIRECTORY."""

import argparse
import json
import tempfile
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

PAIR = ["element", "gene"]
GUIDE_FIELDS = (
    "guide_relative_efficiency_mean",
    "guide_relative_efficiency_scale",
    "guide_effect_mean",
    "guide_effect_scale",
)
PRIMARY = {"saddlepoint": "crt_saddlepoint_", "resampling": "crt_"}


def _check_keys(frame, keys):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"Missing or duplicate identifiers: {keys}")


def _check_equal(left, right, description):
    if not np.array_equal(left, right, equal_nan=True):
        raise ValueError(f"Values differ: {description}")


def _method(frame, metadata):
    if "method" in frame:
        labels = frame["method"].drop_duplicates()
        if len(labels) != 1 or pd.isna(labels.iloc[0]):
            raise ValueError("The result table needs one run-wide method")
        value = str(labels.iloc[0])
        if metadata.setdefault("method", value) != value:
            raise ValueError("Method labels differ across the run")


def _read_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)


def _effects(root, metadata, test_meta, primary, cis):
    source = pd.read_parquet(root / "element_effects.parquet")
    _check_keys(source, PAIR)
    _method(source, metadata)
    effects = source[PAIR + ["posterior_mean", "posterior_scale"]].copy()
    for name in ("observed_nonzero", "expected_nonzero", "low_information"):
        source_name = name if name in source else f"crt_{name}"
        if source_name in source:
            effects[name] = source[source_name]
    if "p_value" in source:
        saved_primary = test_meta.get("primary_test")
        if saved_primary not in PRIMARY or (primary is not None and primary != saved_primary):
            raise ValueError("Primary test conflicts with the fitted bundle metadata")
        if cis and "cis_family" not in test_meta:
            raise ValueError("This bundle did not declare a cis family at fit time")
        names = ["p_value", "q_global", "q_element"]
        if "q_cis" in test_meta["result_fields"]:
            names.append("q_cis")
        if ("q_cis" in source) != ("q_cis" in names):
            raise ValueError("Cis results differ from the recorded correction family")
        for name in names:
            effects[name] = source[name].astype("float64")
        return effects
    if primary is None:
        if cis or any(
            key in source and source[key].notna().any() for key in ("crt_p_value", "crt_saddlepoint_p_value")
        ):
            raise ValueError("Choose the run's primary test explicitly")
        return effects
    prefix = PRIMARY[primary]
    p, q = prefix + "p_value", prefix + "q_value"
    effects["p_value"] = source[p].astype("float64")
    effects["q_global"] = source[q].astype("float64")
    fields = {
        "p_value": {"file": "element_effects.parquet", "column": p},
        "q_global": {
            "file": "element_effects.parquet",
            "column": q,
            "family": "global",
            "finite_p": int(np.isfinite(source[p]).sum()),
        },
    }
    if cis:
        name = "element_effects_requested_pairs.parquet"
        restricted = pd.read_parquet(root / name, columns=PAIR + [p, q])
        _check_keys(restricted, PAIR)
        idx = pd.MultiIndex.from_frame(effects[PAIR])
        idx_cis = pd.MultiIndex.from_frame(restricted[PAIR])
        positions = idx.get_indexer(idx_cis)
        if (positions < 0).any():
            raise ValueError("Requested pair is absent from the global results")
        _check_equal(source[p].to_numpy()[positions], restricted[p].to_numpy(), "cis source p-values")
        if restricted[q].notna().any():
            effects["q_cis"] = restricted.set_index(PAIR)[q].reindex(idx).to_numpy(dtype="float64")
            fields["q_cis"] = {
                "file": name,
                "column": q,
                "family": "cis",
                "finite_p": int(np.isfinite(restricted[p]).sum()),
            }
    if "q_element" in source:
        effects["q_element"] = source["q_element"].astype("float64")
        fields["q_element"] = {
            "file": "element_effects.parquet",
            "column": "q_element",
            "family": "within_element",
            "finite_p": np.isfinite(source[p]).groupby(source["element"]).sum().to_dict(),
        }
    test_meta.update(primary_test=primary, result_fields=fields)
    return effects


def _efficiency(root, metadata):
    source = pd.read_parquet(root / "guide_efficiency.parquet")
    _check_keys(source, ["guide", "gene"])
    _method(source, metadata)
    if source["element"].isna().any() or source.groupby("guide")["element"].nunique().ne(1).any():
        raise ValueError("Expected a single parent element per guide")
    guides = pd.Index(metadata["guide_names"])
    genes = pd.Index(metadata["gene_names"])
    gi, ri = guides.get_indexer(source["guide"]), genes.get_indexer(source["gene"])
    if (gi < 0).any() or (ri < 0).any() or len(source) != len(guides) * len(genes):
        raise ValueError("Expected complete rc12 guide and response gene axes")
    if not source["element"].isin(metadata["perturbation_names"]).all():
        raise ValueError("Guide has an unknown or non-single parent element")
    result = source[["guide", *PAIR]].copy()
    saved = _read_npz(root / "guide_posteriors.npz")
    for field in GUIDE_FIELDS:
        values = saved[field]
        if values.shape != (len(guides), len(genes)):
            raise ValueError(f"Invalid posterior shape: {field}")
        result[field] = values[gi, ri]
    _check_equal(result[GUIDE_FIELDS[0]], source["posterior_mean"], "guide posterior means")
    return result


def _indices(values, n_cells, key):
    indices = np.asarray(values)
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or len(np.unique(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= n_cells)
    ):
        raise ValueError(f"Invalid source-cell indices: {key}")
    return indices


def export(bundle, destination, *, primary=None, cis=False, source=None):
    """Export a small rc12 bundle in memory, leaving its source files untouched."""
    root, dest = Path(bundle), Path(destination)
    if dest.exists():
        raise FileExistsError(dest)
    metadata = json.loads((root / "metadata.json").read_text())
    setup = metadata.get("setup", {})
    rna_key = setup.get("rna_modality", "rna")
    source_path = root / "mdata.h5mu"
    if source is not None:
        source_path = Path(source)
    elif not source_path.exists():
        source_path = Path(metadata["source"]["path"])
    if source_path.suffix == ".h5ad":
        mdata = md.MuData({rna_key: ad.read_h5ad(source_path)})
    else:
        mdata = md.read_h5mu(source_path)
    if "_perturbo" in mdata.uns:
        raise ValueError("Source already contains PerTurbo export results")
    rna = mdata.mod[rna_key]
    gene_key = setup.get("gene_name_key")
    genes = rna.var[gene_key] if gene_key else rna.var_names
    if list(genes.astype(str)) != metadata["gene_names"]:
        raise ValueError("Source gene IDs or order differ from this fit")
    result = {"metadata": metadata}
    base = root / "control_fit.npz"
    if base.exists():
        result["control_fit"] = _read_npz(base)
    beta = root / "beta_fit.npz"
    if beta.exists():
        with np.load(beta) as saved:
            if "losses" in saved:
                result["beta_fit"] = {"losses": saved["losses"]}
    for key in ("crt_metadata", "covariate_metadata"):
        path = root / (key + ".json")
        if path.exists():
            result[key] = json.loads(path.read_text())

    n = rna.n_obs
    kept = np.arange(n)
    for key in ("baseline_cell_indices", "cell_keep_indices"):
        filename = metadata.get(key + "_file", key + ".npy")
        path = root / filename
        if key in metadata:
            values = metadata[key]
            if path.exists():
                _check_equal(np.asarray(values), np.load(path), key)
        elif path.exists() or key + "_file" in metadata:
            values = np.load(path, allow_pickle=False)
        else:
            continue
        rows = _indices(values, n, key)
        if key == "cell_keep_indices":
            if np.any(rows[1:] <= rows[:-1]):
                raise ValueError("Screen indices must retain source order")
            kept = rows
        else:
            recorded = result.get("control_fit", {}).get("size_factors")
            if recorded is not None and len(recorded) != len(rows):
                raise ValueError("Baseline indices and offsets differ in length")
            metadata["baseline_cell_mapping"] = "verified_indices"
        if not np.array_equal(rows, np.arange(n)):
            result[key] = rows
    if "control_fit" in result:
        metadata.setdefault("baseline_cell_mapping", "unknown")
    offsets = root / "size_factors.npy"
    if offsets.exists():
        result["size_factors"] = np.load(offsets)
        if len(result["size_factors"]) != len(kept):
            raise ValueError("Screen offset axis mismatch")
    cov = root / "cell_covariates.parquet"
    if cov.exists():
        frame = pd.read_parquet(cov)
        if not frame.index.equals(rna.obs_names[kept]):
            raise ValueError("Covariate rows differ from retained source cells")
        result["cell_covariates"] = frame
    elif metadata.get("covariate_transform_state"):
        from .core import CovariateTransformState, apply_covariate_transform

        state = CovariateTransformState(**metadata["covariate_transform_state"])
        data, names = apply_covariate_transform(rna.obs.iloc[kept], state)
        result["cell_covariates"] = pd.DataFrame(data, index=rna.obs_names[kept], columns=names)
    elif "covariate_coef" in result.get("control_fit", {}):
        raise ValueError("Saved covariate coefficients require a design table or frozen transform")
    if (root / "element_effects.parquet").exists():
        test_meta = result.get("crt_metadata", {})
        fx = _effects(root, metadata, test_meta, primary, cis)
        if (
            not fx["gene"].isin(metadata["gene_names"]).all()
            or not fx["element"].isin(metadata["perturbation_names"]).all()
        ):
            raise ValueError("Effect IDs outside fit axes")
        result["effects"] = fx
        if test_meta:
            result["crt_metadata"] = test_meta
    if (root / "guide_efficiency.parquet").exists():
        if metadata.get("guide_effect_strategy") != "relative":
            raise ValueError("Unexpected efficiency table for this strategy")
        result["efficiency"] = _efficiency(root, metadata)
    elif (root / "guide_posteriors.npz").exists():
        raise ValueError("Reconstructing guide parent IDs without a guide table is outside this pilot")

    if "cell_covariates" in result and "covariate_coef" in result.get("control_fit", {}):
        if result["control_fit"]["covariate_coef"].shape != (result["cell_covariates"].shape[1], len(genes)):
            raise ValueError("Covariate design and baseline coefficient axes disagree")
    metadata["source_bundle_format"] = metadata.get("bundle_format", "unknown")
    metadata["bundle_format"] = "fitted_export_pilot"
    metadata["export_schema_version"] = 1
    metadata["export_source_assay"] = str(source_path.resolve())
    mdata.uns["_perturbo"] = result
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        stage = Path(tmp) / "export"
        stage.mkdir()
        mdata.write_h5mu(stage / "fitted.h5mu")
        reloaded = md.read_h5mu(stage / "fitted.h5mu").uns["_perturbo"]
        for key, filename in (("effects", "element_effects.parquet"), ("efficiency", "guide_efficiency.parquet")):
            if key in result:
                # AnnData stores observation-like DataFrame indices as strings.
                result[key].to_parquet(stage / filename, index=False)
                expected = pd.read_parquet(stage / filename)
                actual = reloaded[key].reset_index(drop=True)
                pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual, check_exact=True)
        for key in ("metadata", "crt_metadata", "covariate_metadata"):
            if key in result:
                (stage / f"{key}.json").write_text(json.dumps(result[key], indent=2) + "\n")
        for key in ("control_fit", "beta_fit"):
            if key in result:
                np.savez(stage / f"{key}.npz", **result[key])
        for key in ("size_factors", "cell_keep_indices", "baseline_cell_indices"):
            if key in result:
                np.save(stage / f"{key}.npy", result[key], allow_pickle=False)
        if "cell_covariates" in result:
            result["cell_covariates"].to_parquet(stage / "cell_covariates.parquet")
        if dest.exists():
            raise FileExistsError(dest)
        stage.rename(dest)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--primary", choices=PRIMARY)
    parser.add_argument("--source", type=Path, help="Relocated original source assay file")
    parser.add_argument("--cis", action="store_true", help="Declare the rc12 requested-pairs family to be cis")
    args = parser.parse_args()
    export(args.bundle, args.destination, primary=args.primary, cis=args.cis, source=args.source)
