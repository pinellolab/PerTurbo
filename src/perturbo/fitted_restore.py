"""Restore the manuscript-scope fitted MuData pilot. Streams native result table columns."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import h5py
import mudata as md
import numpy as np
import pandas as pd
from anndata.io import read_elem

from .io import (
    MuDataSetup,
    _extract_element_names,
    _low_moi_source_to_mudata,
    _SAVED_SIZE_FACTOR_KEY,
    _SETUP_UNS_KEY,
)

BLOCK_ROWS = 131072
GUIDE_FIELDS = (
    "guide_effect_mean",
    "guide_effect_scale",
    "guide_relative_efficiency_mean",
    "guide_relative_efficiency_scale",
)


def _plain(value):
    """Restore JSON metadata containers after MuData encodes string lists as arrays."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _axis_codes(column, expected):
    """Return a block reader; categorical dictionary labels are translated once."""
    if not expected.is_unique or expected.hasnans:
        raise ValueError("Nonunique or missing fit axis identifiers")
    if isinstance(column, h5py.Dataset):
        strings = column.asstr()
        return lambda rows: expected.get_indexer(strings[rows])
    encoding = column.attrs.get("encoding-type")
    if encoding == "nullable-string-array":
        values = column["values"].asstr()
        mask = column["mask"]
        if values.shape != mask.shape:
            raise ValueError(f"Identifier values and mask differ in shape at {column.name}")

        def read_strings(rows):
            if mask[rows].any():
                raise ValueError(f"Missing fit identifier at {column.name}")
            return expected.get_indexer(values[rows])

        return read_strings
    if encoding != "categorical":
        raise ValueError(f"Unsupported identifier encoding at {column.name}")
    lookup = _axis_codes(column["categories"], expected)(slice(None))
    codes = column["codes"]

    def read(rows):
        v = codes[rows]
        if v.min() < 0 or v.max() >= len(lookup):
            raise ValueError(f"Missing/invalid category code at {column.name}")
        return lookup[v]

    return read


def _restore_matrix(table, axes, fields):
    if table.attrs.get("encoding-type") != "dataframe":
        raise ValueError(f"Expected native AnnData DataFrame at {table.name}")
    (r_key, r_idx), (c_key, c_idx) = axes
    n, m = len(r_idx), len(c_idx)
    if not n or not m:
        raise ValueError("Empty fitted axis")
    index_key = table.attrs["_index"]
    if len(table[index_key]) != n * m:
        raise ValueError("Restoring a fitted model requires its complete rectangular posterior domain")
    read_r = _axis_codes(table[r_key], r_idx)
    read_c = _axis_codes(table[c_key], c_idx)
    seen = np.zeros(n * m, dtype=bool)
    arrays = {}
    for column in fields:
        data = table[column]
        if data.dtype.kind != "f" or data.shape != (n * m,):
            raise ValueError(f"Expected floating-point result column on full domain: {column}")
        arrays[column] = np.empty((n, m), dtype=data.dtype)
    for start in range(0, n * m, BLOCK_ROWS):
        rows = slice(start, min(start + BLOCK_ROWS, n * m))
        rr, cc = read_r(rows), read_c(rows)
        if len(rr) != len(cc) or len(rr) != rows.stop - start or (rr < 0).any() or (cc < 0).any():
            raise ValueError("Result IDs conflict with saved fit axes")
        flat = rr * m + cc
        if seen[flat].any() or len(np.unique(flat)) != len(flat):
            raise ValueError("Duplicate fitted result pair")
        seen[flat] = True
        for column, array in arrays.items():
            array.flat[flat] = table[column][rows]
    return arrays


def _prepare_assays(path, metadata, setup, keep, offsets):
    # Read the source modalities backed individually. Generic read_h5mu(backed='r')
    # would materialize the entire result DataFrame from the root .uns during open.
    if Path(path).suffix == ".h5ad":
        import anndata as ad
        rna = ad.read_h5ad(path, backed="r")
    else:
        rna = md.read_h5ad(path, mod=setup.rna_modality, backed="r")
    if keep is not None:
        if (
            keep.ndim != 1
            or keep.dtype.kind not in "ui"
            or (keep < 0).any()
            or (keep >= rna.n_obs).any()
            or (keep[1:] <= keep[:-1]).any()
        ):
            raise ValueError("Invalid retained source rows")
        rna = rna[keep]
    if setup.gene_name_key:
        genes = rna.var[setup.gene_name_key]
    else:
        genes = rna.var_names
    expected = pd.Index(metadata["gene_names"])
    if not expected.equals(pd.Index(genes).astype(str)):
        raise ValueError("Source RNA genes differ from the fitted gene axis")
    # Keep counts backed; reassigning obs on a backed view would materialize it.
    if rna.is_view and (offsets is not None or setup.gene_name_key):
        rna = rna.to_memory()
    if setup.gene_name_key:
        rna.var_names = expected
    if offsets is not None:
        values = np.asarray(offsets).reshape(-1)
        if len(values) != rna.n_obs or not np.isfinite(values).all():
            raise ValueError("Invalid analysis-cell offsets")
        obs = rna.obs.copy()
        obs[_SAVED_SIZE_FACTOR_KEY] = values
        rna.obs = obs
        setup.size_factor_key = _SAVED_SIZE_FACTOR_KEY
        setup.library_size_key = None
    if metadata.get("input_mode") == "low_moi":
        # Many .h5mu inputs carry measured guide sequences, whereas rc12 fits
        # constructs from one categorical label per cell. Reuse rc12's exact map.
        mdata = _low_moi_source_to_mudata(rna, metadata, setup)
    else:
        pert = md.read_h5ad(path, mod=setup.perturbation_modality, backed="r")
        if keep is not None or not pert.obs_names.equals(rna.obs_names):
            pert = pert[rna.obs_names]
        if not np.array_equal(pert.obs_names.to_numpy(), rna.obs_names.to_numpy()):
            raise ValueError("RNA and guide cell axes differ")
        mdata = md.MuData({setup.rna_modality: rna, setup.perturbation_modality: pert})
    actual_guide = mdata[setup.perturbation_modality].var_names
    if not np.array_equal(metadata["guide_names"], actual_guide.to_numpy()):
        raise ValueError("Guide IDs conflict with the fitted guide axis")
    actual_elements = _extract_element_names(
        mdata,
        perturbation_modality=setup.perturbation_modality,
        guide_by_element_key=setup.guide_by_element_key,
        guide_element_uns_key=setup.guide_element_uns_key,
    )
    if setup.guide_by_element_key is None:
        actual_elements = list(actual_guide)
    if list(actual_elements) != list(metadata["perturbation_names"]):
        raise ValueError("Element IDs conflict with the source assignment axis")
    mdata.uns[_SETUP_UNS_KEY] = setup.to_json_dict()
    return mdata


def load_native_fit(path):
    with h5py.File(path, "r") as h:
        pt = h["uns/_perturbo"]
        metadata = _plain(read_elem(pt["metadata"]))
        if "effects" not in pt:
            raise ValueError("A stage-two fitted model requires an effects table")
        control = read_elem(pt["control_fit"])
        genes = ("gene", pd.Index(metadata["gene_names"]))
        effects = _restore_matrix(
            pt["effects"],
            (("element", pd.Index(metadata["perturbation_names"])), genes),
            ("posterior_mean", "posterior_scale"),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            effects["z_values"] = effects["posterior_mean"] / effects["posterior_scale"]
        if "beta_fit" in pt:
            effects.update(read_elem(pt["beta_fit"]))
        guides = read_elem(pt["guide_fit"]) if "guide_fit" in pt else {}
        if "efficiency" in pt:
            guides.update(_restore_matrix(
                pt["efficiency"], (("guide", pd.Index(metadata["guide_names"])), genes), GUIDE_FIELDS
            ))
            with np.errstate(divide="ignore", invalid="ignore"):
                guides["guide_effect_z_values"] = guides["guide_effect_mean"] / guides["guide_effect_scale"]
        keep = pt["cell_keep_indices"][:] if "cell_keep_indices" in pt else None
        offsets = pt["size_factors"][:] if "size_factors" in pt else None
    setup = MuDataSetup.from_json_dict(metadata["setup"])
    assay = _prepare_assays(path, metadata, setup, keep, offsets)
    return dict(metadata=metadata, mdata=assay, control_arrays=control, beta_arrays=effects, guide_posteriors=guides)


def audit(path, reference):
    from .inference import PerTurboModel

    tick = time.perf_counter()
    model = PerTurboModel.load(path)
    report = {
        "restore_seconds": time.perf_counter() - tick,
        "restore_peak_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        "effects_shape": list(model.beta_fit.posterior_mean.shape),
        "guide_strategy": model.guide_effect_strategy,
    }
    metadata = json.loads((reference / "metadata.json").read_text())
    for ids, key in [
        (model.gene_names, "gene_names"),
        (model.element_names, "perturbation_names"),
        (model.adata[model.setup.perturbation_modality].var_names.tolist(), "guide_names"),
    ]:
        if ids != metadata[key]:
            raise AssertionError(f"Reference axis mismatch: {key}")
    with np.load(reference / "beta_fit.npz") as z:
        for key in ("posterior_mean", "posterior_scale", "z_values"):
            orig = z[key]
            restored = getattr(model.beta_fit, key)
            if orig.shape != restored.shape or orig.dtype != restored.dtype:
                raise AssertionError(f"Reference shape/dtype mismatch: {key}")
            for i in range(0, len(orig), 128):
                left, right = orig[i : i + 128], restored[i : i + 128]
                if key == "z_values":
                    # rc12 can divide on a GPU. The manuscript schema stores the
                    # two operands exactly and intentionally omits this derived field.
                    np.testing.assert_allclose(left, right, rtol=3 * np.finfo(orig.dtype).eps, atol=0, equal_nan=True)
                else:
                    np.testing.assert_array_equal(left, right)
    if model.guide_effect_strategy == "relative":
        with np.load(reference / "guide_posteriors.npz") as z:
            for key in GUIDE_FIELDS + ("guide_effect_z_values",):
                if key == "guide_effect_z_values":
                    np.testing.assert_allclose(
                        getattr(model.beta_fit, key),
                        z[key],
                        rtol=3 * np.finfo(z[key].dtype).eps,
                        atol=0,
                        equal_nan=True,
                    )
                else:
                    np.testing.assert_array_equal(getattr(model.beta_fit, key), z[key])
    from .io import control_fit_arrays

    restored_control = control_fit_arrays(model.control_fit)
    with np.load(reference / "control_fit.npz") as z:
        for key in z.files:
            np.testing.assert_array_equal(restored_control[key], z[key])
    if (reference / "size_factors.npy").is_file():
        np.testing.assert_array_equal(
            model.adata[model.setup.rna_modality].obs[_SAVED_SIZE_FACTOR_KEY].to_numpy(),
            np.load(reference / "size_factors.npy").reshape(-1),
        )
    report.update(
        status="PASS",
        saved_means_scales="exact",
        derived_z_check="3 dtype eps relative, 0 absolute",
        baseline_fields="exact",
        axes="exact",
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file", type=Path)
    p.add_argument("--reference-bundle", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.file, a.reference_bundle)), flush=True)
