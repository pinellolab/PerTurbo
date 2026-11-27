"""Publish self-contained MuData or a compact, source-authenticated fit directory."""

import hashlib
import json
import os
import shutil
from pathlib import Path
import tempfile

import anndata as ad
from anndata.io import read_elem, write_elem
import h5py
import mudata as md
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .fitted_restore import BLOCK_ROWS, GUIDE_FIELDS, _axis_codes, _plain, _prepare_assays
from .io import MuDataSetup

REQUESTED = "element_effects_requested_pairs.parquet"
COMPRESSION = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
TABLES = {"effects": "element_effects.parquet", "efficiency": "guide_efficiency.parquet"}
IDS = {"element": "perturbation_names", "gene": "gene_names", "guide": "guide_names"}

EFFECT_COLUMNS = (
    "element",
    "gene",
    "posterior_mean",
    "posterior_scale",
    "p_value",
    "q_global",
    "q_element",
    "q_cis",
    "observed_nonzero",
    "expected_nonzero",
    "low_information",
)


def source_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_assays(src, dst, *, skip_results=False):
    """Losslessly repack ordinary numeric assay arrays with bounded memory."""
    # Creation-order tracking on destination objects allows HDF5 to store
    # large attributes, including column names for wide AnnData tables.
    dst.attrs.update(src.attrs)
    for key, value in src.items():
        if skip_results and value.name == "/uns/_perturbo":
            continue
        if isinstance(value, h5py.Group):
            _copy_assays(value, dst.create_group(key, track_order=True), skip_results=skip_results)
        elif (
            value.ndim
            and value.size
            and value.dtype.kind in "biufc"
            and (value.compression != "gzip" or value.external or value.is_virtual)
        ):
            out = dst.create_dataset(key, shape=value.shape, dtype=value.dtype, track_order=True, **COMPRESSION)
            out.attrs.update(value.attrs)
            row_bytes = int(np.prod(value.shape[1:])) * value.dtype.itemsize
            block = max(1, (8 << 20) // row_bytes)
            for start in range(0, value.shape[0], block):
                out[start : start + block] = value[start : start + block]
        else:
            src.copy(value, dst, name=key)


def _assay_file(source, path, rna_key, *, skip_results=False):
    with h5py.File(source, "r") as src:
        if "mod" in src:
            if "uns/_perturbo" in src and not skip_results:
                raise ValueError("Source already contains PerTurbo results")
            with h5py.File(path, "w", userblock_size=src.userblock_size, track_order=True) as dst:
                _copy_assays(src, dst, skip_results=skip_results)
            if src.userblock_size:
                with open(source, "rb") as inp, open(path, "r+b") as out:
                    out.write(inp.read(src.userblock_size))
        else:
            # Generate native MuData axis maps, then preserve the complete AnnData.
            axes = {}
            for axis in ("obs", "var"):
                group = src[axis]
                axes[axis] = pd.DataFrame(index=read_elem(group[group.attrs["_index"]]))
            md.MuData({rna_key: ad.AnnData(**axes)}).write_h5mu(path)
            with h5py.File(path, "r+") as dst:
                del dst["mod"][rna_key]
                _copy_assays(src, dst["mod"].create_group(rna_key, track_order=True))


def _frames(root, key, metadata, beta, guides):
    parquet = pq.ParquetFile(root / TABLES[key])
    cols = EFFECT_COLUMNS
    if key == "efficiency":
        cols = ["guide", "element", "gene"]
    axes = {name: pd.Index(metadata[field]) for name, field in IDS.items()}
    posterior = (
        {name: np.asarray(beta[name]) for name in ("posterior_mean", "posterior_scale") if name in beta}
        if key == "effects"
        else {name: np.asarray(guides[name]) for name in GUIDE_FIELDS}
    )
    for batch in parquet.iter_batches(
        batch_size=BLOCK_ROWS, columns=[c for c in cols if c in parquet.schema_arrow.names]
    ):
        frame = batch.to_pandas()
        for name in cols:
            if name in axes and name in frame:
                frame[name] = pd.Categorical(frame[name], categories=axes[name])
                if (frame[name].cat.codes < 0).any():
                    raise ValueError(f"Unknown {name} in {key}")
        if key == "effects" and "posterior_mean" in beta:
            ii, jj = frame.element.cat.codes, frame.gene.cat.codes
            for name in ("posterior_mean", "posterior_scale"):
                frame[name] = posterior[name][ii, jj]
        if key == "efficiency":
            gi, ri = frame.guide.cat.codes, frame.gene.cat.codes
            for name in GUIDE_FIELDS:
                frame[name] = posterior[name][gi, ri]
        yield frame


def _native_frame(pt, key, frames, count):
    # Initialize encoding metadata with AnnData itself, including dictionary labels.
    table = None
    start = 0
    for frame in frames:
        if table is None:
            write_elem(pt, key, frame.iloc[:0])
            table = pt[key]
            columns = {}
            idx = table.attrs["_index"]
            for name in [idx, *frame]:
                group = table
                field = name
                if name != idx and isinstance(frame[name].dtype, pd.CategoricalDtype):
                    group, field = table[name], "codes"
                old = group[field]
                attrs = dict(old.attrs)
                dtype = np.int64 if name == idx else old.dtype
                del group[field]
                col = group.create_dataset(field, shape=(count,), dtype=dtype, **COMPRESSION)
                col.attrs.update(attrs)
                columns[name] = col
        stop = start + len(frame)
        columns[idx][start:stop] = np.arange(start, stop)
        for name in frame:
            col = frame[name]
            values = col.cat.codes if isinstance(col.dtype, pd.CategoricalDtype) else col
            columns[name][start:stop] = values.to_numpy()
        start = stop
    if start != count:
        raise ValueError(f"Incomplete {key} table")


def publish(
    root,
    dest,
    output_format,
    *,
    metadata,
    control,
    beta,
    guides,
    offsets,
    keep,
    baseline,
    n_cells,
    obs,
    assays=None,
    overwrite=False,
    skip_source_results=False,
):
    """Publish one artifact atomically. Input tables in root are private fit checkpoints."""
    from . import __version__

    source = Path(metadata["source"]["path"]).resolve() if assays is None else None
    metadata = dict(metadata)
    metadata.setdefault("perturbo_version", __version__)
    metadata = dict(
        metadata,
        bundle_format="compact_v1" if output_format == "bundle" else "fitted_mudata",
        export_schema_version=1,
        method="perturbo",
    )
    if source is not None:
        metadata["source"] = dict(metadata["source"], sha256=source_hash(source))
    arrays = {
        "control_fit": {k: np.asarray(v) for k, v in control.items() if v is not None},
        "beta_fit": {
            k: np.asarray(v)
            for k, v in beta.items()
            if k not in ("posterior_mean", "posterior_scale", "z_values") and v is not None
        },
    }
    extra = {k: np.asarray(v) for k, v in guides.items() if k not in GUIDE_FIELDS and k != "guide_effect_z_values"}
    # Nonstandard guide modes may have guide posteriors without an efficiency table.
    if not (root / TABLES["efficiency"]).exists():
        extra = {k: np.asarray(v) for k, v in guides.items()}
    if extra:
        arrays["guide_fit"] = extra
    if offsets is not None:
        arrays["size_factors"] = np.asarray(offsets)
    if keep is not None and not np.array_equal(keep, np.arange(n_cells)):
        arrays["cell_keep_indices"] = keep
    if baseline is not None:
        metadata["n_baseline_cells"] = len(baseline)
        metadata["baseline_cell_mapping"] = "all_cells"
        if not np.array_equal(baseline, np.arange(n_cells)):
            metadata["baseline_cell_mapping"] = "verified_indices"
            arrays["baseline_cell_indices"] = baseline
    annotations = {}
    for key in ("crt_metadata", "covariate_metadata"):
        file = root / (key + ".json")
        if file.exists():
            annotations[key] = json.loads(file.read_text())
    cov = None
    if (root / "cell_covariates.parquet").exists():
        cov = pd.read_parquet(root / "cell_covariates.parquet")
        rows = obs if keep is None else obs.iloc[keep]
        if not cov.index.equals(rows.index):
            raise ValueError("Covariate rows differ from retained screen cells")
    elif metadata.get("covariate_transform_state"):
        from .core import CovariateTransformState, apply_covariate_transform

        rows = obs if keep is None else obs.iloc[keep]
        values, names = apply_covariate_transform(
            rows, CovariateTransformState(**metadata["covariate_transform_state"])
        )
        cov = pd.DataFrame(values, index=rows.index, columns=names)
    if dest.exists() and not overwrite:
        raise FileExistsError(dest)
    with tempfile.TemporaryDirectory(dir=dest.parent) as temp:
        path = Path(temp) / dest.name
        if output_format == "bundle":
            path.mkdir()
            if assays is not None:
                assays.write_h5mu(path / "source.h5mu", compression="gzip", compression_opts=1)
                metadata["source"] = {"sha256": source_hash(path / "source.h5mu"), "relative_path": "source.h5mu"}
            elif source.is_relative_to(dest.resolve()):
                filename = "source" + source.suffix
                shutil.copyfile(source, path / filename)
                metadata["source"]["relative_path"] = filename
            else:
                metadata["source"]["relative_path"] = os.path.relpath(source, dest)
            (path / "metadata.json").write_text(json.dumps({**metadata, **annotations}, indent=2))
            if (root / REQUESTED).exists():
                shutil.copyfile(root / REQUESTED, path / REQUESTED)
            flat = {}
            for group, value in arrays.items():
                if isinstance(value, dict):
                    flat.update({f"{group}/{k}": v for k, v in value.items() if v is not None})
                else:
                    flat[group] = value
            np.savez_compressed(path / "fit_arrays.npz", **flat)
            if cov is not None:
                cov.to_parquet(path / "cell_covariates.parquet")
            for key, name in TABLES.items():
                if (root / name).exists():
                    writer = None
                    try:
                        for frame in _frames(root, key, metadata, beta, guides):
                            table = pa.Table.from_pandas(frame, preserve_index=False)
                            if writer is None:
                                writer = pq.ParquetWriter(path / name, table.schema)
                            writer.write_table(table)
                    finally:
                        if writer is not None:
                            writer.close()
        else:
            if assays is None:
                _assay_file(source, path, metadata["setup"]["rna_modality"], skip_results=skip_source_results)
            else:
                assays.write_h5mu(path, compression="gzip", compression_opts=1)
            with h5py.File(path, "r+") as h:
                pt = h.require_group("uns").create_group("_perturbo")
                pt.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})
                for key, value in {"metadata": metadata, **annotations}.items():
                    write_elem(pt, key, value)
                for key, value in arrays.items():
                    if isinstance(value, dict):
                        write_elem(pt, key, {})
                        for field, array in value.items():
                            write_elem(pt[key], field, array, dataset_kwargs=COMPRESSION if np.ndim(array) else {})
                    else:
                        write_elem(pt, key, value, dataset_kwargs=COMPRESSION if np.ndim(value) else {})
                if (root / REQUESTED).exists():
                    write_elem(pt, "requested_pairs", pd.read_parquet(root / REQUESTED), dataset_kwargs=COMPRESSION)
                if cov is not None:
                    write_elem(pt, "cell_covariates", cov, dataset_kwargs=COMPRESSION)
                for key, name in TABLES.items():
                    if (root / name).exists():
                        n = pq.ParquetFile(root / name).metadata.num_rows
                        _native_frame(pt, key, _frames(root, key, metadata, beta, guides), n)
        if output_format == "h5mu":
            with h5py.File(path) as check:
                if "uns/_perturbo/effects" not in check or "mod" not in check:
                    raise ValueError("Incomplete fitted MuData")
            if overwrite:
                os.replace(path, dest)
            else:
                os.link(path, dest)
        elif dest.exists() and overwrite:
            previous = Path(temp) / "previous"
            dest.rename(previous)
            try:
                path.rename(dest)
            except OSError:
                previous.rename(dest)
                raise
        else:
            path.rename(dest)


def load_compact(root, relocated=None):
    metadata = json.loads((root / "metadata.json").read_text())
    reference = metadata["source"]
    if relocated is None:
        source = root / reference["relative_path"]
        if not source.exists():
            source = Path(reference.get("path", source))
    else:
        source = Path(relocated)
    if not source.is_file():
        raise FileNotFoundError(f"Original assays unavailable at {source}. Pass source_path= with their new location.")
    if source_hash(source) != reference["sha256"]:
        raise ValueError(f"Source checksum mismatch: {source}")
    with np.load(root / "fit_arrays.npz", allow_pickle=False) as archive:
        arrays = dict(archive)

    def group(prefix):
        return {k.split("/", 1)[1]: v for k, v in arrays.items() if k.startswith(prefix + "/")}

    effects, guides = group("beta_fit"), group("guide_fit")
    for key, target, row, fields in (
        ("effects", effects, "element", ("posterior_mean", "posterior_scale")),
        ("efficiency", guides, "guide", GUIDE_FIELDS),
    ):
        path = root / TABLES[key]
        if not path.exists():
            continue
        rows = pd.Index(metadata[IDS[row]])
        genes = pd.Index(metadata["gene_names"])
        shape = len(rows), len(genes)
        seen = np.zeros(shape, dtype=bool)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BLOCK_ROWS, columns=[row, "gene", *fields]):
            frame = batch.to_pandas()
            i, j = rows.get_indexer(frame[row]), genes.get_indexer(frame.gene)
            if (i < 0).any() or (j < 0).any() or seen[i, j].any() or frame.duplicated([row, "gene"]).any():
                raise ValueError("Invalid fitted pair identifiers")
            seen[i, j] = True
            for name in fields:
                if name not in target:
                    target[name] = np.empty(shape, dtype=frame[name].dtype)
                target[name][i, j] = frame[name]
        if not seen.all():
            raise ValueError("Incomplete fitted pairs")
    with np.errstate(invalid="ignore", divide="ignore"):
        effects["z_values"] = effects["posterior_mean"] / effects["posterior_scale"]
        if "guide_effect_mean" in guides:
            guides["guide_effect_z_values"] = guides["guide_effect_mean"] / guides["guide_effect_scale"]
    setup = MuDataSetup.from_json_dict(metadata["setup"])
    assay = _prepare_assays(source, metadata, setup, arrays.get("cell_keep_indices"), arrays.get("size_factors"))
    return dict(
        metadata=metadata,
        source_file=source.resolve(),
        mdata=assay,
        control_arrays=group("control_fit"),
        beta_arrays=effects,
        guide_posteriors=guides,
    )


def resave(root, source, work, dest, fmt, control, beta, guides, overwrite):
    """Keep the original assays, test families and cell mappings when re-saving a loaded fit."""
    if root.is_file():
        with h5py.File(root) as handle:
            pt = handle["uns/_perturbo"]
            metadata = _plain(read_elem(pt["metadata"]))
            selection = {
                key: pt[key][:] if key in pt else None
                for key in ("size_factors", "baseline_cell_indices", "cell_keep_indices")
            }
            if "requested_pairs" in pt:
                read_elem(pt["requested_pairs"]).to_parquet(work / REQUESTED, index=False)
            if "cell_covariates" in pt:
                read_elem(pt["cell_covariates"]).to_parquet(work / "cell_covariates.parquet")
            for key in ("crt_metadata", "covariate_metadata"):
                if key in pt:
                    (work / (key + ".json")).write_text(json.dumps(_plain(read_elem(pt[key]))))
            for key, name in TABLES.items():
                if key not in pt:
                    continue
                table = pt[key]
                count = len(table[table.attrs["_index"]])
                ids = {field: pd.Index(metadata[axis]) for field, axis in IDS.items() if field in table}
                readers = {field: _axis_codes(table[field], labels) for field, labels in ids.items()}
                writer = None
                try:
                    for i in range(0, count, BLOCK_ROWS):
                        rows = slice(i, min(i + BLOCK_ROWS, count))
                        frame = pd.DataFrame(
                            {
                                col: (
                                    pd.Categorical.from_codes(readers[col](rows), categories=ids[col])
                                    if col in ids
                                    else table[col][rows]
                                )
                                for col in table.attrs["column-order"]
                            }
                        )
                        block = pa.Table.from_pandas(frame, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(work / name, block.schema)
                        writer.write_table(block)
                finally:
                    if writer is not None:
                        writer.close()
    else:
        metadata = json.loads((root / "metadata.json").read_text())
        for key in ("crt_metadata", "covariate_metadata"):
            if key in metadata:
                (work / (key + ".json")).write_text(json.dumps(metadata.pop(key)))
        for name in (*TABLES.values(), "cell_covariates.parquet", REQUESTED):
            if (root / name).exists():
                (work / name).symlink_to((root / name).resolve())
        with np.load(root / "fit_arrays.npz", allow_pickle=False) as z:
            selection = {key: z.get(key) for key in ("size_factors", "baseline_cell_indices", "cell_keep_indices")}
        if source_hash(source) != metadata["source"]["sha256"]:
            raise ValueError("The linked assay changed since the model was loaded")
    with h5py.File(source) as handle:
        rna = handle["mod"][metadata["setup"]["rna_modality"]] if "mod" in handle else handle
        obs = read_elem(rna["obs"])
    metadata["source"] = {**metadata.get("source", {}), "path": str(source)}
    publish(
        work,
        dest,
        fmt,
        metadata=metadata,
        control=control,
        beta=beta,
        guides=guides,
        offsets=selection["size_factors"],
        keep=selection["cell_keep_indices"],
        baseline=selection["baseline_cell_indices"],
        n_cells=len(obs),
        obs=obs,
        skip_source_results=True,
        overwrite=overwrite,
    )
