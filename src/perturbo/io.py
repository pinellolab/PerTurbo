"""I/O helpers for MuData setup and perturbo artifact bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import warnings
from pathlib import Path
import re
from typing import Any

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import scipy.sparse as sp


_SETUP_UNS_KEY = "_perturbo_setup"
# Bundles written by the Cortado prototype carry the old key. They are read and
# upgraded in memory rather than refused, so existing MuData files keep working.
_LEGACY_SETUP_UNS_KEY = "_cortado_setup"
_SAVED_SIZE_FACTOR_KEY = "_perturbo_saved_size_factor"


@dataclass
class MuDataSetup:
    """Registration metadata connecting a MuData object to PerTurbo.

    Attributes
    ----------
    rna_modality, perturbation_modality
        Keys of the RNA and guide modalities in ``mdata.mod``.
    perturbation_layer
        Guide-count layer, or ``None`` to use the perturbation modality's ``X``.
    batch_key, continuous_covariates_keys
        RNA ``obs`` columns used as categorical and continuous covariates.
    library_size_key, size_factor_key
        RNA ``obs`` columns holding library sizes and centered log size factors.
    guide_by_element_key
        Perturbation ``varm`` key mapping guide rows to element columns.
    guide_element_uns_key
        Perturbation ``uns`` key containing element names when the map has no
        labeled columns.
    gene_name_key
        RNA ``var`` column used as gene names, or ``None`` for ``var_names``.
    control_substring
        Substring used to identify control guides or elements.
    size_factor_mode, size_factor_provenance
        Recorded normalization mode and provenance for persistence and reload.

    Notes
    -----
    ``gene_by_element_key`` and ``rna_element_uns_key`` are retained for bundle
    compatibility. The current guide-based workflow uses the corresponding
    guide mapping fields.
    """
    rna_modality: str = "rna"
    perturbation_modality: str = "grna"
    perturbation_layer: str | None = None
    batch_key: str | None = None
    library_size_key: str | None = None
    size_factor_key: str | None = None
    continuous_covariates_keys: list[str] | None = None
    gene_by_element_key: str | None = None
    guide_by_element_key: str | None = None
    rna_element_uns_key: str | None = None
    guide_element_uns_key: str | None = None
    gene_name_key: str | None = None
    control_substring: str | None = None
    size_factor_mode: str | None = None
    size_factor_provenance: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Return registration fields as a JSON-compatible dictionary.

        Missing continuous covariate keys become an empty list. The setup
        object is not modified.
        """
        payload = asdict(self)
        payload["continuous_covariates_keys"] = list(self.continuous_covariates_keys or [])
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "MuDataSetup":
        """Construct registration metadata from serialized fields.

        Parameters
        ----------
        payload
            Setup field names and values. Continuous covariate names are
            normalized to a list of strings; missing values become an empty
            list. The supplied dictionary is not modified.

        Returns
        -------
        MuDataSetup
            Reconstructed registration.
        """
        values = dict(payload)
        raw_covariates = values.get("continuous_covariates_keys")
        if raw_covariates is None:
            values["continuous_covariates_keys"] = []
        else:
            values["continuous_covariates_keys"] = [str(x) for x in list(raw_covariates)]
        return cls(**values)


def _compute_library_size(matrix: Any) -> np.ndarray:
    library_size = np.asarray(matrix.sum(axis=1)).reshape(-1)
    return library_size.astype(np.float32, copy=False)


def _compute_log_size_factor(library_size: np.ndarray) -> np.ndarray:
    safe = np.clip(np.asarray(library_size, dtype=np.float32), a_min=1.0, a_max=None)
    log_cpm = np.log(safe / 1e6)
    return (log_cpm - float(np.mean(log_cpm))).astype(np.float32, copy=False)


def _extract_element_names(
    mdata: md.MuData,
    *,
    perturbation_modality: str,
    guide_by_element_key: str | None,
    guide_element_uns_key: str | None,
) -> list[str]:
    if guide_by_element_key is None:
        return []
    pert = mdata[perturbation_modality]
    if guide_by_element_key not in pert.varm:
        return []
    mapping = pert.varm[guide_by_element_key]
    if hasattr(mapping, "columns"):
        return [str(x) for x in mapping.columns.tolist()]
    if guide_element_uns_key is not None and guide_element_uns_key in pert.uns:
        return [str(x) for x in np.asarray(pert.uns[guide_element_uns_key]).reshape(-1).tolist()]
    return []


def _infer_control_substring(
    mdata: md.MuData,
    *,
    perturbation_modality: str,
    guide_by_element_key: str | None,
    guide_element_uns_key: str | None,
) -> str | None:
    pert = mdata[perturbation_modality]
    candidates = [str(x) for x in pert.var_names.astype(str).tolist()]
    candidates.extend(
        _extract_element_names(
            mdata,
            perturbation_modality=perturbation_modality,
            guide_by_element_key=guide_by_element_key,
            guide_element_uns_key=guide_element_uns_key,
        )
    )
    joined = " ".join(candidates).lower()
    if re.search(r"\bntc\b", joined):
        return "ntc"
    if "non-target" in joined or "nontarget" in joined:
        return "non"
    if "control" in joined:
        return "control"
    return None


def setup_mudata(
    mdata: md.MuData,
    *,
    batch_key: str | None = None,
    library_size_key: str | None = None,
    size_factor_key: str | None = None,
    continuous_covariates_keys: list[str] | None = None,
    gene_by_element_key: str | None = None,
    guide_by_element_key: str | None = None,
    rna_element_uns_key: str | None = None,
    guide_element_uns_key: str | None = None,
    gene_name_key: str | None = None,
    control_substring: str | None = None,
    modalities: dict[str, str] | None = None,
    perturbation_layer: str | None = None,
    size_factor_mode: str | None = None,
    size_factor_provenance: str | None = None,
) -> MuDataSetup:
    """Register an in-memory MuData object for PerTurbo workflows.

    Parameters
    ----------
    mdata
        MuData object containing aligned RNA and perturbation modalities.
    modalities
        Mapping with required keys ``"rna_layer"`` and
        ``"perturbation_layer"``. Their values name modalities in ``mdata.mod``.
    batch_key
        Optional categorical batch column in RNA ``obs``.
    library_size_key, size_factor_key
        RNA ``obs`` columns to use. Missing columns are computed from RNA ``X``;
        omitted names default to ``"_library_size"`` and ``"_size_factor"``.
    continuous_covariates_keys
        Optional continuous RNA ``obs`` columns included in both fitting stages.
    guide_by_element_key
        Perturbation ``varm`` key for a ``guides x elements`` mapping.
    guide_element_uns_key
        Perturbation ``uns`` key with element names when the mapping does not
        expose labeled columns.
    gene_by_element_key, rna_element_uns_key
        RNA mapping metadata retained for bundle compatibility. The current
        guide-based workflow uses ``guide_by_element_key`` and
        ``guide_element_uns_key`` instead.
    gene_name_key
        Optional RNA ``var`` column used for gene names.
    control_substring
        Substring identifying controls. If omitted, a limited name-based
        inference searches for NTC, non-targeting, or control labels.
    perturbation_layer
        Perturbation layer to read instead of the modality's ``X``.
    size_factor_mode, size_factor_provenance
        Normalization metadata stored for later fitting and bundle reload.

    Returns
    -------
    MuDataSetup
        The registration stored in ``mdata.uns["_perturbo_setup"]``.

    Raises
    ------
    ValueError
        If ``modalities`` is not provided.
    KeyError
        If required modality keys or named modalities are absent.

    Notes
    -----
    This function mutates ``mdata``. It may add library size and centered log
    size-factor columns to RNA ``obs``, add ``"_gene_mean"`` to RNA ``var``,
    write ``"_perturbo_setup"`` to ``mdata.uns``, and remove the deprecated
    ``"_cortado_setup"`` key.
    """
    if modalities is None:
        raise ValueError("modalities must be provided.")

    rna_modality = str(modalities["rna_layer"])
    perturbation_modality = str(modalities["perturbation_layer"])
    rna = mdata[rna_modality]

    if library_size_key is None:
        library_size_key = "_library_size"
    if library_size_key not in rna.obs.columns:
        rna.obs[library_size_key] = _compute_library_size(rna.X)

    if size_factor_key is None:
        size_factor_key = "_size_factor"
    if size_factor_key not in rna.obs.columns:
        rna.obs[size_factor_key] = _compute_log_size_factor(np.asarray(rna.obs[library_size_key]))

    if "_gene_mean" not in rna.var.columns:
        gene_mean = np.asarray(rna.X.mean(axis=0)).reshape(-1)
        rna.var["_gene_mean"] = gene_mean.astype(np.float32, copy=False)

    setup = MuDataSetup(
        rna_modality=rna_modality,
        perturbation_modality=perturbation_modality,
        perturbation_layer=perturbation_layer,
        batch_key=batch_key,
        library_size_key=library_size_key,
        size_factor_key=size_factor_key,
        continuous_covariates_keys=list(continuous_covariates_keys or []),
        gene_by_element_key=gene_by_element_key,
        guide_by_element_key=guide_by_element_key,
        rna_element_uns_key=rna_element_uns_key,
        guide_element_uns_key=guide_element_uns_key,
        gene_name_key=gene_name_key,
        control_substring=control_substring,
        size_factor_mode=size_factor_mode,
        size_factor_provenance=size_factor_provenance,
    )
    if setup.control_substring is None:
        setup.control_substring = _infer_control_substring(
            mdata,
            perturbation_modality=perturbation_modality,
            guide_by_element_key=guide_by_element_key,
            guide_element_uns_key=guide_element_uns_key,
        )
    mdata.uns[_SETUP_UNS_KEY] = setup.to_json_dict()
    mdata.uns.pop(_LEGACY_SETUP_UNS_KEY, None)
    return setup


def get_mudata_setup(mdata: md.MuData) -> MuDataSetup:
    """Read a PerTurbo registration from a MuData object.

    Parameters
    ----------
    mdata
        Registered MuData object.

    Returns
    -------
    MuDataSetup
        Parsed registration metadata.

    Raises
    ------
    KeyError
        If neither the current nor legacy registration key exists.

    Notes
    -----
    Reading a legacy ``"_cortado_setup"`` registration warns, writes the same
    payload under ``"_perturbo_setup"``, and removes the legacy key in memory.
    """
    if _SETUP_UNS_KEY in mdata.uns:
        payload = mdata.uns[_SETUP_UNS_KEY]
        mdata.uns.pop(_LEGACY_SETUP_UNS_KEY, None)
    elif _LEGACY_SETUP_UNS_KEY in mdata.uns:
        warnings.warn(
            "MuData uses the deprecated '_cortado_setup' registration key; "
            "it has been upgraded in memory to '_perturbo_setup'.",
            DeprecationWarning,
            stacklevel=2,
        )
        payload = mdata.uns[_LEGACY_SETUP_UNS_KEY]
        mdata.uns[_SETUP_UNS_KEY] = payload
        del mdata.uns[_LEGACY_SETUP_UNS_KEY]
    else:
        raise KeyError(
            "MuData is not registered for perturbo. Call setup_mudata(...) first."
        )
    if isinstance(payload, str):
        payload = json.loads(payload)
    return MuDataSetup.from_json_dict(dict(payload))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def save_array_bundle(path: str | Path, arrays: dict[str, Any]) -> None:
    payload = {key: np.asarray(value) for key, value in arrays.items() if value is not None}
    np.savez(Path(path), **payload)


def control_fit_arrays(control_fit: Any) -> dict[str, Any]:
    arrays = {
        "beta_0": control_fit.beta_0,
        "theta": control_fit.theta,
        "noise_scale": control_fit.noise_scale,
        "factor_loadings": control_fit.factor_loadings,
        "factor_scores": control_fit.factor_scores,
        "factor_center": control_fit.factor_center,
        "pca_loadings": control_fit.pca_loadings,
        "size_factors": control_fit.size_factors,
        "losses": control_fit.losses,
        "pi_outlier": control_fit.pi_outlier,
        "theta_outlier": control_fit.theta_outlier,
        "outlier_mean_shift": control_fit.outlier_mean_shift,
        "covariate_coef": control_fit.covariate_coef,
        "guide_random_effect_tau": control_fit.guide_random_effect_tau,
        "guide_random_effect_log_tau_loc": control_fit.guide_random_effect_log_tau_loc,
        "guide_random_effect_log_tau_scale": control_fit.guide_random_effect_log_tau_scale,
        "count_censoring_threshold": control_fit.count_censoring_threshold,
    }
    if control_fit.baseline_posterior is not None:
        arrays.update(
            {
                "baseline_beta_0_loc": control_fit.baseline_posterior.beta_0_loc,
                "baseline_beta_0_scale": control_fit.baseline_posterior.beta_0_scale,
                "baseline_theta_log_loc": control_fit.baseline_posterior.theta_log_loc,
                "baseline_theta_log_scale": control_fit.baseline_posterior.theta_log_scale,
            }
        )
    return arrays


def load_array_bundle(path: str | Path) -> dict[str, np.ndarray]:
    bundle_path = Path(path)
    if not bundle_path.exists():
        return {}
    with np.load(bundle_path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def save_fit_bundle(
    out_dir: str | Path,
    *,
    metadata: dict[str, Any],
    mdata: md.MuData,
    control_arrays: dict[str, Any],
    beta_arrays: dict[str, Any],
    element_effects: pd.DataFrame | None = None,
    guide_posteriors: dict[str, Any] | None = None,
    guide_efficacy: np.ndarray | None = None,
) -> Path:
    """Write a complete fitted-model bundle.

    Parameters
    ----------
    out_dir
        Destination directory, created if needed.
    metadata
        JSON-serializable model and workflow metadata. Producer and version
        fields are added by this function.
    mdata
        Registered data written as ``mdata.h5mu``.
    control_arrays, beta_arrays
        Named arrays written to ``control_fit.npz`` and ``beta_fit.npz``.
    element_effects
        Optional long-form element table written as Parquet.
    guide_posteriors, guide_efficacy
        Optional guide-level arrays.

    Returns
    -------
    pathlib.Path
        The bundle directory.

    Notes
    -----
    Existing files with the standard bundle names are overwritten. This
    low-level writer does not perform the non-empty-directory check used by
    :meth:`perturbo.PerTurboModel.save`.
    """
    bundle_dir = Path(out_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    mdata.write_h5mu(bundle_dir / "mdata.h5mu")
    # Bundles declare who wrote them and in what shape, so a reader can tell a v2
    # bundle from a Cortado-prototype one without guessing from its contents.
    payload = dict(metadata)
    payload["producer"] = "perturbo"
    payload["bundle_format"] = "fit_bundle"
    payload["bundle_version"] = 2
    write_json(bundle_dir / "metadata.json", payload)
    save_array_bundle(bundle_dir / "control_fit.npz", control_arrays)
    save_array_bundle(bundle_dir / "beta_fit.npz", beta_arrays)
    if element_effects is not None:
        element_effects.to_parquet(bundle_dir / "element_effects.parquet", index=False)
    if guide_posteriors is not None:
        save_array_bundle(bundle_dir / "guide_posteriors.npz", guide_posteriors)
    if guide_efficacy is not None:
        np.save(bundle_dir / "guide_efficacy.npy", np.asarray(guide_efficacy))
    return bundle_dir


def save_light_fit_bundle(
    out_dir: str | Path,
    *,
    metadata: dict[str, Any],
    control_arrays: dict[str, Any],
    beta_arrays: dict[str, Any],
    guide_posteriors: dict[str, Any] | None = None,
    guide_efficacy: np.ndarray | None = None,
    size_factors: np.ndarray | None = None,
    cell_keep_indices: np.ndarray | None = None,
) -> Path:
    bundle_dir = Path(out_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload["producer"] = "perturbo"
    payload["bundle_format"] = "light_model_params"
    payload["bundle_version"] = 2
    if cell_keep_indices is not None:
        keep = np.asarray(cell_keep_indices, dtype=np.int64).reshape(-1)
        np.save(bundle_dir / "cell_keep_indices.npy", keep)
        payload["cell_keep_indices_file"] = "cell_keep_indices.npy"
    if size_factors is not None:
        values = np.asarray(size_factors, dtype=np.float32).reshape(-1, 1)
        np.save(bundle_dir / "size_factors.npy", values)
        payload["size_factors_file"] = "size_factors.npy"
    write_json(bundle_dir / "metadata.json", payload)
    save_array_bundle(bundle_dir / "control_fit.npz", control_arrays)
    save_array_bundle(bundle_dir / "beta_fit.npz", beta_arrays)
    if guide_posteriors is not None:
        save_array_bundle(bundle_dir / "guide_posteriors.npz", guide_posteriors)
    if guide_efficacy is not None:
        np.save(bundle_dir / "guide_efficacy.npy", np.asarray(guide_efficacy))
    return bundle_dir


def _read_element_effects(root: Path) -> pd.DataFrame | None:
    parquet_path = root / "element_effects.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    csv_path = root / "element_effects.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def _read_cell_keep_indices(root: Path, metadata: dict[str, Any]) -> np.ndarray | None:
    if "cell_keep_indices" in metadata:
        return np.asarray(metadata["cell_keep_indices"], dtype=np.int64).reshape(-1)
    keep_file = metadata.get("cell_keep_indices_file")
    if keep_file is None:
        return None
    keep_path = root / str(keep_file)
    if not keep_path.exists():
        raise FileNotFoundError(f"Light fit bundle references missing cell keep index file: {keep_path}")
    return np.load(keep_path).astype(np.int64, copy=False).reshape(-1)


def _source_path_from_metadata(metadata: dict[str, Any]) -> Path:
    source = metadata.get("source") or {}
    raw_path = source.get("path") or metadata.get("source_path")
    if raw_path is None:
        raise ValueError(
            "Light fit bundle is missing source path metadata. "
            "Recreate the bundle or pass a source AnnData/MuData object explicitly."
        )
    path = Path(str(raw_path)).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Light fit bundle source data is missing: {path}. "
            "Restore the original training data path or pass adata=... to PERTURBO.load(...)."
        )
    return path


def _load_source_data(metadata: dict[str, Any], source_data: Any | None) -> Any:
    if source_data is not None:
        return source_data
    source_path = _source_path_from_metadata(metadata)
    if source_path.suffix == ".h5ad":
        return ad.read_h5ad(source_path, backed="r")
    if source_path.suffix == ".h5mu":
        try:
            return md.read_h5mu(source_path, backed="r")
        except TypeError:
            return md.read_h5mu(source_path)
    raise ValueError(f"Unsupported light bundle source extension: {source_path.suffix!r}")


def _extract_source_gene_names(rna: Any, gene_name_key: str | None) -> list[str]:
    if gene_name_key is not None and gene_name_key in rna.var.columns:
        return [str(x) for x in rna.var[gene_name_key].astype(str).tolist()]
    return [str(x) for x in rna.var_names.astype(str).tolist()]


def _subset_source_by_obs_indices(data: Any, setup: MuDataSetup, keep_indices: np.ndarray | None) -> Any:
    if keep_indices is None:
        return data
    keep = np.asarray(keep_indices, dtype=np.int64).reshape(-1)
    if hasattr(data, "mod"):
        rna = data[setup.rna_modality]
        obs_names = rna.obs_names[keep]
        mods = {}
        for name, mod in data.mod.items():
            try:
                mods[name] = mod[obs_names]
            except Exception:
                mods[name] = mod
        subset = md.MuData(mods)
        subset.uns.update(dict(getattr(data, "uns", {})))
        return subset
    return data[keep]


def _low_moi_source_to_mudata(data: Any, metadata: dict[str, Any], setup: MuDataSetup) -> md.MuData:
    if hasattr(data, "mod"):
        rna = data[setup.rna_modality]
    else:
        rna = data
    perturbation_key = metadata.get("perturbation_key")
    if perturbation_key is None:
        cli_payload = metadata.get("cli_args") or {}
        perturbation_key = cli_payload.get("perturbation_key")
    if perturbation_key is None:
        raise ValueError("Light low-MOI bundle is missing perturbation_key metadata.")
    if perturbation_key not in rna.obs.columns:
        raise KeyError(f"perturbation_key '{perturbation_key}' not found in source obs.")

    pert_names = [str(x) for x in metadata.get("perturbation_names", [])]
    if not pert_names:
        pert_names = [str(x) for x in pd.Categorical(rna.obs[perturbation_key].astype(str)).categories.tolist()]
    categories = pd.Categorical(rna.obs[perturbation_key].astype(str), categories=pert_names)
    codes = np.asarray(categories.codes, dtype=np.int64)
    valid = codes >= 0
    matrix = sp.csr_matrix(
        (
            np.ones(int(np.count_nonzero(valid)), dtype=np.float32),
            (np.flatnonzero(valid), codes[valid]),
        ),
        shape=(rna.n_obs, len(pert_names)),
    )
    pert = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=rna.obs_names.copy()),
        var=pd.DataFrame(index=pd.Index(pert_names, dtype="object")),
    )
    mdata = md.MuData({setup.rna_modality: rna, setup.perturbation_modality: pert})
    mdata.uns[_SETUP_UNS_KEY] = setup.to_json_dict()
    return mdata


def _load_light_bundle_mdata(root: Path, metadata: dict[str, Any], source_data: Any | None) -> md.MuData:
    setup = MuDataSetup.from_json_dict(dict(metadata["setup"]))
    source = _load_source_data(metadata, source_data)
    source = _subset_source_by_obs_indices(source, setup, _read_cell_keep_indices(root, metadata))
    if setup.perturbation_modality in getattr(source, "mod", {}):
        mdata = source
    else:
        mdata = _low_moi_source_to_mudata(source, metadata, setup)

    size_factors_file = metadata.get("size_factors_file")
    if size_factors_file is not None:
        values_path = root / str(size_factors_file)
        if not values_path.exists():
            raise FileNotFoundError(f"Light fit bundle references missing size-factor file: {values_path}")
        values = np.asarray(np.load(values_path), dtype=np.float32).reshape(-1)
        rna = mdata[setup.rna_modality]
        if values.shape[0] != rna.n_obs:
            raise ValueError(
                "Saved size-factor row count does not match the light bundle source data: "
                f"{values.shape[0]} != {rna.n_obs}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Saved size factors must contain only finite values.")
        obs = rna.obs.copy()
        obs[_SAVED_SIZE_FACTOR_KEY] = values
        rna.obs = obs
        setup.size_factor_key = _SAVED_SIZE_FACTOR_KEY
        setup.library_size_key = None
    if setup.size_factor_mode is None:
        raw_mode = metadata.get("size_factor_mode")
        setup.size_factor_mode = None if raw_mode is None else str(raw_mode)
    if setup.size_factor_provenance is None:
        raw_provenance = metadata.get("size_factor_provenance")
        setup.size_factor_provenance = None if raw_provenance is None else str(raw_provenance)
    mdata.uns[_SETUP_UNS_KEY] = setup.to_json_dict()

    expected_genes = metadata.get("gene_names")
    if expected_genes is not None:
        actual_genes = _extract_source_gene_names(mdata[setup.rna_modality], setup.gene_name_key)
        if list(map(str, expected_genes)) != actual_genes:
            raise ValueError("Light fit bundle gene names do not match the source data.")
    return mdata


def load_fit_bundle(bundle_dir: str | Path, *, source_data: Any | None = None) -> dict[str, Any]:
    """Load a complete or light fitted-model bundle.

    Parameters
    ----------
    bundle_dir
        Directory containing ``metadata.json`` and fitted array archives.
    source_data
        Optional AnnData or MuData source for a light bundle. If omitted, the
        source path recorded in metadata is opened, backed when supported.

    Returns
    -------
    dict
        Metadata, reconstructed ``mdata``, control and beta arrays, optional
        element effects, guide posteriors, and guide efficacy.

    Raises
    ------
    FileNotFoundError
        If required metadata or a referenced light-bundle source is missing.
    ValueError
        If a light source has an unsupported extension, incompatible genes or
        cell counts, or invalid saved size factors.

    Notes
    -----
    Complete bundles load ``mdata.h5mu`` from disk. Light bundles reconstruct
    the registered data from ``source_data`` or recorded source metadata and
    restore the exact saved size-factor vector when present.
    """
    root = Path(bundle_dir)
    metadata = read_json(root / "metadata.json")
    mdata_path = root / "mdata.h5mu"
    if mdata_path.exists():
        mdata = md.read_h5mu(mdata_path)
    else:
        mdata = _load_light_bundle_mdata(root, metadata, source_data)
    guide_efficacy_path = root / "guide_efficacy.npy"
    return {
        "metadata": metadata,
        "mdata": mdata,
        "control_arrays": load_array_bundle(root / "control_fit.npz"),
        "beta_arrays": load_array_bundle(root / "beta_fit.npz"),
        "element_effects": _read_element_effects(root),
        "guide_posteriors": load_array_bundle(root / "guide_posteriors.npz"),
        "guide_efficacy": np.load(guide_efficacy_path) if guide_efficacy_path.exists() else None,
    }


__all__ = [
    "MuDataSetup",
    "control_fit_arrays",
    "get_mudata_setup",
    "load_fit_bundle",
    "save_light_fit_bundle",
    "save_fit_bundle",
    "setup_mudata",
]
