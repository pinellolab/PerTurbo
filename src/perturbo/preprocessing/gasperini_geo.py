from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


@dataclass(frozen=True)
class GasperiniGeoSpec:
    dataset: str
    prefix: str
    pheno_columns: tuple[str, ...]
    grna_groups_filename: str
    gene_grna_pair_filename: str

    @property
    def expr_filename(self) -> str:
        return f"{self.prefix}.exprs.mtx.gz"

    @property
    def genes_filename(self) -> str:
        if self.dataset == "at_scale":
            return f"{self.prefix}.genes.txt.gz"
        return f"{self.prefix}.genes.txt"

    @property
    def pheno_filename(self) -> str:
        if self.dataset == "at_scale":
            return f"{self.prefix}.phenoData.txt.gz"
        return f"{self.prefix}.phenoData.txt"

    @property
    def output_filename(self) -> str:
        return f"{self.prefix}.h5mu"


GASPERINI_GEO_SPECS: dict[str, GasperiniGeoSpec] = {
    "at_scale": GasperiniGeoSpec(
        dataset="at_scale",
        prefix="GSE120861_at_scale_screen",
        pheno_columns=(
            "sample",
            "cell",
            "total_umis",
            "size_factor",
            "gene",
            "all_gene",
            "barcode",
            "read_count",
            "umi_count",
            "proportion",
            "guide_count",
            "sample_directory",
            "ko_barcode_file",
            "id",
            "prep_batch",
            "within_batch_chip",
            "within_chip_lane",
            "percent_mito",
        ),
        grna_groups_filename="GSE120861_grna_groups.at_scale.txt.gz",
        gene_grna_pair_filename="GSE120861_gene_gRNAgroup_pair_table.at_scale.txt",
    ),
    "pilot": GasperiniGeoSpec(
        dataset="pilot",
        prefix="GSE120861_pilot_highmoi_screen",
        pheno_columns=(
            "sample",
            "cell",
            "total_umis",
            "size_factor",
            "gene",
            "all_gene",
            "barcode",
            "read_count",
            "umi_count",
            "proportion",
            "guide_count",
            "sample_directory",
            "ko_barcode_file",
            "sample_name",
        ),
        grna_groups_filename="GSE120861_grna_groups.pilot.txt",
        gene_grna_pair_filename="GSE120861_gene_gRNAgroup_pair_table.pilot.txt",
    ),
}

_CELL_METADATA_DROP_COLUMNS = [
    "gene",
    "all_gene",
    "barcode",
    "sample_directory",
    "ko_barcode_file",
]

_GENE_METADATA_RENAME = {
    "chr.targetgene": "gene_chr",
    "start.targetgene": "gene_start",
    "stop.targetgene": "gene_end",
    "ENSG.targetgene": "gene_id",
    "targetgene_short_name": "symbol",
    "strand.targetgene": "gene_strand",
}

_GUIDE_METADATA_RENAME = {
    "gRNAgroup.chr": "intended_target_chr",
    "gRNAgroup.start": "intended_target_start",
    "gRNAgroup.stop": "intended_target_end",
    "gRNAgroup": "intended_target_name",
    "general_group": "guide_group",
}


def summarize_gasperini_geo_inputs(data_dir: str | Path, dataset: str) -> dict[str, Any]:
    spec = _get_spec(dataset)
    data_dir = Path(data_dir).expanduser()

    matrix_shape = _read_matrix_market_shape(data_dir / spec.expr_filename)
    gene_ids = _read_gene_ids(data_dir / spec.genes_filename)
    obs = _read_cell_metadata(data_dir / spec.pheno_filename, spec)
    guide_targets = _read_guide_targets(data_dir / spec.grna_groups_filename)
    guide_pairs = _read_guide_pair_table(data_dir / spec.gene_grna_pair_filename)

    guide_metadata = _build_guide_metadata(guide_pairs)
    guide_var = _build_guide_var(guide_targets, guide_metadata)
    unknown_guides = _find_unknown_barcode_guides(obs["barcode"], guide_var.index)
    missing_group_metadata = sorted(set(guide_targets["intended_target_name"]) - set(guide_pairs["gRNAgroup"]))

    return {
        "dataset": spec.dataset,
        "prefix": spec.prefix,
        "matrix_gene_rows": matrix_shape[0],
        "matrix_cell_columns": matrix_shape[1],
        "genes_file_rows": len(gene_ids),
        "cell_metadata_rows": len(obs),
        "cell_metadata_columns_used": list(spec.pheno_columns),
        "guide_rows": len(guide_targets),
        "guide_groups": int(guide_targets["intended_target_name"].nunique()),
        "guide_groups_missing_pair_metadata": missing_group_metadata,
        "guides_with_missing_target_coordinates": int(
            guide_var["intended_target_chr"].isna().sum()
        ),
        "unknown_barcode_guides": unknown_guides,
        "expression_matrix_is_gzip": _is_gzip_path(data_dir / spec.expr_filename),
    }


def build_gasperini_geo_mudata(
    data_dir: str | Path,
    dataset: str,
    *,
    gene_id_subset: list[str] | pd.Index | pd.Series | None = None,
) -> md.MuData:
    spec = _get_spec(dataset)
    data_dir = Path(data_dir).expanduser()

    counts = _read_counts_matrix(data_dir / spec.expr_filename)
    gene_ids = _read_gene_ids(data_dir / spec.genes_filename)
    if counts.shape[1] != len(gene_ids):
        raise ValueError(
            f"{spec.dataset}: gene matrix width {counts.shape[1]} does not match gene ID count {len(gene_ids)}."
        )

    obs = _read_cell_metadata(data_dir / spec.pheno_filename, spec)
    if counts.shape[0] != len(obs):
        raise ValueError(
            f"{spec.dataset}: gene matrix height {counts.shape[0]} does not match cell metadata rows {len(obs)}."
        )

    guide_targets = _read_guide_targets(data_dir / spec.grna_groups_filename)
    guide_pairs = _read_guide_pair_table(data_dir / spec.gene_grna_pair_filename)

    gene_var = _build_gene_var(gene_ids, guide_pairs)
    gene_adata = ad.AnnData(X=counts, obs=_clean_obs(obs), var=gene_var)
    if gene_id_subset is not None:
        gene_adata = _subset_gene_adata(gene_adata, gene_id_subset)

    guide_var = _build_guide_var(guide_targets, _build_guide_metadata(guide_pairs))
    guide_matrix = _build_guide_matrix(obs["barcode"], guide_var.index)
    guide_adata = ad.AnnData(X=guide_matrix, obs=_clean_obs(obs), var=guide_var)

    guide_by_target = pd.get_dummies(guide_adata.var["intended_target_name"], dtype=bool)
    guide_adata.varm["guide_intended_target_pairs"] = sparse.csr_matrix(guide_by_target.to_numpy(dtype=np.bool_))
    guide_adata.uns["intended_targets"] = guide_by_target.columns.tolist()
    gene_adata.uns["intended_targets"] = guide_by_target.columns.tolist()

    mdata = md.MuData({"gene": gene_adata, "guide": guide_adata})
    mdata.uns["source"] = {
        "accession": "GSE120861",
        "dataset": spec.dataset,
        "prefix": spec.prefix,
    }
    mdata.strings_to_categoricals()
    mdata.update()
    return mdata


def write_gasperini_geo_h5mu(
    data_dir: str | Path,
    dataset: str,
    output_path: str | Path | None = None,
    *,
    gene_id_subset: list[str] | pd.Index | pd.Series | None = None,
) -> Path:
    spec = _get_spec(dataset)
    data_dir = Path(data_dir).expanduser()
    output_path = Path(output_path).expanduser() if output_path is not None else data_dir / spec.output_filename
    mdata = build_gasperini_geo_mudata(data_dir=data_dir, dataset=dataset, gene_id_subset=gene_id_subset)
    mdata.write(output_path)
    return output_path


def plot_gasperini_element_gene_histogram(
    mdata: md.MuData,
    gene_id: str,
    element_name: str,
    *,
    gene_modality: str = "gene",
    guide_modality: str = "guide",
    target_name_key: str = "intended_target_name",
    background_n: int | None = None,
    random_seed: int = 0,
    ax: Any | None = None,
):
    """Plot a per-gene count histogram for guides targeting one element.

    A gray background histogram is drawn from a random sample of cells, then one
    density histogram is overlaid for each guide whose intended target matches
    ``element_name``.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    background_frame, guide_frame = _build_gasperini_element_gene_histogram_frames(
        mdata,
        gene_id,
        element_name,
        gene_modality=gene_modality,
        guide_modality=guide_modality,
        target_name_key=target_name_key,
        background_n=background_n,
        random_seed=random_seed,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    sns.histplot(
        data=background_frame,
        x="count",
        stat="density",
        color="0.75",
        alpha=0.5,
        edgecolor=None,
        label=f"random cells (n={len(background_frame)})",
        ax=ax,
    )
    sns.histplot(
        data=guide_frame,
        x="count",
        hue="guide",
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        ax=ax,
    )
    ax.set_xlabel(f"{gene_id} count")
    ax.set_ylabel("Density")
    ax.set_title(f"{element_name}: {gene_id}")
    return ax


def _build_gasperini_element_gene_histogram_frames(
    mdata: md.MuData,
    gene_id: str,
    element_name: str,
    *,
    gene_modality: str = "gene",
    guide_modality: str = "guide",
    target_name_key: str = "intended_target_name",
    background_n: int | None = None,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_adata = mdata[gene_modality]
    guide_adata = mdata[guide_modality][gene_adata.obs_names, :]

    if gene_id not in gene_adata.var_names:
        raise KeyError(f"Gene ID {gene_id!r} not found in {gene_modality!r} modality.")
    if target_name_key not in guide_adata.var.columns:
        raise KeyError(f"Guide metadata field {target_name_key!r} not found in {guide_modality!r} modality.")

    guide_var = guide_adata.var
    guide_names = guide_var.index[guide_var[target_name_key].astype(str) == str(element_name)].tolist()
    if not guide_names:
        raise ValueError(f"No guides in modality {guide_modality!r} target element {element_name!r}.")

    gene_counts = _extract_dense_column(gene_adata.X, gene_adata.var_names.get_loc(gene_id))

    guide_frames: list[pd.DataFrame] = []
    any_targeting = np.zeros(gene_adata.n_obs, dtype=bool)
    for guide_name in guide_names:
        guide_membership = _extract_dense_column(guide_adata[:, [guide_name]].X, 0) > 0
        any_targeting |= guide_membership
        guide_frames.append(
            pd.DataFrame(
                {
                    "count": gene_counts[guide_membership],
                    "guide": guide_name,
                }
            )
        )

    if not any_targeting.any():
        raise ValueError(f"No cells carry guides targeting element {element_name!r}.")

    if background_n is None:
        background_n = int(any_targeting.sum())
    background_n = min(int(background_n), gene_adata.n_obs)
    if background_n < 1:
        raise ValueError("background_n must be at least 1 after sampling constraints are applied.")

    rng = np.random.default_rng(random_seed)
    background_idx = rng.choice(gene_adata.n_obs, size=background_n, replace=False)
    background_frame = pd.DataFrame({"count": gene_counts[background_idx]})
    guide_frame = pd.concat(guide_frames, ignore_index=True)
    return background_frame, guide_frame


def _get_spec(dataset: str) -> GasperiniGeoSpec:
    try:
        return GASPERINI_GEO_SPECS[dataset]
    except KeyError as exc:
        valid = ", ".join(sorted(GASPERINI_GEO_SPECS))
        raise ValueError(f"Unknown dataset {dataset!r}. Expected one of: {valid}.") from exc


def _extract_dense_column(matrix: Any, column_index: int) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.getcol(column_index).toarray()).ravel()
    return np.asarray(matrix[:, column_index]).reshape(-1)


def _read_counts_matrix(path: Path) -> sparse.csr_matrix:
    with _open_maybe_gzip(path, binary=True) as handle:
        matrix = mmread(handle)
    counts = sparse.csr_matrix(matrix.T)
    return _downcast_sparse_integer_matrix(counts)


def _read_matrix_market_shape(path: Path) -> tuple[int, int]:
    with _open_maybe_gzip(path, binary=False) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            n_rows, n_cols, _ = (int(value) for value in stripped.split())
            return n_rows, n_cols
    raise ValueError(f"Could not read Matrix Market shape from {path}.")


def _read_gene_ids(path: Path) -> pd.DataFrame:
    return pd.read_table(path, header=None, names=["gene_id"]).set_index("gene_id")


def _read_cell_metadata(path: Path, spec: GasperiniGeoSpec) -> pd.DataFrame:
    obs = pd.read_csv(
        path,
        sep=" ",
        header=None,
        names=list(spec.pheno_columns),
        usecols=range(len(spec.pheno_columns)),
        low_memory=False,
    )
    return obs.set_index("cell", drop=True)


def _read_guide_targets(path: Path) -> pd.DataFrame:
    return pd.read_table(path, header=None, names=["intended_target_name", "guide"])


def _read_guide_pair_table(path: Path) -> pd.DataFrame:
    return pd.read_table(path, low_memory=False)


def _build_gene_var(gene_ids: pd.DataFrame, guide_pairs: pd.DataFrame) -> pd.DataFrame:
    group_columns = list(_GENE_METADATA_RENAME)
    if "type.targetgene" in guide_pairs.columns:
        group_columns.append("type.targetgene")

    gene_metadata = guide_pairs[group_columns].rename(columns=_GENE_METADATA_RENAME)
    if "type.targetgene" in gene_metadata.columns:
        gene_metadata = gene_metadata.rename(columns={"type.targetgene": "gene_type"})

    _assert_single_value_per_gene(gene_metadata, ["gene_chr", "symbol", "gene_strand"])
    if "gene_type" in gene_metadata.columns:
        _assert_single_value_per_gene(gene_metadata, ["gene_type"])

    aggregated = (
        gene_metadata.groupby("gene_id", sort=False)
        .agg(
            {
                "gene_start": "min",
                "gene_end": "max",
                "gene_chr": "first",
                "symbol": "first",
                "gene_strand": "first",
                **({"gene_type": "first"} if "gene_type" in gene_metadata.columns else {}),
            }
        )
    )
    return gene_ids.join(aggregated, how="left")


def _build_guide_metadata(guide_pairs: pd.DataFrame) -> pd.DataFrame:
    guide_metadata = guide_pairs[list(_GUIDE_METADATA_RENAME)].rename(columns=_GUIDE_METADATA_RENAME)
    guide_metadata = guide_metadata.replace("NTC", np.nan)
    guide_metadata["intended_target_start"] = pd.to_numeric(
        guide_metadata["intended_target_start"], errors="coerce"
    )
    guide_metadata["intended_target_end"] = pd.to_numeric(
        guide_metadata["intended_target_end"], errors="coerce"
    )

    return (
        guide_metadata.groupby("intended_target_name", sort=False)
        .agg(
            {
                "intended_target_start": "min",
                "intended_target_end": "max",
                "intended_target_chr": "first",
                "guide_group": "first",
            }
        )
        .reset_index()
    )


def _build_guide_var(guide_targets: pd.DataFrame, guide_metadata: pd.DataFrame) -> pd.DataFrame:
    guide_var = guide_targets.merge(guide_metadata, on="intended_target_name", how="left")
    guide_var["is_non_targeting"] = (
        guide_var["guide_group"].fillna("").eq("NTC")
        | guide_var["intended_target_name"].str.contains("random|scrambled|bassik", case=False, na=False)
    )
    return guide_var.set_index("guide", drop=True)


def _clean_obs(obs: pd.DataFrame) -> pd.DataFrame:
    existing_drop_columns = [column for column in _CELL_METADATA_DROP_COLUMNS if column in obs.columns]
    return obs.drop(columns=existing_drop_columns)


def _build_guide_matrix(barcode_series: pd.Series, guide_index: pd.Index) -> sparse.csr_matrix:
    guide_to_col = {guide: index for index, guide in enumerate(guide_index)}
    row_indices: list[int] = []
    col_indices: list[int] = []
    unknown_guides: set[str] = set()

    for row_index, barcode_value in enumerate(barcode_series.fillna("")):
        if not barcode_value:
            continue
        for guide in str(barcode_value).split("_"):
            if not guide or guide == "NA":
                continue
            column_index = guide_to_col.get(guide)
            if column_index is None:
                unknown_guides.add(guide)
                continue
            row_indices.append(row_index)
            col_indices.append(column_index)

    if unknown_guides:
        examples = ", ".join(sorted(unknown_guides)[:10])
        raise ValueError(f"Encountered barcode guides absent from the guide table: {examples}")

    matrix = sparse.coo_matrix(
        (np.ones(len(row_indices), dtype=np.float32), (row_indices, col_indices)),
        shape=(len(barcode_series), len(guide_index)),
        dtype=np.float32,
    ).tocsr()
    if matrix.nnz:
        matrix.data[:] = 1.0
    return matrix


def _subset_gene_adata(
    gene_adata: ad.AnnData,
    gene_id_subset: list[str] | pd.Index | pd.Series,
) -> ad.AnnData:
    ordered_gene_ids = [str(gene_id) for gene_id in gene_id_subset]
    gene_index = pd.Index(gene_adata.var_names.astype(str))
    shared_gene_ids = [gene_id for gene_id in ordered_gene_ids if gene_id in gene_index]
    if not shared_gene_ids:
        raise ValueError("Gene subset does not overlap the dataset gene IDs.")
    return gene_adata[:, shared_gene_ids].copy()


def _find_unknown_barcode_guides(barcode_series: pd.Series, guide_index: pd.Index) -> list[str]:
    guide_set = set(guide_index)
    unknown: set[str] = set()
    for barcode_value in barcode_series.fillna(""):
        if not barcode_value:
            continue
        for guide in str(barcode_value).split("_"):
            if guide and guide != "NA" and guide not in guide_set:
                unknown.add(guide)
    return sorted(unknown)


def _downcast_sparse_integer_matrix(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    if matrix.nnz == 0:
        return matrix.astype(np.uint8)

    data = matrix.data
    if np.allclose(data, np.round(data)) and data.min() >= 0:
        max_value = int(data.max())
        if max_value <= np.iinfo(np.uint8).max:
            return matrix.astype(np.uint8)
        if max_value <= np.iinfo(np.uint16).max:
            return matrix.astype(np.uint16)
        if max_value <= np.iinfo(np.uint32).max:
            return matrix.astype(np.uint32)
    return matrix.astype(np.float32)


def _assert_single_value_per_gene(gene_metadata: pd.DataFrame, columns: list[str]) -> None:
    grouped = gene_metadata.groupby("gene_id", sort=False)[columns].nunique(dropna=False)
    inconsistent_columns = [column for column in columns if (grouped[column] > 1).any()]
    if inconsistent_columns:
        joined = ", ".join(inconsistent_columns)
        raise ValueError(f"Gene metadata is not uniquely defined per gene_id for columns: {joined}")


def _is_gzip_path(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def _open_maybe_gzip(path: Path, *, binary: bool):
    if _is_gzip_path(path):
        import gzip

        mode = "rb" if binary else "rt"
        return gzip.open(path, mode)
    mode = "rb" if binary else "r"
    return path.open(mode)
