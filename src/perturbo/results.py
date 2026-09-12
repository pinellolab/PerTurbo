"""Posterior accessors and summary tables for perturbo fits."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from math import erf
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ._statistics import (
    benjamini_hochberg_over_finite,
    empirical_pvals_from_tnull_fixed0,
    z_to_two_sided_pvalues,
)


@dataclass(frozen=True)
class PosteriorParameter:
    name: str
    value: np.ndarray
    loc: np.ndarray | None = None
    scale: np.ndarray | None = None


class PosteriorMedians(Mapping[str, np.ndarray]):
    """Mapping-like posterior median accessor."""

    def __init__(self, values: dict[str, np.ndarray]):
        self._values = {key: np.asarray(value) for key, value in values.items()}

    def __getitem__(self, key: str) -> np.ndarray:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def to_dict(self) -> dict[str, np.ndarray]:
        return {key: np.asarray(value) for key, value in self._values.items()}


def build_element_effects_df(
    *,
    effect_loc: np.ndarray,
    effect_scale: np.ndarray,
    element_names: list[str],
    gene_names: list[str],
) -> pd.DataFrame:
    loc = np.asarray(effect_loc, dtype=float)
    scale = np.clip(np.asarray(effect_scale, dtype=float), a_min=1e-6, a_max=None)
    if loc.shape != scale.shape:
        raise ValueError(
            f"effect_loc/effect_scale shape mismatch: {loc.shape} vs {scale.shape}"
        )
    if loc.shape != (len(element_names), len(gene_names)):
        raise ValueError(
            "effect arrays must match (n_elements, n_genes); "
            f"got {loc.shape} for {len(element_names)} elements and {len(gene_names)} genes."
        )
    z = loc / scale
    prob = 2.0 * (1.0 - 0.5 * (1.0 + np.vectorize(erf)(np.abs(z) / np.sqrt(2.0))))
    return pd.DataFrame(
        {
            "element": np.repeat(element_names, len(gene_names)),
            "gene": np.tile(gene_names, len(element_names)),
            "loc": loc.reshape(-1),
            "scale": scale.reshape(-1),
            "z_value": z.reshape(-1),
            "q_value": prob.reshape(-1),
        }
    )


def build_guide_effects_df(
    *,
    effect_loc: np.ndarray,
    effect_scale: np.ndarray,
    guide_names: list[str],
    guide_parent_elements: list[str],
    gene_names: list[str],
) -> pd.DataFrame:
    loc = np.asarray(effect_loc, dtype=float)
    scale = np.clip(np.asarray(effect_scale, dtype=float), a_min=1e-6, a_max=None)
    if loc.shape != scale.shape:
        raise ValueError(
            f"effect_loc/effect_scale shape mismatch: {loc.shape} vs {scale.shape}"
        )
    if len(guide_names) != len(guide_parent_elements):
        raise ValueError("guide_names and guide_parent_elements must have the same length.")
    if loc.shape != (len(guide_names), len(gene_names)):
        raise ValueError(
            "guide effect arrays must match (n_guides, n_genes); "
            f"got {loc.shape} for {len(guide_names)} guides and {len(gene_names)} genes."
        )
    z = loc / scale
    prob = 2.0 * (1.0 - 0.5 * (1.0 + np.vectorize(erf)(np.abs(z) / np.sqrt(2.0))))
    return pd.DataFrame(
        {
            "guide": np.repeat(guide_names, len(gene_names)),
            "element": np.repeat(guide_parent_elements, len(gene_names)),
            "gene": np.tile(gene_names, len(guide_names)),
            "loc": loc.reshape(-1),
            "scale": scale.reshape(-1),
            "z_value": z.reshape(-1),
            "q_value": prob.reshape(-1),
        }
    )


def load_pairs_to_test(path) -> pd.DataFrame:
    """Read a two-column ``element,gene`` restriction table.

    The contract is the one the IGVF CRISPR pipeline's PerTurbo adapter already
    writes: a CSV, TSV or Parquet file with columns ``element`` and ``gene``,
    duplicates dropped. It is deliberately unchanged so that nothing upstream
    has to be edited.
    """
    from pathlib import Path

    pair_path = Path(path)
    if not pair_path.exists():
        raise FileNotFoundError(f"pairs-to-test file not found: {pair_path}")
    if pair_path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(pair_path)
    else:
        frame = pd.read_csv(pair_path, sep=None, engine="python")
    missing = {"element", "gene"}.difference(frame.columns)
    if missing:
        raise ValueError(f"pairs-to-test requires columns named 'element' and 'gene'; missing {sorted(missing)}.")
    if frame[["element", "gene"]].isna().any().any():
        raise ValueError("pairs-to-test contains missing element or gene names.")
    frame = frame.loc[:, ["element", "gene"]].astype(str).drop_duplicates(ignore_index=True)
    if frame.empty:
        raise ValueError("pairs-to-test must contain at least one pair.")
    return frame


def restrict_effects_to_pairs(effects: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``effects`` named by ``pairs``, with every q-value recorrected.

    A restricted table is not a filtered copy of the transcriptome-wide one: the
    Benjamini-Hochberg family is the restricted set, so the same pair carries a
    different q-value in each. Everything else, including the effect estimates
    and the p-values, is identical, because one fit and one test produced both.
    That is the whole point of emitting the two tables from a single run rather
    than running the analysis twice over different pair sets.
    """
    for column in ("element", "gene"):
        if column not in effects.columns:
            raise ValueError(f"effects table is missing the {column!r} column.")
    wanted = pd.MultiIndex.from_frame(pairs[["element", "gene"]].astype(str))
    keys = pd.MultiIndex.from_frame(effects[["element", "gene"]].astype(str))
    keep = keys.isin(wanted)
    restricted = effects.loc[keep].reset_index(drop=True)
    for column in restricted.columns:
        if column == "q_value" or column.endswith("_q_value"):
            source = column[:-len("_q_value")] + "_p_value" if column.endswith("_q_value") else "p_value"
            if source in restricted.columns:
                restricted[column] = benjamini_hochberg_over_finite(restricted[source].to_numpy(dtype=float))
    return restricted


def build_standard_element_effects_df(
    *,
    method: str,
    effect_loc: np.ndarray,
    effect_scale: np.ndarray,
    element_names: list[str],
    gene_names: list[str],
    null_z_values: np.ndarray | None = None,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Element-by-gene effect table.

    ``extra_columns`` appends further ``(n_elements, n_genes)`` matrices under
    their own names, flattened in the same row-major order as the built-in
    columns. It exists so an additional test computed over the same grid - the
    CRT, in particular - can travel in the same table without this builder
    having to know what such a test is.
    """

    return pd.concat(
        iter_standard_element_effects_frames(
            method=method,
            effect_loc=effect_loc,
            effect_scale=effect_scale,
            element_names=element_names,
            gene_names=gene_names,
            null_z_values=null_z_values,
            extra_columns=extra_columns,
            row_block_size=max(np.asarray(effect_loc).size, 1),
        ),
        ignore_index=True,
    )


def iter_standard_element_effects_frames(
    *,
    method: str,
    effect_loc: np.ndarray,
    effect_scale: np.ndarray,
    element_names: list[str],
    gene_names: list[str],
    null_z_values: np.ndarray | None = None,
    extra_columns: dict[str, np.ndarray] | None = None,
    row_block_size: int = 250_000,
) -> Iterator[pd.DataFrame]:
    """Yield the standard table in exact row-major blocks.

    Blocking avoids the several full-grid float64 temporaries and Python object
    columns that a transcriptome-wide long DataFrame otherwise holds at once.
    """

    loc = np.asarray(effect_loc)
    scale = np.asarray(effect_scale)
    if loc.shape != scale.shape:
        raise ValueError(f"effect_loc/effect_scale shape mismatch: {loc.shape} vs {scale.shape}")
    if loc.shape != (len(element_names), len(gene_names)):
        raise ValueError(
            "effect arrays must match (n_elements, n_genes); "
            f"got {loc.shape} for {len(element_names)} elements and {len(gene_names)} genes."
        )

    if row_block_size < 1:
        raise ValueError("row_block_size must be positive.")
    null_params = None
    if null_z_values is not None:
        null_arr = np.asarray(null_z_values, dtype=float).reshape(-1)
        null_arr = null_arr[np.isfinite(null_arr)]
        if null_arr.size:
            _, null_params = empirical_pvals_from_tnull_fixed0(
                null_arr, np.empty(0, dtype=float), return_params=True
            )
    extras = {}
    for name, values in (extra_columns or {}).items():
        array = np.asarray(values)
        if array.shape != loc.shape:
            raise ValueError(
                f"extra column {name!r} must match (n_elements, n_genes) {loc.shape}; got {array.shape}."
            )
        extras[name] = array

    flat_loc = loc.reshape(-1)
    flat_scale = scale.reshape(-1)
    elements = np.asarray(element_names, dtype=str)
    genes = np.asarray(gene_names, dtype=str)
    num_genes = len(gene_names)
    if flat_loc.size == 0:
        frame = pd.DataFrame({
            "method": pd.Series(dtype=str),
            "element": pd.Series(dtype=str),
            "gene": pd.Series(dtype=str),
            "posterior_mean": pd.Series(dtype=np.float32),
            "posterior_scale": pd.Series(dtype=np.float32),
            "z_value": pd.Series(dtype=np.float32),
            "posterior_prob": pd.Series(dtype=np.float32),
            "empirical_p_value": pd.Series(dtype=np.float32),
        })
        for name in extras:
            frame[name] = pd.Series(dtype=np.float64)
        yield frame
        return
    for start in range(0, flat_loc.size, row_block_size):
        stop = min(start + row_block_size, flat_loc.size)
        positions = np.arange(start, stop)
        element_index = positions // num_genes
        gene_index = positions % num_genes
        block_loc = flat_loc[start:stop].astype(float, copy=False)
        block_scale = np.clip(flat_scale[start:stop].astype(float, copy=False), 1e-6, None)
        z = block_loc / block_scale
        if null_params is None:
            empirical = np.full(z.size, np.nan, dtype=float)
        else:
            empirical = np.clip(
                2.0 * stats.t.sf(np.abs(z / null_params["scale"]), null_params["df"]),
                0.0,
                1.0,
            )
        frame = pd.DataFrame({
            "method": method,
            "element": elements[element_index],
            "gene": genes[gene_index],
            "posterior_mean": block_loc.astype(np.float32, copy=False),
            "posterior_scale": block_scale.astype(np.float32, copy=False),
            "z_value": z.astype(np.float32, copy=False),
            "posterior_prob": z_to_two_sided_pvalues(z).astype(np.float32, copy=False),
            "empirical_p_value": empirical.astype(np.float32, copy=False),
        })
        for name, values in extras.items():
            # Tail columns stay float64 to preserve probabilities below float32 range.
            frame[name] = np.asarray(values[element_index, gene_index], dtype=np.float64)
        yield frame


def write_standard_element_effects_parquet(
    path: str | Path,
    *,
    requested_pairs: pd.DataFrame | None = None,
    **kwargs,
) -> pd.DataFrame | None:
    """Write row-major blocks and optionally retain only requested rows."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    selected: list[pd.DataFrame] = []
    empty: pd.DataFrame | None = None
    wanted = (
        None
        if requested_pairs is None
        else pd.MultiIndex.from_frame(requested_pairs[["element", "gene"]].astype(str))
    )
    try:
        for frame in iter_standard_element_effects_frames(**kwargs):
            if empty is None:
                empty = frame.iloc[:0].copy()
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
            if requested_pairs is not None:
                keys = pd.MultiIndex.from_frame(frame[["element", "gene"]].astype(str))
                if np.any(keep := keys.isin(wanted)):
                    selected.append(frame.loc[keep].copy())
    finally:
        if writer is not None:
            writer.close()
    if requested_pairs is None:
        return None
    if not selected:
        if empty is None:
            raise ValueError("The element-by-gene grid must not be empty.")
        return empty
    return restrict_effects_to_pairs(pd.concat(selected, ignore_index=True), requested_pairs)


def build_guide_efficiency_df(
    *,
    method: str,
    guide_names: list[str],
    guide_parent_elements: list[str],
    guide_efficiency_mean: np.ndarray,
    gene_names: list[str],
) -> pd.DataFrame:
    eff = np.asarray(guide_efficiency_mean, dtype=float)
    if eff.ndim != 2:
        raise ValueError(f"guide_efficiency_mean must be 2D (n_guides × n_genes); got shape {eff.shape}.")
    n_guides, n_genes = eff.shape
    if len(guide_names) != n_guides:
        raise ValueError(
            f"guide_names length ({len(guide_names)}) does not match guide_efficiency rows ({n_guides})."
        )
    if len(guide_parent_elements) != n_guides:
        raise ValueError("guide_names and guide_parent_elements must have same length.")
    if len(gene_names) != n_genes:
        raise ValueError(
            f"gene_names length ({len(gene_names)}) does not match guide_efficiency columns ({n_genes})."
        )
    guide_col = np.repeat(np.asarray(guide_names, dtype=str), n_genes)
    element_col = np.repeat(np.asarray(guide_parent_elements, dtype=str), n_genes)
    gene_col = np.tile(np.asarray(gene_names, dtype=str), n_guides)
    return pd.DataFrame(
        {
            "method": method,
            "guide": guide_col,
            "element": element_col,
            "gene": gene_col,
            "posterior_mean": eff.ravel().astype(np.float32),
        }
    )


def extract_parameter_table(parameters: dict[str, np.ndarray]) -> dict[str, PosteriorParameter]:
    out: dict[str, PosteriorParameter] = {}
    for name, value in parameters.items():
        arr = np.asarray(value)
        if name.endswith("_loc"):
            base = name[: -len("_loc")]
            param = out.get(base)
            out[base] = PosteriorParameter(
                name=base,
                value=param.value if param is not None else arr,
                loc=arr,
                scale=param.scale if param is not None else None,
            )
        elif name.endswith("_scale"):
            base = name[: -len("_scale")]
            param = out.get(base)
            out[base] = PosteriorParameter(
                name=base,
                value=param.value if param is not None else arr,
                loc=param.loc if param is not None else None,
                scale=arr,
            )
        else:
            param = out.get(name)
            out[name] = PosteriorParameter(
                name=name,
                value=arr,
                loc=param.loc if param is not None else None,
                scale=param.scale if param is not None else None,
            )
    return out


__all__ = [
    "PosteriorMedians",
    "PosteriorParameter",
    "build_element_effects_df",
    "build_standard_element_effects_df",
    "iter_standard_element_effects_frames",
    "write_standard_element_effects_parquet",
    "build_guide_efficiency_df",
    "build_guide_effects_df",
    "extract_parameter_table",
]
