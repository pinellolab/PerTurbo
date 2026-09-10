"""Posterior accessors and summary tables for perturbo fits."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from math import erf

import numpy as np
import pandas as pd

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
    wanted = set(zip(pairs["element"].astype(str), pairs["gene"].astype(str)))
    keys = list(zip(effects["element"].astype(str), effects["gene"].astype(str)))
    keep = np.fromiter((key in wanted for key in keys), dtype=bool, count=len(keys))
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

    loc = np.asarray(effect_loc, dtype=float)
    scale = np.clip(np.asarray(effect_scale, dtype=float), a_min=1e-6, a_max=None)
    if loc.shape != scale.shape:
        raise ValueError(f"effect_loc/effect_scale shape mismatch: {loc.shape} vs {scale.shape}")
    if loc.shape != (len(element_names), len(gene_names)):
        raise ValueError(
            "effect arrays must match (n_elements, n_genes); "
            f"got {loc.shape} for {len(element_names)} elements and {len(gene_names)} genes."
        )

    z = loc / scale
    posterior_prob = z_to_two_sided_pvalues(z.reshape(-1))
    empirical_p_value: np.ndarray
    if null_z_values is None:
        empirical_p_value = np.full(z.size, np.nan, dtype=float)
    else:
        null_arr = np.asarray(null_z_values, dtype=float).reshape(-1)
        null_arr = null_arr[np.isfinite(null_arr)]
        if null_arr.size == 0:
            empirical_p_value = np.full(z.size, np.nan, dtype=float)
        else:
            empirical_p_value = np.asarray(
                empirical_pvals_from_tnull_fixed0(null_arr, z.reshape(-1), return_params=False),
                dtype=float,
            )

    frame = pd.DataFrame(
        {
            "method": method,
            "element": np.repeat(np.asarray(element_names, dtype=str), len(gene_names)),
            "gene": np.tile(np.asarray(gene_names, dtype=str), len(element_names)),
            "posterior_mean": loc.reshape(-1).astype(np.float32, copy=False),
            "posterior_scale": scale.reshape(-1).astype(np.float32, copy=False),
            "z_value": z.reshape(-1).astype(np.float32, copy=False),
            "posterior_prob": posterior_prob.astype(np.float32, copy=False),
            "empirical_p_value": empirical_p_value.astype(np.float32, copy=False),
        }
    )
    for name, values in (extra_columns or {}).items():
        array = np.asarray(values, dtype=float)
        if array.shape != loc.shape:
            raise ValueError(
                f"extra column {name!r} must match (n_elements, n_genes) {loc.shape}; got {array.shape}."
            )
        # Kept in float64, unlike the columns above. A parametric tail p-value
        # exists precisely to resolve below the empirical floor, and float32
        # flushes anything under ~1e-45 to zero - which would silently discard
        # the resolution these columns are carried for.
        frame[name] = array.reshape(-1)
    return frame


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
    "build_guide_efficiency_df",
    "build_guide_effects_df",
    "extract_parameter_table",
]
