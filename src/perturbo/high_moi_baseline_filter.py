"""Shared hard-gene baseline filtering for high-MOI datasets."""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp

from perturbo.simulation.benchmark_gene_panel import load_gene_panel_ids, resolve_gene_panel


HIGH_MOI_BASELINE_FILTER_ALL = "all"
HIGH_MOI_BASELINE_FILTER_HARD_TSS = "exclude_hard_tss"
HIGH_MOI_BASELINE_FILTER_HARD_TSS_AND_CIS = "exclude_hard_tss_and_cis"
HIGH_MOI_CIS_WINDOW_BP = 500_000


@dataclass(frozen=True)
class HighMOIBaselineFilterResult:
    eligible_mask: np.ndarray
    baseline_description: str
    notes: list[str]
    hard_gene_table: pd.DataFrame | None = None


def _resolve_hard_gene_table(
    adata: ad.AnnData,
    *,
    gene_name_var_key: str | None,
    hard_gene_panel_path: str | None,
) -> pd.DataFrame:
    if hard_gene_panel_path in (None, ""):
        raise ValueError("No hard-gene panel is configured for this high-MOI baseline filter.")
    requested_ids = load_gene_panel_ids(hard_gene_panel_path)
    _, subset_gene_table = resolve_gene_panel(
        adata,
        requested_ids=requested_ids,
        gene_name_key=gene_name_var_key,
    )
    return subset_gene_table


def _normalize_contig(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return text if text.startswith("chr") else f"chr{text}"


def _excluded_cell_mask_from_guides(guide_adata: ad.AnnData, excluded_guides: np.ndarray) -> np.ndarray:
    if excluded_guides.ndim != 1 or excluded_guides.shape[0] != guide_adata.n_vars:
        raise ValueError("excluded_guides must be a 1D boolean mask with one entry per guide.")
    if not bool(excluded_guides.any()):
        return np.zeros((guide_adata.n_obs,), dtype=bool)

    x = guide_adata.X
    if sp.issparse(x):
        guide_counts = x[:, excluded_guides].sum(axis=1)
        if hasattr(guide_counts, "A1"):
            guide_counts = guide_counts.A1
        else:
            guide_counts = np.asarray(guide_counts).reshape(-1)
    else:
        guide_counts = np.asarray(x[:, excluded_guides]).sum(axis=1)
    return np.asarray(guide_counts).reshape(-1) > 0


def _compute_structured_cis_guide_mask(
    rna_adata: ad.AnnData,
    guide_adata: ad.AnnData,
    hard_gene_table: pd.DataFrame,
) -> np.ndarray:
    required_rna_cols = ("contig", "start", "end")
    required_guide_cols = ("chr.candidate_enhancer", "start.candidate_enhancer", "stop.candidate_enhancer")
    missing_rna = [col for col in required_rna_cols if col not in rna_adata.var.columns]
    missing_guide = [col for col in required_guide_cols if col not in guide_adata.var.columns]
    if missing_rna or missing_guide:
        missing = ", ".join(missing_rna + missing_guide)
        raise ValueError(f"Nearby-element exclusion requires coordinate metadata; missing columns: {missing}")

    gene_meta = rna_adata.var.loc[hard_gene_table["matched_var_name"], list(required_rna_cols)].copy()
    gene_meta["contig"] = gene_meta["contig"].map(_normalize_contig)
    gene_meta["start"] = pd.to_numeric(gene_meta["start"], errors="coerce")
    gene_meta["end"] = pd.to_numeric(gene_meta["end"], errors="coerce")
    gene_meta = gene_meta.dropna(subset=["contig", "start", "end"])
    if gene_meta.empty:
        raise ValueError("Nearby-element exclusion could not resolve genomic coordinates for the hard-gene panel.")

    guide_chr = guide_adata.var["chr.candidate_enhancer"].map(_normalize_contig)
    guide_start = pd.to_numeric(guide_adata.var["start.candidate_enhancer"], errors="coerce")
    guide_stop = pd.to_numeric(guide_adata.var["stop.candidate_enhancer"], errors="coerce")
    guide_chr_arr = guide_chr.fillna("").to_numpy(dtype=str)
    guide_start_arr = guide_start.to_numpy(dtype=float)
    guide_stop_arr = guide_stop.to_numpy(dtype=float)
    valid_guides = guide_chr.notna().to_numpy() & np.isfinite(guide_start_arr) & np.isfinite(guide_stop_arr)

    cis_mask = np.zeros((guide_adata.n_vars,), dtype=bool)
    for contig, start, end in gene_meta.itertuples(index=False):
        window_start = max(int(start) - HIGH_MOI_CIS_WINDOW_BP, 0)
        window_end = int(end) + HIGH_MOI_CIS_WINDOW_BP
        cis_mask |= (
            valid_guides
            & (guide_chr_arr == str(contig))
            & (guide_stop_arr >= window_start)
            & (guide_start_arr <= window_end)
        )
    return cis_mask


def _compute_high_moi_baseline_excluded_guides(
    rna_adata: ad.AnnData,
    guide_adata: ad.AnnData,
    *,
    gene_name_var_key: str | None,
    hard_gene_panel_path: str | None,
    profile_name: str,
    baseline_filter_mode: str,
) -> tuple[np.ndarray, str, list[str], pd.DataFrame]:
    hard_gene_table = _resolve_hard_gene_table(
        rna_adata,
        gene_name_var_key=gene_name_var_key,
        hard_gene_panel_path=hard_gene_panel_path,
    )
    hard_gene_symbols = pd.Index(hard_gene_table["matched_gene_name"].astype(str)).drop_duplicates()
    if len(hard_gene_symbols) == 0:
        raise ValueError("High-MOI baseline filter resolved zero hard-gene symbols.")

    notes = [f"High-MOI baseline filter resolved {len(hard_gene_symbols):,} hard-gene symbols from the configured panel."]
    excluded_guides = np.zeros((guide_adata.n_vars,), dtype=bool)
    effective_description = "all sampled cells"

    structured_tss = "Category" in guide_adata.var.columns and "Target_Site" in guide_adata.var.columns
    if structured_tss and profile_name != "gasperini_pilot_high_moi":
        category = guide_adata.var["Category"].astype(str).to_numpy()
        target_site = guide_adata.var["Target_Site"].astype(str).to_numpy()
        tss_guides = (category == "TSS") & np.isin(target_site, hard_gene_symbols.to_numpy())
        excluded_guides |= tss_guides
        if baseline_filter_mode == HIGH_MOI_BASELINE_FILTER_HARD_TSS_AND_CIS:
            excluded_guides |= _compute_structured_cis_guide_mask(rna_adata, guide_adata, hard_gene_table)
            effective_description = "sampled cells excluding hard-gene TSS-targeted cells and nearby elements (+/-500 kb)"
        else:
            effective_description = "sampled cells excluding hard-gene TSS-targeted cells"
        return excluded_guides, effective_description, notes, hard_gene_table

    guide_names = guide_adata.var_names.astype(str).to_numpy()
    guide_stems = np.asarray([name.split("|", 1)[0] for name in guide_names], dtype=object)
    tss_like = np.asarray([stem.endswith("_TSS") for stem in guide_stems], dtype=bool)
    guide_targets = np.asarray([stem.removesuffix("_TSS") for stem in guide_stems], dtype=object)
    excluded_guides |= tss_like & np.isin(guide_targets, hard_gene_symbols.to_numpy())
    if baseline_filter_mode == HIGH_MOI_BASELINE_FILTER_HARD_TSS_AND_CIS:
        notes.append(
            "Nearby-element exclusion is unavailable for this guide metadata layout; using hard-gene TSS-only exclusion."
        )
    return excluded_guides, "sampled cells excluding hard-gene TSS-targeted cells", notes, hard_gene_table


def apply_high_moi_baseline_filter(
    rna_adata: ad.AnnData,
    guide_adata: ad.AnnData,
    *,
    gene_name_var_key: str | None,
    hard_gene_panel_path: str | None,
    profile_name: str,
    baseline_filter_mode: str,
) -> HighMOIBaselineFilterResult:
    if baseline_filter_mode == HIGH_MOI_BASELINE_FILTER_ALL:
        return HighMOIBaselineFilterResult(
            eligible_mask=np.ones((rna_adata.n_obs,), dtype=bool),
            baseline_description="all sampled cells",
            notes=[],
            hard_gene_table=None,
        )

    if baseline_filter_mode not in {
        HIGH_MOI_BASELINE_FILTER_HARD_TSS,
        HIGH_MOI_BASELINE_FILTER_HARD_TSS_AND_CIS,
    }:
        raise ValueError(f"Unsupported high-MOI baseline filter mode '{baseline_filter_mode}'.")

    excluded_guides, baseline_description, notes, hard_gene_table = _compute_high_moi_baseline_excluded_guides(
        rna_adata,
        guide_adata,
        gene_name_var_key=gene_name_var_key,
        hard_gene_panel_path=hard_gene_panel_path,
        profile_name=profile_name,
        baseline_filter_mode=baseline_filter_mode,
    )
    eligible_mask = ~_excluded_cell_mask_from_guides(guide_adata, excluded_guides)
    return HighMOIBaselineFilterResult(
        eligible_mask=eligible_mask,
        baseline_description=baseline_description,
        notes=notes,
        hard_gene_table=hard_gene_table,
    )
