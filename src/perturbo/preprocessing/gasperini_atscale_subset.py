from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
from scipy import sparse

from perturbo.preprocessing.counts import to_dense_array


@dataclass(frozen=True)
class GasperiniAtScaleSubsetSummary:
    cells_before: int
    cells_after: int
    genes_before: int
    genes_after: int
    guides_before: int
    guides_after: int
    tss_guides: int
    control_guides: int
    targeted_genes: int


def build_gasperini_atscale_tss_subset(
    mdata: md.MuData,
    *,
    gene_modality: str = "gene",
    guide_modality: str = "guide",
    guide_by_element_key: str = "guide_intended_target_pairs",
    guide_element_uns_key: str = "intended_targets",
) -> tuple[md.MuData, GasperiniAtScaleSubsetSummary]:
    gene = mdata[gene_modality]
    guide = mdata[guide_modality]

    target_names = _guide_target_names(
        guide,
        guide_by_element_key=guide_by_element_key,
        guide_element_uns_key=guide_element_uns_key,
    )
    target_names_index = pd.Index(target_names)
    target_names_lower = target_names_index.str.lower()
    keep_target_mask = (
        target_names_lower.str.endswith("_tss")
        | target_names_lower.str.startswith("random")
        | target_names_lower.str.startswith("scrambled")
    )
    if not bool(np.any(keep_target_mask)):
        raise ValueError("No targets matched the requested TSS/control prefixes.")

    selected_target_names = target_names_index[keep_target_mask].tolist()
    tss_target_names = [name for name in selected_target_names if str(name).lower().endswith("_tss")]
    if not tss_target_names:
        raise ValueError("No TSS targets were found among the selected guide targets.")

    gene_symbols = _gene_symbols(gene)
    tss_gene_symbols = _gene_symbols_from_tss_targets(tss_target_names)
    keep_gene_mask = np.asarray(gene_symbols.isin(tss_gene_symbols), dtype=bool)
    if not bool(np.any(keep_gene_mask)):
        raise ValueError("No genes targeted by the TSS guides were found in the gene modality.")

    mapping = to_dense_array(guide.varm[guide_by_element_key]).astype(bool, copy=False)
    guide_keep_mask = _guide_row_keep_mask(
        mapping,
        target_names=target_names,
        selected_target_names=selected_target_names,
    )
    if not bool(np.any(guide_keep_mask)):
        raise ValueError("No guide barcodes matched the selected TSS/control targets.")

    guide_subset = guide[:, guide_keep_mask].copy()
    gene_subset = gene[:, keep_gene_mask].copy()

    selected_target_mask = np.asarray(keep_target_mask, dtype=bool)
    mapping_subset = mapping[guide_keep_mask][:, selected_target_mask]
    guide_subset.varm[guide_by_element_key] = sparse.csr_matrix(mapping_subset)
    guide_subset.uns[guide_element_uns_key] = selected_target_names
    gene_subset.uns[guide_element_uns_key] = tss_gene_symbols

    cell_keep_mask = np.asarray(guide_subset.X.sum(axis=1)).reshape(-1) > 0
    if not bool(np.any(cell_keep_mask)):
        raise ValueError("No cells contain the retained guides.")

    guide_subset = guide_subset[cell_keep_mask, :].copy()
    gene_subset = gene_subset[cell_keep_mask, :].copy()

    subset = md.MuData({gene_modality: gene_subset, guide_modality: guide_subset})
    subset.uns["source"] = dict(mdata.uns.get("source", {}))
    subset.uns["source"]["subset"] = "gasperini_atscale_tss_controls"
    subset.uns["selection"] = {
        "guide_prefixes": ["random", "scrambled"],
        "guide_suffix": "_TSS",
        "guide_by_element_key": guide_by_element_key,
        "guide_element_uns_key": guide_element_uns_key,
        "target_names": selected_target_names,
        "target_gene_symbols": tss_gene_symbols,
    }
    subset.uns["selection_summary"] = asdict(
        GasperiniAtScaleSubsetSummary(
            cells_before=int(mdata.n_obs),
            cells_after=int(subset.n_obs),
            genes_before=int(gene.n_vars),
            genes_after=int(gene_subset.n_vars),
            guides_before=int(guide.n_vars),
            guides_after=int(guide_subset.n_vars),
            tss_guides=int(sum(name.lower().endswith("_tss") for name in selected_target_names)),
            control_guides=int(
                sum(
                    name.lower().startswith("random") or name.lower().startswith("scrambled")
                    for name in selected_target_names
                )
            ),
            targeted_genes=int(len(tss_gene_symbols)),
        )
    )
    subset.strings_to_categoricals()
    subset.update()
    summary = GasperiniAtScaleSubsetSummary(**subset.uns["selection_summary"])
    return subset, summary


def write_gasperini_atscale_tss_subset_h5mu(
    input_path: str | Path,
    output_path: str | Path,
    *,
    gene_modality: str = "gene",
    guide_modality: str = "guide",
    guide_by_element_key: str = "guide_intended_target_pairs",
    guide_element_uns_key: str = "intended_targets",
) -> tuple[Path, GasperiniAtScaleSubsetSummary]:
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()

    mdata = md.read(input_path)
    subset, summary = build_gasperini_atscale_tss_subset(
        mdata,
        gene_modality=gene_modality,
        guide_modality=guide_modality,
        guide_by_element_key=guide_by_element_key,
        guide_element_uns_key=guide_element_uns_key,
    )
    subset.write(output_path)
    return output_path, summary


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Subset the Gasperini at-scale MuData object to TSS-targeting guides plus random/scrambled controls, "
            "and retain only genes targeted by the TSS guides."
        )
    )
    parser.add_argument(
        "--input",
        default="~/Data/gasperini_atscale/GSE120861_at_scale_screen.h5mu",
        help="Input Gasperini at-scale .h5mu file.",
    )
    parser.add_argument(
        "--output",
        default="~/Data/gasperini_atscale/GSE120861_at_scale_screen.tss_controls.h5mu",
        help="Output path for the subset .h5mu file.",
    )
    parser.add_argument("--gene-modality", default="gene", help="RNA modality key.")
    parser.add_argument("--guide-modality", default="guide", help="Guide modality key.")
    parser.add_argument(
        "--guide-by-element-key",
        default="guide_intended_target_pairs",
        help="Guide-to-target mapping stored in guide.varm.",
    )
    parser.add_argument(
        "--guide-element-uns-key",
        default="intended_targets",
        help="Target-name list stored in guide.uns and gene.uns.",
    )
    args = parser.parse_args(argv)

    output_path, summary = write_gasperini_atscale_tss_subset_h5mu(
        input_path=args.input,
        output_path=args.output,
        gene_modality=args.gene_modality,
        guide_modality=args.guide_modality,
        guide_by_element_key=args.guide_by_element_key,
        guide_element_uns_key=args.guide_element_uns_key,
    )
    print(f"wrote {output_path}")
    print(summary)


def _guide_target_names(
    guide: ad.AnnData,
    *,
    guide_by_element_key: str,
    guide_element_uns_key: str,
) -> list[str]:
    if guide_by_element_key not in guide.varm:
        raise KeyError(f"guide.varm['{guide_by_element_key}'] was not found.")

    mapping = to_dense_array(guide.varm[guide_by_element_key]).astype(bool, copy=False)
    if mapping.shape[0] != guide.n_vars:
        raise ValueError(
            f"guide.varm['{guide_by_element_key}'] has {mapping.shape[0]} guide rows, "
            f"but the guide modality has {guide.n_vars} guides."
        )

    if guide_element_uns_key in guide.uns:
        element_names = [str(x) for x in np.asarray(guide.uns[guide_element_uns_key]).reshape(-1).tolist()]
    elif hasattr(guide.varm[guide_by_element_key], "columns"):
        element_names = [str(x) for x in guide.varm[guide_by_element_key].columns.tolist()]
    else:
        raise KeyError(
            f"Could not infer target names from guide.uns['{guide_element_uns_key}'] or guide.varm['{guide_by_element_key}']."
        )

    if mapping.shape[1] != len(element_names):
        raise ValueError(
            f"guide.varm['{guide_by_element_key}'] has {mapping.shape[1]} targets, "
            f"but guide.uns['{guide_element_uns_key}'] has {len(element_names)} names."
        )

    return element_names


def _guide_row_keep_mask(
    mapping: np.ndarray,
    *,
    target_names: list[str],
    selected_target_names: list[str],
) -> np.ndarray:
    target_lookup = {name.lower() for name in selected_target_names}
    keep_mask = np.zeros(mapping.shape[0], dtype=bool)
    for row_idx, row in enumerate(np.asarray(mapping, dtype=bool)):
        row_target_names = [str(target_names[col_idx]) for col_idx in np.flatnonzero(row)]
        if any(target_name.lower() in target_lookup for target_name in row_target_names):
            keep_mask[row_idx] = True
    return keep_mask


def _gene_symbols(gene: ad.AnnData) -> pd.Index:
    if "symbol" in gene.var.columns:
        return pd.Index(gene.var["symbol"].astype(str))
    return pd.Index(gene.var_names.astype(str))


def _gene_symbols_from_tss_targets(tss_target_names: list[str]) -> list[str]:
    symbols: list[str] = []
    for target_name in tss_target_names:
        value = str(target_name)
        if value.lower().endswith("_tss"):
            value = value[:-4]
        if value and value.lower() != "nan":
            symbols.append(value)
    return symbols


if __name__ == "__main__":
    main()
