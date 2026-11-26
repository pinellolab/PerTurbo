"""Rebuild the small K562 tutorial inputs from the public raw count files."""

import argparse
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
from scipy import sparse


def selected_genes(var, column, symbols):
    selected = var[column].isin(symbols)
    if sorted(var.loc[selected, column].astype(str)) != sorted(symbols):
        raise ValueError(f"Expected exactly one source gene for each of {symbols}")
    return selected.to_numpy()


def gasperini(source, destination):
    original = md.read_h5mu(source, backed="r")
    try:
        gene, guide = original.mod["gene"], original.mod["guide"]
        if not gene.obs_names.equals(guide.obs_names):
            raise ValueError("RNA and guide rows must have the same cell IDs in the same order.")
        elements = pd.Index(guide.uns["intended_targets"])
        names = ["ACTB_TSS", "LMO2_TSS", "chr11.1735_top_two", "random_1"]
        element_ids = elements.get_indexer(names)
        if not elements.is_unique or (element_ids < 0).any():
            raise ValueError("Missing or ambiguous guide-to-element labels")
        mapping = sparse.csr_matrix(guide.varm["guide_intended_target_pairs"])
        guide_mask = np.asarray(mapping[:, element_ids].sum(axis=1)).ravel() > 0
        genes = selected_genes(gene.var, "symbol", ["ACTB", "LMO2", "GATA1", "HBG1", "RPL11", "HSPA5"])
        rows = np.sort(np.random.default_rng(12).choice(gene.n_obs, 12000, replace=False))
        obs = gene.obs.iloc[rows][["total_umis", "percent_mito", "prep_batch", "guide_count"]].copy()
        rna = ad.AnnData(gene.X[rows, :][:, genes], obs=obs, var=gene.var.loc[genes, ["symbol"]].copy())
        guides = ad.AnnData(
            guide.X[rows, :][:, guide_mask],
            obs=obs.copy(),
            var=guide.var.loc[guide_mask, ["intended_target_name", "is_non_targeting"]].copy(),
        )
        guides.varm["guide_intended_target_pairs"] = mapping[guide_mask][:, element_ids]
        guides.uns["intended_targets"] = np.array(names, dtype=str)
        assert (np.asarray(guides.varm["guide_intended_target_pairs"].sum(axis=1)).ravel() == 1).all()
        md.MuData({"gene": rna, "guide": guides}).write_h5mu(destination, compression="gzip")
    finally:
        original.file.close()


def replogle(source, destination):
    original = ad.read_h5ad(source, backed="r")
    try:
        obs = original.obs
        constructs = obs["gene_transcript"].astype(str)
        controls = constructs[constructs.str.contains("non-targeting", regex=False)].value_counts().head(6).index
        treated = obs["gene"].isin(["HSPA5", "NEDD8", "RPL11"])
        rows = np.flatnonzero(treated | constructs.isin(controls))
        genes = selected_genes(original.var, "gene_name", ["HSPA5", "NEDD8", "RPL11", "CALR", "HSP90B1", "ACTB"])
        # Slice the backed matrix by row first (the deposited h5ad is dense).
        counts = original.X[rows, :][:, genes]
        example = ad.AnnData(
            counts,
            obs=obs.iloc[rows][["gene_transcript", "gene", "UMI_count", "gem_group"]].copy(),
            var=original.var.loc[genes, ["gene_name"]].copy(),
        )
        example.write_h5ad(destination, compression="gzip")
    finally:
        original.file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen", choices=["gasperini", "replogle"])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error("Choose a new destination or remove the old copy first.")
    {"gasperini": gasperini, "replogle": replogle}[args.screen](args.source, args.destination)
