"""Build the two laptop-sized tutorial panels from the deposited K562 screens."""

import argparse
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
from scipy import sparse


HERE = Path(__file__).resolve().parent
GENES = 256


def fill_panel(selected, candidates, size, rng):
    """Keep named anchors and sample the remaining identifiers without replacement."""
    selected = set(selected)
    eligible = sorted(set(candidates) - selected)
    selected.update(rng.choice(eligible, size=size - len(selected), replace=False))
    return selected


def gasperini(source, destination):
    rng = np.random.default_rng(12)
    original = md.read_h5mu(source, backed="r")
    try:
        rna, grna = original.mod["gene"], original.mod["guide"]
        if not rna.obs_names.equals(grna.obs_names):
            raise ValueError("RNA and guide cell axes differ.")
        elements = pd.Index(grna.uns["intended_targets"])
        design = sparse.csr_matrix(grna.varm["guide_intended_target_pairs"])
        pairs = pd.read_parquet(HERE.parents[1] / "examples/manuscript/gasperini-cis-candidates.parquet")
        pairs = pairs.loc[pairs.gene.isin(rna.var_names) & pairs.element.isin(elements)]
        candidates = pairs.groupby("element", observed=True).gene.nunique()

        # Include the PPP2R5A neighborhood and the LMO2 showcase. Keep both
        # guides for each selected element. Select other enhancers by eligibility,
        # before inspecting effect estimates or significance.
        guide_var = grna.var
        locus = guide_var.intended_target_chr.eq("chr1") & guide_var.intended_target_start.between(
            212224575, 213224575
        )
        neighbors = set(guide_var.loc[locus, "intended_target_name"]) & set(candidates.index)
        anchors = neighbors | {"chr11.1735_top_two"}
        two_guides = set(elements[np.asarray(design.sum(axis=0)).ravel() == 2])
        eligible = set(candidates[candidates >= 2].index) & two_guides
        enhancers = fill_panel(anchors, eligible, 94, rng)
        selected = enhancers | {"ACTB_TSS", "LMO2_TSS", "random_1", "random_2", "random_3", "random_4"}
        column_ids = np.flatnonzero(elements.isin(selected))
        if len(column_ids) != 100:
            raise ValueError("The source lacks expected elements.")
        guide_mask = np.asarray(design[:, column_ids].sum(axis=1)).ravel() > 0

        genes = set(rna.var_names[rna.var.symbol.isin(["ACTB", "LMO2", "GATA1", "HBG1", "RPL11", "HSPA5"])])
        genes.update(rna.var_names[rna.var.gene_chr.eq("chr1") & rna.var.gene_start.between(212224575, 213224575)])
        selected_pairs = pairs.loc[pairs.element.isin(enhancers)]
        for _, group in selected_pairs.groupby("element", sort=True, observed=True):
            candidates_for_element = sorted(set(group.gene))
            genes.update(rng.choice(candidates_for_element, min(2, len(candidates_for_element)), replace=False))
        genes = fill_panel(genes, selected_pairs.gene, GENES, rng)
        gene_mask = rna.var_names.isin(genes)
        rows = np.sort(rng.choice(rna.n_obs, 8000, replace=False))
        obs = rna.obs.iloc[rows][["total_umis", "percent_mito", "prep_batch", "guide_count"]].copy()
        gene = ad.AnnData(rna.X[rows, :][:, gene_mask], obs=obs, var=rna.var.loc[gene_mask].copy())
        guide = ad.AnnData(grna.X[rows, :][:, guide_mask], obs=obs.copy(), var=guide_var.loc[guide_mask].copy())
        guide.varm["guide_intended_target_pairs"] = design[guide_mask][:, column_ids]
        guide.uns["intended_targets"] = elements[column_ids].to_numpy(dtype=str)
        if not (np.asarray(guide.varm["guide_intended_target_pairs"].sum(axis=1)).ravel() == 1).all():
            raise ValueError("Each retained guide must map to one element.")
        md.MuData({"gene": gene, "guide": guide}).write_h5mu(destination, compression="gzip")
    finally:
        original.file.close()


def replogle(source, destination):
    rng = np.random.default_rng(12)
    original = ad.read_h5ad(source, backed="r")
    try:
        obs = original.obs
        unique = original.var.loc[~original.var.gene_name.duplicated(keep=False)]
        constructs = obs.gene_transcript.astype(str)
        controls = constructs.str.contains("non-targeting", regex=False)
        counts = obs.loc[~controls].groupby(["gene", "gene_transcript"], observed=True).size().rename("n")
        choices = counts.reset_index().query("n >= 64").sort_values(
            ["gene", "n", "gene_transcript"], ascending=[True, False, True]
        ).drop_duplicates("gene").set_index("gene")
        choices = choices.loc[choices.index.isin(unique.gene_name)]
        targets = fill_panel(["GATA1", "HSPA5", "NEDD8", "RPL11"], choices.index, 100, rng)
        groups = obs.groupby("gene_transcript", observed=True).indices
        rows = [rng.choice(np.flatnonzero(controls), 1024, replace=False)]
        for construct in choices.loc[sorted(targets), "gene_transcript"]:
            rows.append(rng.choice(groups[construct], 64, replace=False))
        rows = np.sort(np.concatenate(rows))

        regulon = set((HERE / "inputs/gata1_targets.txt").read_text().splitlines())
        anchors = targets | regulon | {"CALR", "HSP90B1", "ACTB"}
        if not anchors <= set(unique.gene_name):
            raise ValueError(f"Missing response genes: {sorted(anchors - set(unique.gene_name))}")
        symbols = fill_panel(anchors, unique.gene_name, GENES, rng)
        genes = original.var.gene_name.isin(symbols)
        # AnnData may be dense on disk. Read short row batches to cap memory.
        counts = np.concatenate([original.X[part, :][:, genes] for part in np.array_split(rows, 64)])
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
        parser.error("Choose a new destination to preserve existing inputs.")
    {"gasperini": gasperini, "replogle": replogle}[args.screen](args.source, args.destination)
