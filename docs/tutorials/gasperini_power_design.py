"""Construct the two-guide Gasperini tutorial screen from a fitted MuData and a cis map."""
from pathlib import Path

import anndata as ad
import jax
import mudata as md
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import brentq
from scipy.special import expit

from perturbo.simulation.predictive import sample_predictive_counts


def choose_pairs(assays, declared, *, seed, max_elements=32):
    """Choose candidate-rich enhancers before drawing any counts or looking at calls."""
    pt = assays.uns["_perturbo"]
    rna = assays.mod["gene"]
    gene_ids = pd.Index(pt["metadata"]["gene_names"])
    if not gene_ids.is_unique or not gene_ids.equals(rna.var_names):
        raise ValueError("Run the Gasperini analysis tutorial with Ensembl IDs on the fit gene axis.")
    map_elements = assays.mod["guide"].uns["intended_targets"]
    pairs = declared.loc[
        declared.element.isin(map_elements) & declared.gene.isin(gene_ids)
        & declared.element.str.startswith("chr"), ["element", "gene"]
    ].drop_duplicates().sort_values(["element", "gene"])
    pairs = pairs.groupby("element").filter(lambda group: len(group) >= 2)
    ppp = rna.var.index[rna.var.symbol == "PPP2R5A"]
    locus = pairs.loc[pairs.gene.isin(ppp), "element"].sort_values().to_list()[:6]
    if len(locus) < 2:
        raise ValueError("The tutorial needs two or more mapped elements at the PPP2R5A locus.")
    rng = np.random.default_rng(seed)
    others = np.setdiff1d(pairs.element.unique(), locus)
    n_other = min(len(others), max_elements - len(locus))
    elements = pd.Index([*locus, *rng.choice(others, n_other, replace=False)], name="element")
    if len(elements) < 12:
        raise ValueError("Use the analysis tutorial input with at least 12 multi-candidate enhancers.")
    pairs = pairs.loc[pairs.element.isin(elements)].copy()
    # Keep the focal links visible and reserve one quarter of all elements as complete nulls.
    nulls = set(rng.choice(elements.difference(locus), len(elements) // 4, replace=False))
    targets = [
        (element, rng.choice(pairs.loc[pairs.element == element, "gene"]))
        for element in elements if element not in nulls
    ]
    truth = pd.DataFrame(targets, columns=["element", "gene"])
    pairs = pairs.merge(truth.assign(planted=True), how="left", validate="one_to_one")
    pairs["planted"] = pairs.planted.eq(True)
    return elements, pairs.reset_index(drop=True), locus


def write_draw(assays, elements, pairs, efficiencies, destination, *,
               coverage, parent_lnfc, replicate, scenario, seed):
    """Simulate one independent draw, keeping fitted offsets and covariates attached to donors."""
    pt = assays.uns["_perturbo"]
    control = pt["control_fit"]
    rna = assays.mod["gene"]
    genes = rna.var_names[rna.var_names.isin(pairs.gene)]
    positions = rna.var_names.get_indexer(genes)
    cov = pt["cell_covariates"]
    if not cov.index.equals(rna.obs_names):
        raise ValueError("The covariate rows must align to the source cells.")
    expected_columns = ["total_umis", "percent_mito", "batch:prep_batch=prep_batch_2"]
    if list(cov.columns) != expected_columns:
        raise ValueError(f"The tutorial uses the Gasperini covariates in this order: {expected_columns}")
    # Match donors and guide assignments across effect sizes within a replicate.
    rng = np.random.default_rng([seed, coverage, replicate])
    n_cells = round(coverage * len(elements) / 4)
    donors = rng.choice(rna.n_obs, n_cells, replace=n_cells > rna.n_obs)
    obs = rna.obs.iloc[donors][["total_umis", "percent_mito", "prep_batch"]].copy()
    obs["source_cell"] = obs.index.astype(str)
    obs.index = pd.Index([f"cell_{i}" for i in range(n_cells)])
    offsets = np.asarray(pt["size_factors"], dtype="float32")[donors].reshape(-1, 1)
    obs["log_sf"] = offsets[:, 0]
    z = cov.iloc[donors].to_numpy(dtype="float32")
    # Fitted Figure 2 guide-assignment slopes for UMI, mitochondrial fraction, and batch.
    scores = z @ np.array([0.18814895, -0.03074144, -0.01101596])
    def expected_cells(intercept):
        p = expit(scores + intercept)
        return (p * (2 - p)).sum()
    intercept = brentq(lambda a: expected_cells(a) - coverage, -80, 30)
    p = expit(scores + intercept)
    assignments = rng.random((n_cells, len(elements), 2)) < p[:, None, None]
    design = (assignments * efficiencies[None, :, :]).sum(axis=2).astype("float32")

    targets = pairs.loc[pairs.planted]
    beta = np.zeros((len(elements), len(genes)), dtype="float32")
    beta[elements.get_indexer(targets.element), genes.get_indexer(targets.gene)] = parent_lnfc
    sites = dict(beta=beta, beta_0=control["beta_0"][positions], theta=control["theta"][positions],
                 covariate_coef=control["covariate_coef"][:, positions])
    key = jax.random.PRNGKey(seed)
    for value in (coverage, replicate, scenario):
        key = jax.random.fold_in(key, value)
    counts, returned_offsets = sample_predictive_counts(
        sites, design, key, size_factors=offsets, covariates=z, chunk_rows=1024,
    )
    np.testing.assert_array_equal(offsets, returned_offsets)
    gene = ad.AnnData(counts, obs=obs, var=rna.var.loc[genes].copy())
    guide = ad.AnnData(
        sparse.csr_matrix(assignments.reshape(n_cells, -1), dtype="float32"),
        obs=pd.DataFrame(index=obs.index),
        var=pd.DataFrame({"efficiency_true": efficiencies.ravel()}, index=[
            f"{element}_g{j}" for element in elements for j in (1, 2)
        ]),
    )
    guide.varm["mapping"] = sparse.csr_matrix(np.repeat(np.eye(len(elements), dtype="float32"), 2, axis=0))
    guide.uns["elements"] = elements.to_numpy(dtype=str)
    md.MuData({"gene": gene, "guide": guide}).write_h5mu(Path(destination))
    return assignments.any(axis=2).sum(axis=0).mean()
