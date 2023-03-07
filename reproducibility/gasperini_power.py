from argparse import ArgumentError
import mudata as md
import anndata as ad
import numpy as np
import pandas as pd
import perturbvi
import os
import scipy
import pyro
from scipy.stats import norm
import torch
import sys

smoke_test = True
if len(sys.argv) == 5:
    _, data_dir, output_dir, subsample_frac, downsample_frac = sys.argv
    subsample_frac = float(subsample_frac)
    downsample_frac = float(downsample_frac)
else:
    raise Exception(
        "usage: python gasperini_power.py DATA_DIR p_SUBSAMPLE p_DOWNSAMPLE"
    )

# selected_guides = ["ACTG1_TSS|1", "ACTG1_TSS|2", "ACTB_TSS|1", "ACTB_TSS|2"]
# selected_genes = ["ACTG1"]
selected_guides = ["TMED10_TSS|1", "TMED10_TSS|2"]
selected_genes = ["TMED10"]
# selected_guides = ["KRT18_TSS|1", "KRT18_TSS|2"]
# selected_genes = ["KRT18"]


sceptre_df = pd.read_csv(f"{data_dir}/sceptre_results-2023_01_13.csv")
sceptre_df = sceptre_df.rename(
    columns={
        "gRNA_id": "guide",
        "gene_id": "gene",
        "z_value": "SCEPTRE_z_value",
        "p_value": "SCEPTRE_p_value",
    }
)
if not smoke_test:
    selected_genes = sceptre_df["gene"].unique()
    selected_guides = sceptre_df["guide"].unique()

# sceptre_df.query("guide=='GATA2_TSS|1'")


def construct_mudata(data_dir="."):
    rna_adata = ad.read_h5ad(f"{data_dir}/ann_exp.h5ad")
    rna_adata.obs["library_size"] = rna_adata.X.sum(axis=1)
    # rna_adata.obs["size_factor"] = rna_adata.obs["library_size"]
    rna_adata.varm["element_tested"] = (
        ad.read_h5ad(f"{data_dir}/ann_Element_x_tested_genes.h5ad").to_df().T
    )
    # sc.pp.filter_genes(rna_adata, min_cells=len(rna_adata)*0.1) # need genes where at least 10% of cells with nonzero counts
    grna_adata = ad.read_h5ad(f"{data_dir}/ann_guide.h5ad")
    grna_adata.varm["element_targeted"] = (
        ad.read_h5ad(f"{data_dir}/ann_Element_guide.h5ad").to_df().T
    )
    assert (
        rna_adata.varm["element_tested"].columns
        == grna_adata.varm["element_targeted"].columns
    ).all()
    mdata = md.MuData({"rna": rna_adata, "grna": grna_adata})
    return mdata


force = True
mudata_file = "gasperini_pilot_highMOI.h5mu"
if mudata_file not in os.listdir(data_dir) or force:
    mdata = construct_mudata(data_dir)
else:
    mdata = md.read_h5mu(os.path.join(data_dir, mudata_file))

# mdata.raw=mdata.copy()
total_umis = mdata.mod["rna"].X.sum()
# print('total depth', total_umis/1e6, 'million UMI counts')

# downsample UMIS (w/ replacement)
counts_tensor = torch.tensor(mdata.mod["rna"].X.todense(), dtype=torch.int64)
counts_sampler = torch.distributions.Binomial(
    counts_tensor, probs=torch.tensor(downsample_frac)
)
counts_sampled = counts_sampler.sample((1,))
mdata["rna"].X = scipy.sparse.csr_array(counts_sampled[0, ...].cpu().numpy())
# mdata['rna'].X = scipy.sparse.csr_matrix(counts_sampled)
downsample_umis = int(mdata["rna"].X.sum())
print(
    "subsampled depth",
    downsample_umis / 1e6,
    "million UMI counts",
)
mdata.update()


# guides = mdata["grna"].var_names
# # selected_guides = list(guides[guides.str.contains(r"TSS|random|scrambled", regex=True)])

# elements = list({guide.split("|")[0] for guide in selected_guides})
# selected_elements = [e for e in elements if e in mdata["grna"].varm["element_targeted"].columns]
# element_subset = mdata["grna"].varm["element_targeted"][selected_elements].copy()

# grna_subset = mdata["grna"]
# grna_subset.varm["element_targeted"] = element_subset

# grna_subset = mdata["grna"][:, selected_guides]
# # genes = list(sorted({e.split("_")[0] for e in selected_elements if "_TSS" in e}))
# # subset_genes = [g for g in genes if g in mdata["rna"].var_names]
# # print(selected_genes)

# if smoke_test:
#     selected_genes = [test_gene]  # DANGER SINGLE GENE ONLY
# assert grna_subset.varm["element_targeted"].shape == (len(selected_guides), len(selected_elements))
mdata.mod["grna"] = mdata.mod["grna"][:, selected_guides]
mdata.mod["rna"] = mdata.mod["rna"][:, selected_genes]
mdata.mod["rna"].X.sum()


guide_obs = mdata["grna"][:, selected_guides].X.sum(axis=1).copy().astype(bool)
guide_obs_idx = np.where(guide_obs)[0]
discard_cells = np.random.choice(
    guide_obs_idx, size=int(guide_obs.sum() * (1 - subsample_frac)), replace=False
)
subset_idx = [i for i in range(len(mdata)) if i not in discard_cells]
print("discarded", len(discard_cells), "cells with guide")
mdata.update()
mdata = mdata[subset_idx, :].copy()
mdata
# mdata.mod['rna']
# rna_downsample = sc.pp.downsample_counts(mdata.mod['rna'], total_counts = 1e8)


# rna_subset = mdata["rna"]
# rna_subset.varm["element_tested"] = rna_subset.varm["element_tested"][selected_elements].copy()
# rna_subset = rna_subset[:, selected_genes]
# mdata_subset = md.MuData({"rna": rna_subset.copy(), "grna": grna_subset.copy()})
# assert (
#     mdata_subset["rna"].varm["element_tested"].columns == mdata_subset["grna"].varm["element_targeted"].columns
# ).all()
# mdata_subset


perturbvi.PERTURBVI.setup_mudata(
    mdata,
    batch_key="bath_number",
    library_size_key="library_size",
    # size_factor_key="library_size",
    var_by_element_key="element_tested",
    perturb_by_element_key="element_targeted",
    modalities={
        "rna_layer": "rna",
        "perturbation_layer": "grna",
    },
)

model = perturbvi.PERTURBVI(mdata, fit_lib_size=True)
model.view_anndata_setup()


# requires graphviz to be installed
model.render_model()


model.render_guide()

# ## Model training
#
# We can start out with a really high learning rate, then decrease the learning rate to fine-tune
#


# burn-in
if smoke_test:
    model.train(20, lr=0.1, batch_size=1024)
else:
    model.train(10, lr=0.1, batch_size=4096)


# fine-tuning
model.train(100, lr=0.01, batch_size=8192)


# plt.hist(pyro.get_param_store()["library_size_effect.mu"].cpu().detach().numpy())


# lfc_threshold = -0.1 # ~ at least 10% upreg
lfc_threshold = 0  # nonzero knockdown
# gene_index=45

# scale_factor = 0.01
perturb_mean_lfc_mu = (
    pyro.get_param_store()["perturb_mean_lfc.mu"].detach().cpu().numpy()
)
perturb_disp_lfc_mu = (
    pyro.get_param_store()["perturb_disp_lfc.mu"].detach().cpu().numpy()
)
lfc_cov_tril = pyro.get_param_store()["perturb_lfc.scale_tril"]
lfc_cov = lfc_cov_tril @ lfc_cov_tril.transpose(dim0=-1, dim1=-2)
perturb_mean_lfc_sigma = lfc_cov[..., 0, 0].sqrt().detach().cpu().numpy()
perturb_disp_lfc_sigma = lfc_cov[..., 1, 1].sqrt().detach().cpu().numpy()
assert perturb_mean_lfc_mu.shape == perturb_mean_lfc_sigma.shape
perturb_z_scores = perturb_mean_lfc_mu / perturb_mean_lfc_sigma
perturb_p_vals = norm.sf(
    lfc_threshold, loc=perturb_mean_lfc_mu, scale=perturb_mean_lfc_sigma
)


def make_long_df(mat, value_name):
    return (
        pd.DataFrame(data=mat, columns=selected_genes, index=selected_guides)
        .melt(var_name="gene", value_name=value_name, ignore_index=False)
        .reset_index(names="guide")
        .assign(
            is_target=lambda x: x["guide"].str.split("_", expand=True)[0] == x["gene"]
        )
    )


mu_df = make_long_df(perturb_mean_lfc_mu, "mu")
z_df = make_long_df(perturb_z_scores, "z_value")
p_df = make_long_df(perturb_p_vals, "p_value")


scpower_df = pd.merge(mu_df, p_df)
if not smoke_test:
    scpower_df.to_csv("~/Downloads/all_results.csv")
# significant_results = scpower_df.query("p_value < 1e-8")
# significant_results['element'] = significant_results['guide'].str.split("|", expand=True)[0]
# significant_results
scpower_df

# guide_gene_pairs = significant_results.groupby(['element','gene'])['guide'].count()
# guide_gene_pairs=guide_gene_pairs[guide_gene_pairs==2]
# guide_gene_pairs.to_excel("~/Downloads/guide_pairs.xlsx")
# # guide_gene_pairs.index
# scpower_df.query("p_value < 1e-3").to_excel("~/Downloads/significant_results.xlsx")
# significant_results['guide'].str.split("_", expand=True)[0]
# significant_results.head(1000).to_excel("~/Downloads/significant_results.xlsx")


scpower_df.to_csv(f"{output_dir}/test_{subsample_frac}_{downsample_frac}.csv")
