import sys

import anndata as ad
import matplotlib.pyplot as plt
import mudata as md
import numpy as np
import pandas as pd
import pyro

# import scanpy as sc
import seaborn as sns

import perturbvi

# from scipy.stats import norm


smoke_test = True


def construct_mudata(data_dir="."):
    rna_adata = ad.read_h5ad(f"{data_dir}/ann_exp.h5ad")
    rna_adata.obs["library_size"] = rna_adata.X.sum(axis=1)
    # rna_adata.obs["size_factor"] = rna_adata.obs["library_size"]
    rna_adata.varm["element_tested"] = ad.read_h5ad(f"{data_dir}/ann_Element_x_tested_genes.h5ad").to_df().T
    # sc.pp.filter_genes(rna_adata, min_cells=len(rna_adata)*0.1) # need genes where at least 10% of cells with nonzero counts
    grna_adata = ad.read_h5ad(f"{data_dir}/ann_guide.h5ad")
    grna_adata.varm["element_targeted"] = ad.read_h5ad(f"{data_dir}/ann_Element_guide.h5ad").to_df().T
    # assert (rna_adata.varm["element_tested"].columns == grna_adata.varm["element_targeted"].columns).all()
    mdata = md.MuData({"rna": rna_adata, "grna": grna_adata})
    # mask = np.zeros(len(grna_adata))
    # mask[: int(len(grna_adata) * dilute_guides)] = 1.0
    # grna_adata.X = grna_adata.X * mask
    print(len(mdata))
    # mdata.write_h5mu(f"{data_dir}/gasperini_pilot_highMOI.h5mu")
    return mdata


def downsample_mudata(mdata, downsample_prob=1.0):
    downsample_idx = np.random.choice(len(mdata), size=int(len(mdata) * downsample_prob), replace=False)
    return mdata[downsample_idx, :].copy()


def filter_mudata(mdata, selected_guides, selected_genes):
    # guides = mdata["grna"].var_names
    # selected_guides = list(guides[guides.str.contains(r"TSS|random|scrambled", regex=True)])

    elements = list({guide.split("|")[0] for guide in selected_guides})
    selected_elements = [e for e in elements if e in mdata["grna"].varm["element_targeted"].columns]
    element_subset = mdata["grna"].varm["element_targeted"][selected_elements].copy()

    grna_subset = mdata["grna"].copy()
    grna_subset.varm["element_targeted"] = element_subset

    grna_subset = mdata["grna"][:, selected_guides]
    # genes = list(sorted({e.split("_")[0] for e in selected_elements if "_TSS" in e}))
    # subset_genes = [g for g in genes if g in mdata["rna"].var_names]
    # print(selected_genes)

    # assert grna_subset.varm["element_targeted"].shape == (len(selected_guides), len(selected_elements))

    rna_subset = mdata["rna"].copy()
    rna_subset.varm["element_tested"] = rna_subset.varm["element_tested"][selected_elements].copy()
    rna_subset = rna_subset[:, selected_genes]
    mdata_subset = md.MuData({"rna": rna_subset.copy(), "grna": grna_subset.copy()})
    # assert (
    #     mdata_subset["rna"].varm["element_tested"].columns == mdata_subset["grna"].varm["element_targeted"].columns
    # ).all()
    return mdata_subset


def train_model(mdata_subset):
    perturbvi.PERTURBVI.setup_mudata(
        mdata_subset,
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

    model = perturbvi.PERTURBVI(mdata_subset)

    # burn-in
    model.train(10, lr=0.1, batch_size=2048)

    # fine tune
    max_epochs = 20
    model.train(max_epochs=max_epochs, lr=0.01, batch_size=8192)

    # lfc_threshold = 0.05  # ~ at least 5% knockdown
    # lfc_threshold = 0 # nonzero knockdown
    # gene_index=45

    scale_factor = 0.01
    perturb_mean_lfc_mu = pyro.get_param_store()["perturb_mean_lfc.mu"].detach().cpu().numpy()
    lfc_cov_tril = pyro.get_param_store()["perturb_lfc.scale_tril"] * scale_factor
    lfc_cov = lfc_cov_tril @ lfc_cov_tril.transpose(dim0=-1, dim1=-2)
    perturb_mean_lfc_sigma = lfc_cov[..., 0, 0].sqrt().detach().cpu().numpy()
    assert perturb_mean_lfc_mu.shape == perturb_mean_lfc_sigma.shape
    # perturb_z_scores = perturb_mean_lfc_mu / perturb_mean_lfc_sigma
    # perturb_p_vals = norm.cdf(lfc_threshold, loc=-perturb_mean_lfc_mu, scale=perturb_mean_lfc_sigma)

    results_all_df = (
        pd.DataFrame(data=perturb_mean_lfc_mu, columns=selected_genes, index=selected_guides)
        .melt(var_name="gene", ignore_index=False)
        .reset_index(names="guide")
        .sort_values("value")
        .assign(is_target=lambda x: x["guide"].str.split("_", expand=True)[0] == x["gene"])
    )

    # sns.histplot(
    #     results_all_df,
    #     hue="is_target",
    #     x="value",
    #     common_norm=False,
    #     log_scale=[False, True],
    #     stat="density",
    #     binwidth=0.1,
    # )
    # plt.title(selected_genes[0])
    # plt.savefig(f"{data_dir}/hist_{downsample_prob}.png")

    element = f"{selected_genes[0]}_TSS"
    results_all_df.set_index("guide", inplace=True)
    # print(results_all_df)
    return (results_all_df.loc[f"{element}|1"]["value"], results_all_df.loc[f"{element}|2"]["value"])


if __name__ == "__main__":
    data_dir = sys.argv[1]

    sceptre_df = pd.read_csv(f"{data_dir}/sceptre_results-2023_01_13.csv")
    sceptre_df = sceptre_df.rename(columns={"gRNA_id": "guide", "gene_id": "gene"})
    selected_genes = ["KRT18"]
    selected_guides = sceptre_df["guide"].unique()
    if len(sys.argv) > 2:
        downsample_prob = float(sys.argv[2])
    else:
        downsample_prob = 1
    mdata = construct_mudata(data_dir)
    mdata_subset = filter_mudata(mdata, selected_guides, selected_genes)
    mdata_subset = downsample_mudata(mdata_subset, downsample_prob)
    z1, z2 = train_model(mdata_subset)
    print(z1, z2)
