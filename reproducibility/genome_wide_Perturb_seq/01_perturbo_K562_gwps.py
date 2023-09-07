#!/usr/bin/env python
# coding: utf-8

# ## Set up data and model objects

# In[1]:

import pandas as pd
import numpy as np
import os
import sys

import mudata as md
import anndata as ad
import perturbo
import seaborn as sns
import pyro
from scipy.stats import norm
import matplotlib.pyplot as plt
import torch
import scvi

scvi.settings.seed = 0

# Set training configuration and access to data

# In[2]:

data_dir = "/data/pinello/SHARED_DATA/genome_wide_Perturb-seq"
batch_size=16384
min_umi_count=100
umi_key = "UMI_count"
mito_key = "mitopercent"
batch_key = "gem_group"
    
if torch.cuda.is_available():
    accelerator = "gpu"
    print("training using GPU")
else:
    accelerator = "cpu"
    print("training using CPU")

# In[3]:

chunk_number = int(os.environ["SLURM_ARRAY_TASK_ID"])
mdata_file = f"{data_dir}/K562_gwps_raw_singlecell.chunk_{chunk_number:02d}.h5mu"
# mdata_file = sys.argv[1]
mdata = md.read_h5mu(mdata_file)
mdata

# ### Load the data

# In[4]:

# umi_counts = mdata['rna'].obs[umi_key]
# plt.hist(np.log10(umi_counts), bins = np.arange(0,10,0.1))
# plt.axvline(2.0)
# selected_cells = umi_counts>min_umi_count
# plt.savefig(mdata_file.replace("h5mu", "umi_counts.png"))
# len(selected_cells)
# np.mean()


# In[9]:

# mdata = mdata[selected_cells,:].copy()
# mdata

perturbo.PERTURBO.setup_mudata(
    mdata,
    batch_key=batch_key,
    library_size_key=umi_key,
    continuous_covariates_keys=[mito_key],
    # perturb_by_element_key="element_targeted",
    modalities={
        "rna_layer": "rna",
        "perturbation_layer": "grna",
    },
)

model = perturbo.PERTURBO(
    mdata, 
    likelihood='lnnb', 
    fit_lib_size=False, 
    fit_dispersion=False
)
model.view_anndata_setup()


# burn-in
pyro.clear_param_store()

model.train(10, lr=0.01, batch_size=batch_size//4, accelerator=accelerator) 
model.train(40, lr=0.001, batch_size=batch_size, accelerator=accelerator)

# In[19]:

from scipy.stats import chi2

# compute "q-value" based on 1 - maximum posterior CI which overlaps zero
q_mu, q_sigma = model.module.get_element_effects()
assert q_mu.shape == q_sigma.shape
perturb_z_scores = q_mu/q_sigma
perturb_p_vals=chi2.sf((perturb_z_scores)**2, df=1)

# In[20]:

def make_long_df(mat, value_name):
    return (
        pd.DataFrame(data=mat, index=mdata['grna'].var_names, columns=mdata['rna'].var['gene_name'])
        .melt(var_name="gene", value_name=value_name, ignore_index=False)
        .reset_index(names="element")
    )

scpower_df = pd.merge(
    make_long_df(perturb_z_scores, "z_value"),
    make_long_df(perturb_p_vals, "p_value")
)

results_file = mdata_file.replace("h5mu", "results.csv.gz")
print(f"Writing test results to {results_file}")
scpower_df.to_csv(results_file)


for k, v in pyro.get_param_store().items():
    print(k, v.shape)
    if sum(v.shape) <= 5:
        print(v)
