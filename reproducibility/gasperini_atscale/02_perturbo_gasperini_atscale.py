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


data_dir = "/data/pinello/SHARED_DATA/gasperini_2019/gasperini_atscale"
batch_size=2048
min_umi_count=100


if torch.cuda.is_available():
    accelerator = "gpu"
    print("training using GPU")
else:
    accelerator = "cpu"
    print("training using CPU")

# In[3]:

chunk_number = int(os.environ["SLURM_ARRAY_TASK_ID"])
mdata_file = f"{data_dir}/GSE120861_at_scale_screen_chunk.{chunk_number:02d}.h5mu"
# mdata_file = sys.argv[1]
mdata_all = md.read_h5mu(mdata_file)
mdata_all


# ### Load the data

# In[4]:


plt.hist(np.log10(mdata_all['rna'].obs['umi_count']), bins = np.arange(0,10,0.1))
# plt.axvline(2.0)
selected_cells = mdata_all['rna'].obs['umi_count']>min_umi_count
plt.savefig(mdata_file.replace("h5mu", "umi_counts.png"))
len(selected_cells)
# np.mean()

# In[6]:


# selected_guides = mdata_all['grna'].var.index

# In[7]:


selected_genes = mdata_all['rna'].var_names

# chunk_size=5000
# selected_genes = mdata_all['rna'].var_names[:chunk_size]
# selected_genes = mdata_all['rna'].var_names[log_mean_counts_per_cell > -3]
# selected_genes = sceptre_results['gene_id'].unique()
selected_genes
# len(selected_genes)


# In[8]:

selected_guides = mdata_all['grna'].var.index
# selected_guides = mdata_all['grna'].var.query('Category=="NTC" or Category=="TSS"').index
selected_guides


# In[9]:


mdata_subset = mdata_all[selected_cells,:]

# subset to desired grnas/elements
grna_subset = mdata_subset.mod['grna'][:,selected_guides].copy()
selected_element_idx = np.where(grna_subset.varm['element_targeted'].sum(axis=0)>0)[0]
selected_elements = grna_subset.uns['elements'][selected_element_idx]
grna_subset.varm['element_targeted'] = grna_subset.varm['element_targeted'][:,selected_element_idx]
grna_subset.uns['elements']=selected_elements

# subset to desired genes
rna_subset = mdata_subset.mod['rna'][:,selected_genes].copy()
mdata = md.MuData({'rna':rna_subset, 'grna':grna_subset})
mdata


# check to make sure we properly subset the element labels in the .uns field

# In[10]:

assert len(grna_subset.uns['elements']) == mdata['grna'].varm['element_targeted'].shape[1]

# ## Register data with perturbo

# In[11]:


perturbo.PERTURBO.setup_mudata(
    mdata,
    batch_key="prep_batch",
    library_size_key="total_umis",
    continuous_covariates_keys=["percent.mito", "log1p_guide_count"],
    perturb_by_element_key="element_targeted",
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

# In[12]:


# requires graphviz to be installed
model.render_model()


# In[13]:


model.render_guide()


# In[ ]:





# ### Train the model

# In[14]:


# burn-in
pyro.clear_param_store()
# plan_kwargs = {'optim': pyro.optim.ClippedAdam(dict(lr=0.1, lrd=0.96))}
# model.train(20, lr=0.05, batch_size=512, use_gpu=use_gpu)
# model.train(20, lr=0.05, batch_size=512, use_gpu=use_gpu)


# In[15]:


## fine-tune
# plan_kwargs = {'optim': pyro.optim.ClippedAdam(dict(lr=0.1, lrd=0.99))}
# model.train(20, lr=0.1, batch_size=1024, use_gpu=use_gpu)
# model.train(100, lr=0.03, batch_size=1024, use_gpu=use_gpu)
pyro.get_param_store().keys()


# In[16]:


for k, v in pyro.get_param_store().items():
    if (~torch.isreal(v)).any():
        print(k)


# In[17]:


model.train(20, lr=0.01, batch_size=batch_size, accelerator=accelerator) 


# In[18]:


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
        pd.DataFrame(data=mat, index=mdata['grna'].uns['elements'], columns=mdata['rna'].var['gene_name'])
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


# In[21]:


# model.save("gasperini_atscale_model.save", overwrite=True)


# In[22]:


for k, v in pyro.get_param_store().items():
    print(k, v.shape)
    if sum(v.shape) <= 5:
        print(v)

