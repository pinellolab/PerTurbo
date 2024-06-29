from typing import Optional, List

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, eye, coo_matrix, lil_matrix, vstack, hstack
from scipy.sparse import random as sparse_random

import mudata as md
from mudata import MuData
import perturbo
from ..model._module import LogNormalNegativeBinomial

import matplotlib.pyplot as plt
import seaborn as sns

import pyro.distributions as dist
import scipy.stats as stats
from scipy.stats import lognorm
from scipy.optimize import curve_fit
import statsmodels.api as sm

import time
import warnings
import os
warnings.filterwarnings("ignore")

import pyro
import torch
import scvi


class Fit_PerTurbo():  # keep consistent with perturbo / pyro
    def __init__(
        self, 
        mdata: Optional[MuData] = None, 
        batch_key: Optional[str] = None,
        library_size_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        read_depth_key: Optional[str] = None,
        Size_Factor_key: Optional[str] = None,
        continuous_covariates_keys: Optional[List[str]] = None,
        obs_continuous_covariates_keys: Optional[List[str]] = None,
        gene_by_element_key: Optional[str] = None,
        guide_by_element_key: Optional[str] = None,
        rna_element_uns_key: Optional[str] = None,
        guide_element_uns_key: Optional[str] = None,
        modalities: Optional[dict[str, str]] = None
    ):
        """
        Fit an example MuData with PerTurbo
        
        Parameters
        -----------
        mdata
            The example MuData.
        batch_key
            Key within the RNA AnnData .obs corresponding to the experimental batch
        library_size_key
            .obs key within the RNA AnnData that can be directly observed, and can be used for generatig lisbrary size
        size_factor_key
            .obs key within the RNA AnnData object containing library size factors (log) for each sample
        read_depth_key
            .obs key within the RNA AnnData object that can be directly observed, for computing read depth Size Factor
        Size_Factor_key
            .obs key within the RNA AnnData object containing read depth Size Factor (log) for each sample
        continuous_covariates_keys
            list of .obs keys within the RNA AnnData object containing other continuous covariates to be "regressed out"
        obs_continuous_covariates_keys
            list of .obs keys within the RNA AnnData object that can be directly observed, corresponding to the continuous covariates        gene_by_element_key
            .varm key within the RNA AnnData object containing a mask of which genes can be affected by which genetic elements
        guide_by_element_key
            .varm key within the perturbation AnnData object containing which perturbations target which genetic elements
        rna_element_uns_key
            .uns key within the RNA AnnData object containing names of perturbed elements (if using GENE_BY_ELEMENT_KEY),
            otherwise automatically inferred from column names if .varm object is a DataFrame
        guide_element_uns_key
            .uns key within the perturbation AnnData object containing names of perturbed elements
            (if using GUIDE_BY_ELEMENT_KEY), otherwise automatically inferred from column names if .varm object is a DataFrame
        modalities
            A dict containing these same setup argument
        """
        
        self.mdata = mdata
        self.batch_key = batch_key
        self.library_size_key = library_size_key
        self.size_factor_key = size_factor_key
        self.read_depth_key = read_depth_key
        self.Size_Factor_key = Size_Factor_key
        self.continuous_covariates_keys = continuous_covariates_keys
        self.obs_continuous_covariates_keys = obs_continuous_covariates_keys
        self.gene_by_element_key = gene_by_element_key
        self.guide_by_element_key = guide_by_element_key
        self.rna_element_uns_key = rna_element_uns_key
        self.guide_element_uns_key = guide_element_uns_key
        self.modalities = modalities
        
        
    def mudata_selection(
        self,
        mdata: Optional[MuData] = None,
        selected_genes: Optional[List[str]] = None,
        selected_guides: Optional[List[str]] = None,
        selected_guide_category: Optional[List[str]] = None,
        selected_cells: Optional[List[str]] = None,
        selected_cell_threshold: Optional[int] = 100,
        only_targeted_genes: Optional[bool] = True
    ):
        """
        Filter MuData by selecting genes, guides, cells.
        
        Parameters
        -----------
        mdata
            a MuData object, of no object is registered
        selected_genes
            A list of gene names
        selected_guides
            A list of guide names
        selected_guide_category (alternative for selected_guides)
            A type of guide to be selected, e.g., "TSS", "NTC", etc.
        selected_cells
            A list of cell IDs
        selected_cell_threshold (alternative for selected_cells)
            A value, minimum number of "umi_counts" of a cell for it to be selected, i.e., selected cell have umi_count > threshold.
        only_targeted_genes
            Whether we want to select only genes that has a matched guide, or all the genes indicated by us.
        
        Output
        -----------
        mdata_train
            A MuData that is suitable for training, after selecting guides, genes, cells.
        """
        if mdata == None:
            mdata = self.mdata

        # define list of selected genes (initalize)
        if selected_genes is None:
            selected_genes = mdata["rna"].var_names
            
        # define list of selected guides
        if selected_guides is None and selected_guide_category is None:
            selected_guides = mdata["grna"].var_names
        elif selected_guide_category is not None:
            categories_str = str(tuple(selected_guide_category))  # conver the list to a tuple string
            if len(selected_guide_category) == 1:
                categories_str = f"('{selected_guide_category[0]}')"
            query_string = f'Category in {categories_str}'
            selected_guides = mdata['grna'].var.query(query_string).index
        
        # define list of selected cells
        if selected_cells is None and selected_cell_threshold is None:
            selected_cells = mdata["rna"].index
        elif selected_cell_threshold is not None:
            selected_cells = mdata["rna"].obs["umi_count"] > selected_cell_threshold
            
        mdata_subset = mdata.copy()
        
        # subset to desired grnas & corresponding elements
        grna_subset = mdata_subset.mod['grna'][:,selected_guides].copy()
        selected_element_idx = np.where(grna_subset.varm['element_targeted'].sum(axis=0)>0)[1]
        selected_elements = grna_subset.uns['elements'][selected_element_idx]

        grna_subset.varm['element_targeted'] = grna_subset.varm['element_targeted'][:,selected_element_idx]
        grna_subset.uns['elements'] = selected_elements

        # subset to desired genes
        rna_subset = mdata_subset.mod['rna'][:,selected_genes].copy()
        rna_subset.uns['elements'] = selected_elements
        rna_subset.varm['element_tested'] = rna_subset.varm['element_tested'][:,selected_element_idx]

        # select genes again (if we only want the genes matched with a selected guide)
        if only_targeted_genes is True:
            element_tested_csr = rna_subset.varm['element_tested']
            selected_genes_idx = element_tested_csr.getnnz(axis=1) > 0
            selected_genes = selected_genes[selected_genes_idx]

            #targeted_genes = list({x.split("_")[0] for x in mdata_subset["grna"].var_names if "TSS" in x})

            rna_subset = rna_subset[:, selected_genes]
            rna_subset.varm['element_tested'] = element_tested_csr[selected_genes_idx]
            
        # create new MuData for selected guides, genes
        mdata_train = md.MuData({'rna':rna_subset, 'grna':grna_subset})
        
        # select only cells with enough umi counts
        selected_cells = mdata_train['rna'].obs['umi_count'] > selected_cell_threshold
        mdata_train = mdata_train[selected_cells, :].copy()

        self.mdata_train = mdata_train
        return(mdata_train)

    def fit_obs(
        self,
    ):
        """Output: params_for_simulation, a dataframe, saving lognormal parameters for each obs."""       
        params_for_simulation = pd.DataFrame()
        obs_param_keys = []
        
        for obs_key in self.obs_continuous_covariates_keys:
            obs_param_key = "params_" + obs_key.replace('.', '_')
            obs_param = lognorm.fit(self.mdata_train.mod["rna"].obs[obs_key])
            
            params_for_simulation[obs_param_key] = obs_param
            obs_param_keys = obs_param_keys + [obs_param_key]
        
        self.obs_param_keys = obs_param_keys
        self.params_for_simulation = params_for_simulation
        return(params_for_simulation)
            
    def get_n_steps(
        self,
        max_steps: Optional[int] = 400,
    ):
        """Get number of training steps according to sample size. training steps decrease with increasing sample size. """
        n_steps = min(max_steps, round(max_steps * (20000/self.mdata_train["rna"].X.shape[0])))  # if ncells > 20000 then n_steps decay
        n_steps = max(n_steps, 1)
        
        self.n_steps = n_steps
        return(n_steps)
    
    def get_model(
        self,
        likelihood: Optional[str] = None
    ):
        """Get a model object (as in pyro) that could be trained."""
        # register data with perturbo
        perturbo.model.PERTURBO.setup_mudata(
            self.mdata_train,
            batch_key = self.batch_key,
            library_size_key = self.library_size_key,
            continuous_covariates_keys = self.continuous_covariates_keys,
            gene_by_element_key = self.gene_by_element_key, # <------------ gene * element pairs
            guide_by_element_key = self.guide_by_element_key,
            guide_element_uns_key = self.guide_element_uns_key ,  # <------------ list of elements tested here
            rna_element_uns_key = self.rna_element_uns_key,    # <------------ what is this?
            modalities = self.modalities,
        )
        model = perturbo.model.PERTURBO(self.mdata_train, likelihood=likelihood, n_factors=None)
        
        self.model = model
        return(model)

    def train_perturbo(  # mimic the train in pyro
        self,
        n_steps: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        accelerator: Optional[str] = None
    ):
        if n_steps is None:
            n_steps = self.get_n_steps()
            self.n_steps = n_steps
        if lr is None:
            lr = self.lr
        if batch_size is None:
            batch_size = self.batch_size
        if accelerator is None:
            accelerator = self.accelerator
        
        # train the model
        pyro.clear_param_store()
        if len((pyro.get_param_store().keys())) == 0:
            print("param_store is clean. prepared for training.")

        print(f"Training data with {n_steps} steps, using {lr} learning rate, {batch_size} batch size, on {accelerator}.")

        self.model.train(
            max_epochs = n_steps, 
            lr = lr,
            batch_size = batch_size,
            accelerator = accelerator
        )
        
    def extract_estimation(
        self,
        model = None,
        param_keys: Optional[List[str]] = None,
        estimator_type: Optional[str] = None,
        df_dir_base: Optional[str] = None,
        mdata_name: Optional[str] = None
    ):
        """
        Extract the desired estimation output.
        
        Parameters
        -----------
        model
            The model just trained by PerTurbo
        param_keys
            A list of parameters to save. Can be selected from the param_store output.
        estimator_type
            Whether we want MAP estimation or the whole posterior estimation. 
            e.g. "map" / "posterior"
        df_dir_base 
            Path, to which we extract the estimated parameters, by requirement. Will add the dataset and estimation method to it.
        mdata_name
            Give the data for simulation a name, which will be used to identify estimated param df.
        Output
        -----------
        dfs
            A list of dataframes, with param_keys be the keys. (and multiple .csv files in the initialized folder)
        """        
        if model == None:
            model = self.model
        # Create a dictionary to store DataFrames for each parameter
        dfs = {}

        if estimator_type == "map":
            for k, v in model.module.guide.median().items():
                if k in param_keys:                # only extract the parameters we need
                    arr = v.cpu().detach().numpy() # Convert tensor to numpy array
                    dfs[k] = pd.DataFrame(arr)     # Convert numpy array to DataFrame
            
        elif estimator_type == "posterior":
            param_store = pyro.get_param_store()
            for k, v in param_store.items():
                if k in param_keys:                # only extract the parameters we need
                    arr = v.cpu().detach().numpy() # Convert tensor to numpy array
                    dfs[k] = pd.DataFrame(arr)     # Convert numpy array to DataFrame
        else:
            print("Please choose a correct estimator type from (map/posterior).")
        
        # add two additional estimation dataframes
        dfs["element_effects_res"] = model.get_element_effects()
        dfs["params_for_simulation"] = self.params_for_simulation
        
        self.dfs = dfs
        
        # create folder for saving the estimations
        df_dir = df_dir_base + "_" + f"{mdata_name}" + "_" + f'{estimator_type}'
        
        # extract the estimation
        if not os.path.exists(df_dir):
            os.makedirs(df_dir)
            
        for name, df in dfs.items():
            filename = name.replace('.', '_') + '.csv'  # Create a valid filename from the parameter name
            filepath = os.path.join(df_dir, filename)   # Join the directory with the filename

            df.to_csv(filepath, index=False)            # Save DataFrame to csv
        
        return(dfs)

    def plot_obs(
        self,
    ):
        """plot histograms of observable var from .obs"""
        n_plots = len(self.obs_continuous_covariates_keys)
        
        # Create n_plots number of subplots
        fig, axes = plt.subplots(1, n_plots, figsize = (12, 5))
        
        i = 0
        for obs_key in self.obs_continuous_covariates_keys:
            data = self.mdata_train['rna'].obs[obs_key]  # Your data for the histogram
            axes[i].hist(data, bins=50, color="grey", edgecolor="black", density=True)  # Notice density=True for normalization
            
            # Retrieve fitted parameters
            obs_param_key = "params_" + obs_key.replace('.', '_')
            shape, loc, scale = self.params_for_simulation[obs_param_key]
            
            # Define a range for x values
            xmin, xmax = axes[i].get_xlim()  # Get the limits from the histogram
            x = np.linspace(xmin, xmax, 100)
            
            # Calculate the PDF using the fitted parameters
            pdf = lognorm.pdf(x, s=shape, loc=loc, scale=scale)
            
            # Plot the density curve on the same axes
            axes[i].plot(x, pdf, 'r-', label='Log-normal fit', color='royalblue')
            axes[i].set_xlabel(f"{obs_key}")
            axes[i].set_ylabel("Density")
            
            # Optional: Add a legend
            axes[i].legend()
            
            i += 1
            
        # Display the plot
        plt.tight_layout()
        plt.show()

class Simulate_Data():
    def __init__(
        self,
        df_dir_base: Optional[str] = "from_real_data",
        mdata_name: Optional[str] = None,
        estimator_type: Optional[str] = None,
        library_size_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        read_depth_key: Optional[str] = None,
        Size_Factor_key: Optional[str] = None,
        continuous_covariates_keys: Optional[List[str]] = None,
        obs_continuous_covariates_keys: Optional[List[str]] = None,
    ):
        """
        Extract the desired estimation output.
        
        Parameters
        -----------
        df_dir_base 
            Path, to which we extract the estimated parameters, by requirement. Will add the dataset and estimation method to it.
        mdata_name
            A given name of the dataset we are simulating from. Point to the directory where we save the estimated parameters
        estimator_type
            A given name of the type of estimator we are using (map/posterior). Point to the directory where we save the estimated parameters
        library_size_key
            .obs key within the RNA AnnData that can be directly observed, and can be used for generatig lisbrary size
        size_factor_key
            .obs key within the RNA AnnData object containing library size factors (log) for each sample
        read_depth_key
            .obs key within the RNA AnnData object that can be directly observed, for computing read depth Size Factor
        Size_Factor_key
            .obs key within the RNA AnnData object containing read depth Size Factor (log) for each sample
        continuous_covariates_keys
            list of .obs keys within the RNA AnnData object containing other continuous covariates to be "regressed out"
        obs_continuous_covariates_keys
            list of .obs keys within the RNA AnnData object that can be directly observed, corresponding to the continuous covariates        
        gene_by_element_key
            .varm key within the RNA AnnData object containing a mask of which genes can be affected by which genetic elements

        """
        self.df_dir_base = df_dir_base
        self.mdata_name = mdata_name
        self.estimator_type = estimator_type
        self.library_size_key = library_size_key
        self.size_factor_key = size_factor_key
        self.read_depth_key = read_depth_key
        self.Size_Factor_key = Size_Factor_key
        self.continuous_covariates_keys = continuous_covariates_keys
        self.obs_continuous_covariates_keys = obs_continuous_covariates_keys
 
        # run initial functions
        self._read_params()
        self._assign_values()

        
    def _read_params(self):
        """
        Read in the estimated parameters for later simulation.
        """
        df_dir = self.df_dir_base + "_" + f"{self.mdata_name}" + "_" + f'{self.estimator_type}'
        all_files = [f for f in os.listdir(df_dir) if f.endswith('.csv')]

        dfs = {}
        for csv_file in all_files:
            df_name = os.path.splitext(csv_file)[0]  # Get filename without extension
            dfs[df_name] = pd.read_csv(os.path.join(df_dir, csv_file))
        df = dfs["params_for_simulation"]
        
        self.dfs = dfs  # list of dataframes of all parameters
        self.df = df    # parameters for obs
    
    def _assign_values(self):
        """"Assign values like total number genes from the estimated parameters, which have just been read in."""
        dfs = self.dfs
        
        total_genes = dfs["log_gene_mean"].shape[0]

        self.total_genes = total_genes

    def sample_mudata(
        self,
        simulate_distribution: Optional[str] = "lnnb",
        ncells: Optional[int] = 200000,
        ngenes: Optional[int] = 100,
        nguides_pos: Optional[int] = None,           # number of guides targeting elements that affect genes
        nguides_per_element: Optional[int] = 4, # number of guides per element (that affects a gene)
        nguides_ntc: Optional[int] = 100,           # number of control guides
        ncells_per_guide: Optional[int] = 20,   # number of cells each positive guide is detected in
        guide_efficacy_values: Optional[List[float]] = [1.0,1.0,1.0,1.0],
        guide_category: Optional[List[str]] = ["positive_control", "negative_control"],
        log2_fold_change: Optional[float] = 1.0,
        mean_reads_per_gene: Optional[float] = 1.0,
        read_depth: Optional[int] = None,
        chunk_size: Optional[int] = 10000,
        ):
        """Combine everthing we have together, generate synthetic MuData that contain enough information for training"""

        nelements_pos = ngenes
        if nguides_pos is None:
            nguides_pos = nguides_per_element * nelements_pos
        if read_depth is None and mean_reads_per_gene is not None:
            read_depth = mean_reads_per_gene * ngenes
        elif mean_reads_per_gene is None and read_depth is not None:
            mean_reads_per_gene = read_depth / ngenes
        
        self.simulate_distribution = simulate_distribution
        self.ncells = ncells
        self.ngenes = ngenes
        self.nguides_pos = nguides_pos
        self.nguides_per_element = nguides_per_element
        self.nguides_ntc = nguides_ntc
        self.ncells_per_guide = ncells_per_guide
        self.guide_efficacy_values = guide_efficacy_values
        self.guide_category = guide_category
        self.log2_fold_change = log2_fold_change
        self.read_depth = read_depth
        self.mean_reads_per_gene = mean_reads_per_gene
        self.chunk_size = chunk_size

        nelements_ntc = nguides_ntc // nguides_per_element
        pert_rate = ncells_per_guide / ncells
        lfc = np.log(2**(log2_fold_change))

        self.nelements_pos = nelements_pos
        self.nelements_ntc = nelements_ntc
        self.nelements = nelements_pos + nelements_ntc
        self.nguides = nguides_pos + nguides_ntc
        self.pert_rate = pert_rate
        self.lfc = lfc

        # set names for guides, genes, elements
        self._set_names()

        # grna modality
        self._sample_grna()
        self._get_element_targeted()
        self._get_element_targeted_uns()

        grna_modality = md.AnnData(self.grna_data, 
                                   uns={"elements": np.array(self.element_names)})
        grna_modality.var_names = self.guide_names
        grna_modality.varm['element_targeted'] = self.element_targeted
        
        grna_modality.uns['element_targeted'] = self.element_targeted_df
        grna_modality.uns['guide_efficacy'] = self.guide_efficacy_values

        # rna modality
        self._sample_obs()
        self._get_element_tested()
        self._get_element_tested_uns()
        self._get_logits()
        self._get_total_count()
        self._get_multiplicative_noise()
        self._get_logits_perturb()
        #self._get_dispersion_from_curve()
        self._sample_rna()        

        rna_modality = md.AnnData(self.rna_sparse, 
                                  obs = self.obs, 
                                  uns={"elements": np.array(self.element_names),
                                       "fold_change": round(np.exp(lfc), 2)})
        rna_modality.obs["total_umis"] = rna_modality.X.sum(axis=1) # add the library size obs
        log_cpm = np.log(rna_modality.obs["total_umis"] / 1e6)
        rna_modality.obs["Size_Factor"] = log_cpm - np.mean(log_cpm)
        
        rna_modality.var["gene_mean"] = (np.exp(self.logits_corrected) * self.total_count).mean(axis=0) 
        rna_modality.var["gene_total_count"] = self.total_count.squeeze()
        rna_modality.var["gene_mean_perturbed"] = (np.exp(self.logits_perturb) * self.total_count).mean(axis=0) 

        rna_modality.var_names = self.gene_names
        rna_modality.varm['element_tested'] = self.element_tested.transpose()
        
        rna_modality.uns['element_tested'] = self.element_tested_df
        
        # Construct mudata
        mdata = md.MuData({"rna": rna_modality, "grna": grna_modality})
        
        simulated_read_depth = mdata["rna"].obs["total_umis"]
        print(f"when setting the read depth per cell as {read_depth}, simulated data has an average of {simulated_read_depth.mean()} reads per cell.")
        
        self.mdata = mdata
        return(mdata)
    
    def extract_to_h5mu(self):
        mdata = self.mdata

        mudata_path = "simulated_data" 
        if not os.path.exists(mudata_path):
            os.makedirs(mudata_path)
        
        if "negative_control" in self.guide_category:
            mudata_train_name = f"/{self.simulate_distribution}_SimuData_Ncells{self.ncells}_PertRate{self.pert_rate}_Ngenes{self.ngenes}_Nguides{self.nguides_per_element}_LFC{abs(self.log2_fold_change)}_ReadDepth{self.read_depth}_ControlGuides{self.nguides_ntc}.h5mu"
        else:
            mudata_train_name = f"/{self.simulate_distribution}_SimuData_Ncells{self.ncells}_PertRate{self.pert_rate}_Ngenes{self.ngenes}_Nguides{self.nguides_per_element}_LFC{abs(self.log2_fold_change)}_ReadDepth{self.read_depth}.h5mu"
        mdata.write(mudata_path + mudata_train_name)
    
    def _set_names(self):
        """Set gene names, guide names, and element names. Distinguish between positive guides and non-targeting guides"""        
        # gene names
        self.gene_names = ["gene" + str(i) for i in range(0, self.ngenes)]

        # element names
        self.element_names_pos = []
        self.element_names_ntc = []
        if "positive_control" in self.guide_category:
            self.element_names_pos = self.gene_names
        if "negative_control" in self.guide_category:
            self.element_names_ntc = ["ntc" + str(i) for i in range(0, self.nelements_ntc)]
        self.element_names = self.element_names_pos + self.element_names_ntc
            
        # guide names
        self.guide_names_pos = [name + "_g" + str(k) for name in self.element_names_pos for k in range(0, self.nguides_per_element)]
        self.guide_names_ntc = ["ntc_g" + str(j) for j in range(self.nguides_ntc)]
        self.guide_names = self.guide_names_pos + self.guide_names_ntc
            
    def _sample_obs(self):
        """Generate for .obs for RNA modality. A dataframe, each column is a obs, with ncells rows."""
        df = self.df
        ncells = self.ncells
        obs_continuous_covariates_keys = self.obs_continuous_covariates_keys
        library_size_key = self.library_size_key
        size_factor_key = self.size_factor_key

        obs = pd.DataFrame()
        
        # batch number
        prep_batch = np.full(ncells, "batch_1")
        obs["prep_batch"] = prep_batch
        
        # other obs
        for obs_key in obs_continuous_covariates_keys:
            obs_param_key = "params_" + obs_key.replace('.', '_')  # get the key for params_for_simulation
            obs[obs_key] = lognorm.rvs(size = ncells,
                                       s = df[obs_param_key][0], loc = df[obs_param_key][1], scale = df[obs_param_key][2])

        # compute "size_factor" from library_size (e.g. "umi_count")
        if library_size_key in obs_continuous_covariates_keys:
            log_cpm = np.log(obs[library_size_key].values / 1e6)
            obs[size_factor_key] = (log_cpm - np.mean(log_cpm)).reshape(-1,1)
        
        self.obs = obs
        
    def _sample_grna(self):
        """
        Generate grna.X: self.grna_data, a sparse matrix, with ncell rows and nguides columns
        The number of positive guides in each cell follows number of guides per cell computed from a pre-defined number of cells per guide,
        the number of ntc guides in each cell should also follow the previous computed number of guides per cell, to mimic the real functional guides.
        """
        ncells = self.ncells
        nguides = self.nguides
        pert_rate = self.pert_rate

        grna_data = sparse_random(ncells, nguides, density = pert_rate, data_rvs=lambda n: np.ones(n), format='csr')
        self.grna_data = grna_data

    def _get_element_targeted(self):
        """Generate grna.varm["element_targeted"]. self.element_targeted, a sparse matrix, with nguides rows, nelement cols, indicate which element is targeted by which guide."""
        nguides = self.nguides
        nelements = self.nelements
        nguides_per_element = self.nguides_per_element

        element_targeted_dense = np.zeros((nguides, nelements))

        # Fill the specific entries with 1s
        for j in range(nelements):
            start_row = nguides_per_element * j
            end_row = start_row + nguides_per_element
            if end_row > nguides:
                break  # break the loop if the end_row exceeds the matrix size
            element_targeted_dense[start_row:end_row, j] = 1

        # Convert the dense matrix to a sparse matrix (CSR format)
        element_targeted = csr_matrix(element_targeted_dense)
        
        self.element_targeted = element_targeted
        
    def _get_element_targeted_uns(self):
        """Generate grna.uns["element_targeted"]. self.element_targeted_df, a dataframe, with 2 columns | grna | element |. save guide name and its targeting element."""
        element_names_pos = self.element_names_pos
        guide_names_pos = self.guide_names_pos
        nguides_per_element = self.nguides_per_element

        element_names_pos_extend = [element for element in element_names_pos for _ in range(nguides_per_element)]
        element_targeted_df = pd.DataFrame({"grna": guide_names_pos, 
                                            "element": element_names_pos_extend})
        
        self.element_targeted_df = element_targeted_df
        
    def _get_element_tested(self):
        """
        Generate rna.varm["element_tested"]. self.element_tested, a sparse matrix indicating potentially related elements and genes, with nelement rows, ngenes cols.
        In "positive_control" case, it is an identity matrix, one element is only targeting one gene.
        In "negative_control" case, it is a matrix full of 1. Studying as much negative pairs as possible creates a more realistic negative control distribution.
        """
        guide_category = self.guide_category
        ngenes = self.ngenes
        gene_names = self.gene_names
        element_names_pos = self.element_names_pos
        nelements_ntc = self.nelements_ntc

        # for positive control elements
        element_tested_pos = eye(ngenes, dtype="float32").tocsr()
        
        # add ntc elements (if there are any)            
        if "negative_control" in guide_category:
            element_tested_ntc_dense = np.ones((nelements_ntc, ngenes))
            element_tested_ntc = csr_matrix(element_tested_ntc_dense)
            
        element_tested = vstack([element_tested_pos, element_tested_ntc])

        self.element_tested = element_tested
        self.element_tested_pos = element_tested_pos
        self.element_tested_ntc = element_tested_ntc

    def _get_element_tested_uns(self):
        """Generate rna.uns["element_tested"]. self.element_tested_df, a dataframe, with 2 columns | element | gene |. save element name and its targeting gene."""
        element_names_pos = self.element_names_pos
        gene_names = self.gene_names

        element_tested_df = pd.DataFrame({
            "element": element_names_pos, 
            "gene": gene_names})
        self.element_tested_df = element_tested_df

                
    def _get_total_perturbation_effect(self):
        """
        Get matrix of the effect of each guide on each gene according to guide_efficacy and element_effects.
        Output: self.total_perturbation_effect. An np.array, with nguides rows and ngenes cols
        """
        lfc = self.lfc
        guide_category = self.guide_category
        guide_efficacy_values = self.guide_efficacy_values
        nelements = self.nelements
        nelements_pos = self.nelements_pos
        nelements_ntc = self.nelements_ntc
        nguides_ntc = self.nguides_ntc
        element_targeted = self.element_targeted
        element_tested = self.element_tested

        # for positive guides
        guide_efficacy_long_pos_mean = guide_efficacy_values * nelements_pos  # list of len nguides_pos, saving mean efficacy of each positive guide
        guide_efficacy_long_pos = np.random.binomial(n=1, p=guide_efficacy_long_pos_mean).tolist()  # list of len nguides_pos, saving efficacy generated from the means.

        # if there are negative control guides
        if "negative_control" in guide_category:
            guide_efficacy_long_ntc = [0] * nguides_ntc  # list of nguides_ntc 0s. negative control guides having no efficacy, will not effect logit.

        guide_efficacy_long = guide_efficacy_long_pos + guide_efficacy_long_ntc

        # expand from (nguides, 1) list to (nguides, nelements) matrix
        guide_efficacy_values = torch.tensor(guide_efficacy_long).reshape(-1,1).expand(-1, nelements) 
        guide_efficacy = guide_efficacy_values * (element_targeted).toarray()  # shape (nguides, nelements)
        
        # element_effects
        element_effects = lfc * element_tested  # shape (nelements, ngenes)
        
        # total_perturbation_effect: apply guide efficacy to 
        total_perturbation_effect = guide_efficacy @ element_effects.toarray()  # shape (nguides_total, ngenes)
        self.total_perturbation_effect = total_perturbation_effect

        return(total_perturbation_effect)

    def _get_random_indices(self):
        """Get random indices of genes to simulate from. An np.array, with length ngenes."""
        total_genes = self.total_genes
        ngenes = self.ngenes

        gene_ids = np.random.choice(range(total_genes), size = ngenes, replace = True)
        self.gene_ids = gene_ids

        return(gene_ids)
        
    def _get_logits(self):
        """Get the raw logits matrix. A torch.tensor, with ncells rows, ngenes cols."""
        size_factor_key = self.size_factor_key
        continuous_covariates_keys = self.continuous_covariates_keys
        dfs = self.dfs
        gene_ids = self._get_random_indices()
        obs = self.obs
        simulate_distribution = self.simulate_distribution

        # put size_factor in the last position of continuous covariates
        if size_factor_key not in continuous_covariates_keys:
            continuous_covariates_keys = continuous_covariates_keys + [size_factor_key]
        
        # Get covariate effects
        cov_effect_sizes = {}
        for cov_key in continuous_covariates_keys:
            if cov_key == "Size_Factor":  # skip this, since this is related to the total_umis, and will later modeled by correction_term
                continue
            idx = continuous_covariates_keys.index(cov_key)
            cov_effect = dfs["cont_covariate_effect"].iloc[idx, gene_ids].values
            cov_effect_size = obs[cov_key].values.reshape(-1, 1) @ cov_effect.reshape(1, -1)  # torch, ncells * ngenes
            
            cov_effect_sizes[cov_key] = cov_effect_size
            
        # samples_log_gene_mean (directly choose from the original mean with the given index)
        samples_log_gene_mean = dfs["log_gene_mean"].loc[gene_ids,].values.reshape(1,-1)

        # samples_log_gene_dispersion (directly choose from the original mean with the given index)
        samples_log_gene_dispersion = dfs["log_gene_dispersion"].loc[gene_ids,].values.reshape(1,-1)       

        # mutiplicative noise
        samples_multiplicative_noise = dfs["multiplicative_noise"].loc[gene_ids,].values.reshape(1,-1)

        # logit
        if simulate_distribution == "nb":
            logits = torch.from_numpy(samples_log_gene_mean +  # base mean
                                      obs["size_factor"].values.reshape(-1,1) +    # size factor
                                      sum(cov_effect_sizes.values()) -  # covariate effect sizes
                                      samples_log_gene_dispersion)  # torch.tensor, length = ngenes  # torch.tensor, length = ngenes
        elif simulate_distribution == "lnnb":
            logits = torch.from_numpy(samples_log_gene_mean +  # base mean
                                      obs["size_factor"].values.reshape(-1,1) +    # size factor
                                      sum(cov_effect_sizes.values()) -  # covariate effect sizes
                                      samples_log_gene_dispersion -
                                      samples_multiplicative_noise**2 / 2)  # torch.tensor, length = ngenes
        
        self.samples_log_gene_mean = samples_log_gene_mean
        self.samples_log_gene_dispersion = samples_log_gene_dispersion
        self.samples_multiplicative_noise = samples_multiplicative_noise

        self.logits = logits
        
    def _get_total_count(self):
        """Get self.total count. A torch.tensor, with 1 row, ngenes columns"""
        samples_log_gene_dispersion = self.samples_log_gene_dispersion

        total_count = torch.from_numpy(np.exp(samples_log_gene_dispersion))  # torch.tensor, length = ngenes

        self.total_count = total_count

    def _get_multiplicative_noise(self):
        """Get self.multiplicative_noise. A torch.tensor, with 1 row, ngenes columns"""
        samples_multiplicative_noise = self.samples_multiplicative_noise

        multiplicative_noise = torch.from_numpy(samples_multiplicative_noise)

        self.multiplicative_noise = multiplicative_noise
    
    def _get_correction_term(self):
        """A cell specific shift value of gene mean. A np.array, length ncells."""
        read_depth = self.read_depth
        ngenes = self.ngenes
        grna_data = self.grna_data
        simulate_distribution = self.simulate_distribution
        
        total_perturbation_effect = self._get_total_perturbation_effect()
        logits = self.logits
        total_count = self.total_count
        multiplicative_noise = self.multiplicative_noise
        
        log_pert_effect = grna_data @ total_perturbation_effect  # the perturbation matrix added to log_mean

        if simulate_distribution == "nb":
            orig_means = (np.exp(logits) * total_count).detach().cpu().numpy()  # array, shape = (ncells, ngenes)
        elif simulate_distribution == "lnnb":
            orig_means = (np.exp(logits + multiplicative_noise**2 / 2) * total_count).detach().cpu().numpy()  # array, shape = (ncells, ngenes)
        orig_means_pert = orig_means * np.exp(log_pert_effect)

        correction_term = (read_depth) / (orig_means_pert.sum(axis=1))  # array, length = ncells
        correction_term = correction_term.reshape(-1,1)  # this correction term already account for Size_Factor in simulation (derived from total UMIs)

        self.orig_means = orig_means
        self.correction_term = correction_term
        self.log_pert_effect = log_pert_effect

        return(correction_term)
    
    def _get_logits_corrected(self):
        """Get the logits after correction. A torch.tensor, with ncell rows, ngene columns"""
        correction_term = self._get_correction_term()
        logits = self.logits

        logits_corrected = logits + np.log(correction_term)
        self.logits_corrected = logits_corrected

        return(logits_corrected)

    # def _mean_dispersion_curve(self, x, a, b, c):
    #     return (a + b * x + c * x**2)
        
    # def _get_dispersion_from_curve(self):
    #     orig_means = self.orig_means
    #     total_count = self.total_count
    #     logits_corrected = self.logits_corrected

    #     orig_means = orig_means.mean(axis=0)  # compute mean expressiong level of each gene
    #     total_count = total_count.detach().cpu().numpy()
    #     orig_dispersions = np.log(total_count)
    #     print(f"orig_means {orig_means}")
    #     print(f"orig_dispersions {orig_dispersions}")

    #     # Curve fitting
    #     params, params_covariance = curve_fit(self._mean_dispersion_curve, orig_means.flatten(), orig_dispersions.flatten())
    #     print(params, params_covariance)

    #     # Use fitted parameters to predict
    #     dispersions_corrected = self._mean_dispersion_curve(logits_corrected.detach().cpu().numpy().flatten(), *params)
    #     dispersions_corrected = dispersions_corrected.reshape(logits_corrected.shape)

    #     total_count_corrected = np.exp(dispersions_corrected)
    #     total_count_corrected = torch.from_numpy(total_count_corrected)

    #     self.total_count_corrected = total_count_corrected

    #     print(total_count)
    #     print(total_count_corrected)

    def _get_logits_perturb(self):
        """Get the logits after perturbation. A torch.tensor, with ncell rows, ngene columns"""
        logits_corrected = self._get_logits_corrected()
        log_pert_effect = self.log_pert_effect

        logits_perturb = logits_corrected + log_pert_effect
        self.logits_perturb = logits_perturb
        
    def _sample_rna(self):
        """Generate rna.X. A sparse_matrix, with ncells rows, ngene columns"""
        simulate_distribution = self.simulate_distribution
        ncells = self.ncells
        ngenes = self.ngenes
        chunk_size = self.chunk_size
        total_count = self.total_count
        logits_perturb = self.logits_perturb
        multiplicative_noise = self.multiplicative_noise

        start_time = time.time()
        chunks = []
        i = 1
        print(f"The data will be simulated with {round(ncells / chunk_size)} chunks.")
        
        if simulate_distribution == "nb":
            print("simulate from NB")
            for start_row in range(0, ncells, chunk_size):
                print(f"Simulate chunk {i}.")
                end_row = min(start_row + chunk_size, ncells)
                logits_perturb_chunk = logits_perturb[start_row:end_row, :]
                #total_count_corrected_chunk = total_count_corrected[start_row:end_row, :]
                simulate_chunk = dist.NegativeBinomial(logits=logits_perturb_chunk, 
                                                       total_count=total_count)
                
                data_simu_chunk = csr_matrix(simulate_chunk.sample())
                chunks.append(data_simu_chunk)

                i = i + 1
                
        elif simulate_distribution == "lnnb":
            print("simulate from LNNB")
            for start_row in range(0, ncells, chunk_size):
                print(f"Simulate chunk {i}.")
                end_row = min(start_row + chunk_size, ncells)
                logits_perturb_chunk = logits_perturb[start_row:end_row, :]
                #total_count_corrected_chunk = total_count_corrected[start_row:end_row, :]
                simulate_chunk = LogNormalNegativeBinomial(logits=logits_perturb_chunk, 
                                                           total_count=total_count,
                                                           multiplicative_noise_scale=multiplicative_noise)

                data_simu_chunk = csr_matrix(simulate_chunk.sample())
                chunks.append(data_simu_chunk)

                i = i + 1
        
        rna_sparse = vstack(chunks)
        self.rna_sparse = rna_sparse
        
        end_time = time.time()
        print(f"Simulating the expression count matrix of {ncells} cells over {ngenes} genes takes {end_time - start_time} secs.")



class Support_Functions():
    def __init__(
        self,
        mdata: Optional[MuData] = None,
        ncells_per_guide: Optional[int] = None,
        nguides_per_element: Optional[int] = None,
        read_depth: Optional[float] = None,
        lfc: Optional[float] = None,
        ngenes: Optional[int] = None
    ):
        """
        Parameters
        -----------
        mdata
            A MuData object with 2 modalities, "grna" and "rna", saving perturbation data and gene expr data, of the same shape.
            The "rna" modality has a varm called 'element_tested', whose # rows = rna.shape[1], # cols = grna.shape[1]
        ncells_per_guide, nguides_per_element, read_depth, lfc, ngenes
            same as when we import the mdata, just for outprinting
        """
        self.mdata = mdata
        self.ncells_per_guide = ncells_per_guide
        self.nguides_per_element = nguides_per_element
        self.read_depth = read_depth
        self.lfc = lfc
        self.ngenes = ngenes
    
    @staticmethod
    def mudata_filtering(
        mdata: Optional[MuData] = None,
        nguides_per_element: Optional[int] = 2,
        n_nonzero_trt_thresh: Optional[int] = 7,
        n_nonzero_cntrl_thresh: Optional[int] = 7
    ):
        """ 
        Filter element-gene pairs based on the number of non-zero expressions of perturbed and unperturbed cells. simply change the value to 0 in element_tested if the pair does not pass the filtering.

        Parameters
        ----------
        mdata
            A MuData object with 2 modalities, "grna" and "rna", saving perturbation data and gene expr data, of the same shape.
            The "rna" modality has a varm called 'element_tested', whose # rows = rna.shape[1], # cols = grna.shape[1]
        n_nonzero_trt_thresh
            least number of cells with non-zero expression, receiving a gRNA (same as SCEPTRE)
        n_nonzero_cntrl_thresh
            least number of cells with non-zero expression, not receiving a gRNA (same as SCEPTRE)

        Output
        ----------
        mdata_filtered
            a MuData object, whose main matrix of the "grna" have filtered columns, and 'element_test' in "rna" have filtered columns
        """
        rna = mdata['rna'].X.toarray()
        grna = mdata['grna'].X.toarray()
        element = mdata['grna'].X @ mdata['grna'].varm["element_targeted"].toarray()
        
        element_tested_array = mdata['rna'].varm["element_tested"].toarray()
        element_tested_filtered = mdata["rna"].varm["element_tested"].copy()
        
        for col_element in range(element.shape[1]):
            
            # change the idx of element to gene            
            cols = np.nonzero(element_tested_array[:, col_element] > 0)[0]
            
            element_mask_1 = (element[:, col_element] >= 1).flatten()
            element_mask_0 = (element[:, col_element] == 0).flatten()
            
            for col in cols:
                # Extract the column from rna and convert to dense format
                rna_col = rna[:, col].flatten()
                #print(rna_col.shape)

                # Condition 1: Number of non-zero values in 'rna' where 'element' has entry 1 should be >= n_nonzero_trt_thresh
                condition_trt = np.sum(rna_col[element_mask_1] > 0) >= n_nonzero_trt_thresh

                # Condition 2: Number of non-zero values in 'rna' where 'element' has entry 0 should be >= n_nonzero_cntrl_thresh
                condition_cntrl = np.sum(rna_col[element_mask_0] > 0) >= n_nonzero_cntrl_thresh
                
                # If one of the conditions do not hold, then we remove this pair from element_tested
                if not condition_trt or not condition_cntrl:
                    element_tested_filtered[col, col_element] = 0

        # Create filtered rna & grna modality
        mdata_filtered = mdata.copy()
        element_tested_filtered.eliminate_zeros()
        mdata_filtered.mod["rna"].varm["element_tested"] = element_tested_filtered
        
        # compare number of pairs before and after sampling
        npairs_before = mdata["rna"].varm["element_tested"].nnz
        npairs_after = mdata_filtered["rna"].varm["element_tested"].nnz
        print(f"{npairs_after} element-gene pairs pass the filtering among all {npairs_before} pairs.")
        
        return(mdata_filtered)
    
    @staticmethod
    def split_element_effects(
        element_effects
    ):
        '''
        Split the element_effect_res dataframe into two positive control group and negative control group according to gene name.
        '''
        element_effects_negative_ctrl = element_effects[element_effects['element'].str.contains('ntc')]
        element_effects_positive_ctrl = element_effects[~element_effects['element'].str.contains('ntc')]
        element_effects_dict = {"negative_control": element_effects_negative_ctrl,
                                "positive_control": element_effects_positive_ctrl}
        return(element_effects_dict)

    @staticmethod
    def get_alpha_empirical(
        alpha: Optional[float] = 0.05,
        test_side: Optional[str] = "both",
        p_val_list = None
    ):
        """
        Get the empirical alpha from the empirical p-values of the control pairs.
        
        Parameters
        -----------
        alpha
            A value between 0-1 (usually 0.1 or 0.05). (1 - alpha) is the significance level we want to achieve.
        test_side
            "both" or "single"
        p_val_list
            A list of p-values of the control pairs. (Here a list is not necessarily a list but can also be an array, a tensor, a column of a dataframe, etc., as long as .quantile works for it.)
        
        Output
        -----------
        alpha_empirical
            A value between 0-1
        """
        if test_side == "both":
            alpha_empirical = p_val_list.quantile(alpha/2)
        elif test_side == "single":
            alpha_empirical = p_val_list.quantile(alpha)
            
        return(alpha_empirical)

    @staticmethod
    def get_alpha_empirical_for_each_pair(
        alpha: Optional[float] = 0.05,
        test_side: Optional[str] = "both",
        element_effects_split = None
    ):
        """
        Get the empirical alpha from the empirical p-values of the control pairs.
        
        Parameters
        -----------
        alpha
            A value between 0-1 (usually 0.1 or 0.05). (1 - alpha) is the significance level we want to achieve.
        test_side
            "both" or "single"
        element_effects_split
            A dictionary, contain two keys ["positive_control", "negative_control"]
        
        Output
        -----------
        element_effects_split_modified
            Add an extra column to the positive_control group
        """
        df_neg = element_effects_split["negative_control"].copy()
        df_pos = element_effects_split["positive_control"].copy()
        unique_genes = df_pos["gene"].unique()
        
        alpha_empirical_list = [0] * len(unique_genes)
        for gene in unique_genes:
            
            sub_df = df_neg[df_neg["gene"] == gene]
            p_val_list = sub_df["q_value"]       
            
            if test_side == "both":
                alpha_empirical = p_val_list.quantile(alpha/2)
            elif test_side == "single":
                alpha_empirical = p_val_list.quantile(alpha)
            
            gene_id = np.where(unique_genes == gene)[0][0]
            alpha_empirical_list[gene_id] = alpha_empirical

        df_pos["alpha_empirical"] = alpha_empirical_list
        
        element_effects_split_modified = element_effects_split.copy()
        element_effects_split_modified["positive_control"] = df_pos
        
        return(element_effects_split_modified)
