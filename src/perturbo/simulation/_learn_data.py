import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
from mudata import MuData
from scipy.stats import lognorm

import perturbo


class Learn_Data:  # keep consistent with perturbo / pyro
    def __init__(
        self,
        mdata: MuData | None = None,
        batch_key: str | None = None,
        library_size_key: str | None = None,
        size_factor_key: str | None = None,
        # read_depth_key: Optional[str] = None,
        # Size_Factor_key: Optional[str] = None,
        continuous_covariates_keys: list[str] | None = None,
        # obs_continuous_covariates_keys: Optional[List[str]] = None,
        gene_by_element_key: str | None = None,
        guide_by_element_key: str | None = None,
        rna_element_uns_key: str | None = None,
        guide_element_uns_key: str | None = None,
        modalities: dict[str, str] | None = None,
    ):
        """
        Fit an example MuData with PerTurbo

        Parameters
        ----------
        mdata
            The example MuData.
        batch_key
            Key within the RNA AnnData .obs corresponding to the experimental batch
        library_size_key
            .obs key within the RNA AnnData that can be directly observed, and can be used for generatig lisbrary size
        size_factor_key
            .obs key within the RNA AnnData object containing library size factors (log) for each sample
        continuous_covariates_keys
            list of .obs keys within the RNA AnnData object containing other continuous covariates to be "regressed out"
        gene_by_element_key
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

        if size_factor_key is None:
            size_factor_key = "size_factor"
        if library_size_key is None:
            library_size_key = "library_size"

        self.mdata_train = mdata
        self.batch_key = batch_key
        self.library_size_key = library_size_key
        self.size_factor_key = size_factor_key
        # self.read_depth_key = read_depth_key
        # self.Size_Factor_key = Size_Factor_key

        self.continuous_covariates_keys = continuous_covariates_keys
        # self.obs_continuous_covariates_keys = obs_continuous_covariates_keys
        self.gene_by_element_key = gene_by_element_key
        self.guide_by_element_key = guide_by_element_key
        self.rna_element_uns_key = rna_element_uns_key
        self.guide_element_uns_key = guide_element_uns_key
        self.modalities = modalities
        self.rna_layer = modalities["rna_layer"]
        self.grna_layer = modalities["perturbation_layer"]

    def fit_obs(
        self,
    ):
        """Output: params_for_simulation, a dataframe, saving lognormal parameters for each obs."""
        params_for_simulation = pd.DataFrame()
        obs_param_keys = []

        obs_keys_array = self.mdata_train.mod[self.rna_layer].obs.columns.to_numpy()
        if self.library_size_key not in obs_keys_array:
            self.mdata_train.mod[self.rna_layer].obs[self.library_size_key] = self.mdata_train.mod[self.rna_layer].X.sum(axis=1)
        if self.size_factor_key not in obs_keys_array:
            log_cpm = np.log(self.mdata_train.mod[self.rna_layer].obs[self.library_size_key] / 1e6)
            self.mdata_train.mod[self.rna_layer].obs[self.size_factor_key] = log_cpm - np.mean(log_cpm)

        for obs_key in (self.continuous_covariates_keys + [self.size_factor_key]):
            obs_param_key = "params_" + obs_key.replace(".", "_")
            obs_values = self.mdata_train.mod[self.rna_layer].obs[obs_key]
            # if obs_key == self.library_size_key:
            #     obs_values = obs_values  # / 1e6
            #     obs_param = lognorm.fit(obs_values, floc=0)
            # else:
            obs_param = lognorm.fit(obs_values)

            params_for_simulation[obs_param_key] = obs_param
            obs_param_keys = obs_param_keys + [obs_param_key]

        self.obs_param_keys = obs_param_keys
        self.params_for_simulation = params_for_simulation
        return params_for_simulation

    def get_n_steps(
        self,
        max_steps: int | None = 400,
    ):
        """Get number of training steps according to sample size. training steps decrease with increasing sample size."""
        n_steps = min(
            max_steps, round(max_steps * (20000 / self.mdata_train[self.rna_layer].X.shape[0]))
        )  # if ncells > 20000 then n_steps decay
        n_steps = max(n_steps, 1)

        self.n_steps = n_steps
        return n_steps

    def get_model(
        self,
        likelihood: str | None = None,
        effect_prior_dist="normal",  # "cauchy" | "normal_mixture" | "normal"
        efficiency_mode="scaled",  # "scaled" | "mixture"
    ):
        """Get a model object (as in pyro) that could be trained."""
        # register data with perturbo
        perturbo.models.PERTURBO.setup_mudata(
            self.mdata_train,
            batch_key=self.batch_key,
            library_size_key=self.library_size_key,
            continuous_covariates_keys=self.continuous_covariates_keys,
            gene_by_element_key=self.gene_by_element_key,  # <------------ gene * element pairs
            guide_by_element_key=self.guide_by_element_key,
            guide_element_uns_key=self.guide_element_uns_key,  # <------------ list of elements tested here
            rna_element_uns_key=self.rna_element_uns_key,  # <------------ what is this?
            modalities=self.modalities,
        )
        model = perturbo.models.PERTURBO(self.mdata_train, likelihood=likelihood, effect_prior_dist=effect_prior_dist, efficiency_mode=efficiency_mode, n_factors=None)

        self.model = model
        return model

    def train_perturbo(  # mimic the train in pyro
        self,
        n_steps: int | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
        accelerator: str | None = None,
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
        if len(pyro.get_param_store().keys()) == 0:
            print("param_store is clean. prepared for training.")

        print(
            f"Training data with {n_steps} steps, using {lr} learning rate, {batch_size} batch size, on {accelerator}."
        )

        self.model.train(max_epochs=n_steps, lr=lr, batch_size=batch_size, accelerator=accelerator)

    def extract_obs_params(
        self, estimator_type: str | None = None, df_dir_base: str | None = None, mdata_name: str | None = None
    ):
        df = self.params_for_simulation

        # create folder for saving the estimations
        df_dir = df_dir_base + "_" + f"{mdata_name}" + "_" + f"{estimator_type}"
        if not os.path.exists(df_dir):
            os.makedirs(df_dir)

        # extract the estimation
        filename = "params_for_simulation.csv"  # Create a valid filename from the parameter name
        filepath = os.path.join(df_dir, filename)  # Join the directory with the filename

        df.to_csv(filepath, index=False)  # Save DataFrame to csv

    def extract_estimation(
        self,
        model=None,
        param_keys: list[str] | None = None,
        estimator_type: str | None = None,
        df_dir_base: str | None = None,
        mdata_name: str | None = None,
    ):
        """
        Extract the desired estimation output.

        Parameters
        ----------
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
        if model is None:
            model = self.model
        # Create a dictionary to store DataFrames for each parameter
        dfs = {}

        if estimator_type == "map":
            for k, v in model.module.guide.median().items():
                if k in param_keys:  # only extract the parameters we need
                    arr = v.cpu().detach().numpy()  # Convert tensor to numpy array
                    dfs[k] = pd.DataFrame(arr)  # Convert numpy array to DataFrame

        elif estimator_type == "posterior":
            param_store = pyro.get_param_store()
            for k, v in param_store.items():
                if k in param_keys:  # only extract the parameters we need
                    arr = v.cpu().detach().numpy()  # Convert tensor to numpy array
                    dfs[k] = pd.DataFrame(arr)  # Convert numpy array to DataFrame
        else:
            print("Please choose a correct estimator type from (map/posterior).")

        # add two additional estimation dataframes
        dfs["element_effects_res"] = model.get_element_effects()
        dfs["params_for_simulation"] = self.params_for_simulation

        self.dfs = dfs

        # create folder for saving the estimations
        df_dir = df_dir_base + "_" + f"{mdata_name}" + "_" + f"{estimator_type}"

        # extract the estimation
        if not os.path.exists(df_dir):
            os.makedirs(df_dir)

        for name, df in dfs.items():
            filename = name.replace(".", "_") + ".csv"  # Create a valid filename from the parameter name
            filepath = os.path.join(df_dir, filename)  # Join the directory with the filename

            df.to_csv(filepath, index=False)  # Save DataFrame to csv

        return dfs

    def plot_obs(
        self,
    ):
        """plot histograms of observable var from .obs"""
        n_plots = len(self.continuous_covariates_keys) + 1

        # Create n_plots number of subplots
        fig, axes = plt.subplots(1, n_plots, figsize=(12, 5))

        if n_plots == 1:
            axes = [axes]  # Convert single Axes object to a list for consistent indexing

        i = 0
        for obs_key in (self.continuous_covariates_keys + [self.size_factor_key]):
            data = self.mdata_train[self.rna_layer].obs[obs_key]  # Your data for the histogram
            if obs_key == self.library_size_key:
                data = data  # / 1e6
            axes[i].hist(
                data, bins=50, color="grey", edgecolor="black", density=True
            )  # Notice density=True for normalization

            # Retrieve fitted parameters
            obs_param_key = "params_" + obs_key.replace(".", "_")
            shape, loc, scale = self.params_for_simulation[obs_param_key]

            # Define a range for x values
            xmin, xmax = axes[i].get_xlim()  # Get the limits from the histogram
            x = np.linspace(xmin, xmax, 100)

            # Calculate the PDF using the fitted parameters
            pdf = lognorm.pdf(x, shape, loc, scale)

            # Plot the density curve on the same axes
            axes[i].plot(x, pdf, "r-", label="Fit", color="royalblue")
            axes[i].set_xlabel(f"{obs_key}")
            axes[i].set_ylabel("Density")

            # Optional: Add a legend
            axes[i].legend()

            i += 1

        # Display the plot
        plt.tight_layout()
        plt.show()
