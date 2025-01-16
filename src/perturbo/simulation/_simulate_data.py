import math
import os
import time

import mudata as md
import numpy as np
import pandas as pd
import pyro.distributions as dist
import torch
from scipy.sparse import csr_matrix, eye, vstack
from scipy.sparse import random as sparse_random
from scipy.stats import lognorm

from perturbo.models._module import LogNormalNegativeBinomial


class Simulate_Data:
    def __init__(
        self,
        df_dir_base: str | None = "from_real_data",
        mdata_name: str | None = None,
        estimator_type: str | None = None,
        batch_key: str | None = "prep_batch",
        library_size_key: str | None = "library_size",
        size_factor_key: str | None = "size_factor",
        # read_depth_key: Optional[str] = "read_depth",
        # Size_Factor_key: Optional[str] = "Size_Factor",
        continuous_covariates_keys: list[str] | None = None,
        # obs_continuous_covariates_keys: Optional[List[str]] = None,
        guide_by_element_key: str | None = None,
        gene_by_element_key: str | None = None,
    ):
        """
        Extract the desired estimation output.

        Parameters
        ----------
        df_dir_base
            Path, to which we extract the estimated parameters, by requirement. Will add the dataset and estimation method to it.
        mdata_name
            A given name of the dataset we are simulating from. Point to the directory where we save the estimated parameters
        estimator_type
            A given name of the type of estimator we are using (map/posterior). Point to the directory where we save the estimated parameters
        batch_key
            .obs key within the RNA AnnData that indicate where batch information is saved
        library_size_key
            .obs key within the RNA AnnData that can be directly observed, and can be used for generatig lisbrary size
        size_factor_key
            .obs key within the RNA AnnData object containing library size factors (log) for each sample
        continuous_covariates_keys
            list of .obs keys within the RNA AnnData object containing other continuous covariates to be "regressed out"
        gene_by_element_key
            .varm key within the RNA AnnData object containing a mask of which genes can be affected by which genetic elements

        """
        self.df_dir_base = df_dir_base
        self.mdata_name = mdata_name
        self.estimator_type = estimator_type
        self.batch_key = batch_key
        self.library_size_key = library_size_key
        self.size_factor_key = size_factor_key
        # if read_depth_key is None:
        #     self.read_depth_key = library_size_key
        # else:
        #     self.read_depth_key = read_depth_key  # in case it records total_umis, different from library_size (e.g. Gasperini_atscale)

        # if Size_Factor_key == None:
        #     self.Size_Factor_key = size_factor_key
        # else:
        #     self.Size_Factor_key = Size_Factor_key
        # self.Size_Factor_key = Size_Factor_key  # Size_Factor computed from read_depth that is considered in cont_cov but should not be directly sampled (e.g. Gasperini_atscale)
        self.continuous_covariates_keys = continuous_covariates_keys
        #self.obs_continuous_covariates_keys = obs_continuous_covariates_keys
        self.guide_by_element_key = guide_by_element_key
        self.gene_by_element_key = gene_by_element_key

        # run initial functions
        self._read_params()
        self._assign_values()

    def _read_params(self):
        """Read in the estimated parameters for later simulation."""
        df_dir = self.df_dir_base + "_" + f"{self.mdata_name}" + "_" + f"{self.estimator_type}"
        all_files = [f for f in os.listdir(df_dir) if f.endswith(".csv")]

        dfs = {}
        for csv_file in all_files:
            df_name = os.path.splitext(csv_file)[0]  # Get filename without extension
            dfs[df_name] = pd.read_csv(os.path.join(df_dir, csv_file))
        df = dfs["params_for_simulation"]

        self.dfs = dfs  # list of dataframes of all parameters
        self.df = df  # parameters for obs

    def _assign_values(self):
        """Assign values like total number genes from the estimated parameters, which have just been read in."""
        dfs = self.dfs

        total_genes = dfs["log_gene_mean"].shape[0]
        if self.batch_key is not None:
            total_batches = dfs["batch_effect"].shape[0]

        self.total_genes = total_genes
        self.total_batches = total_batches

    def sample_mudata(
        self,
        simulate_distribution: str | None = "lnnb",
        ncells: int | None = 200000,
        ngenes: int | None = 100,
        nbatches: int | None = 1,  # number of experimental batches
        nguides_pos: int | None = None,  # number of guides targeting elements that affect genes
        nguides_per_element: int | None = 4,  # number of guides per element (that affects a gene)
        nguides_ntc: int | None = 50,  # number of control guides
        ntc_element_target_method: str
        | None = "random",  # "random" / "fixed". Shows if ntc guide are randomly assigned to ntc elements or each ntc guide is assigned to only one ntc element.
        nelements_ntc: int | None = 100,  # number of control elements, i.e. control pairs
        ncells_per_guide: int | None = 20,  # number of cells each positive guide is detected in
        guide_efficacy_values: list[float] | None = None,
        guide_efficacy_type: str | None = "mixed",  # "mixed" / "scaled"
        guide_category: list[str] | None = None,
        log2_fold_change: float | None = 1.0,
        mean_reads_per_gene: float | None = 1.0,
        library_size_mean: float | None = None,
        library_size: int | None = None,
        chunk_size: int | None = 10000,
    ):
        """Combine everthing we have together, generate synthetic MuData that contain enough information for training"""
        nelements_pos = ngenes
        
        if guide_category is None:
            guide_category = ["positive_control", "negative_control"]
        if guide_efficacy_values is None:
            guide_efficacy_values = [1.0, 1.0, 1.0, 1.0]
        if nguides_pos is None:
            nguides_pos = nguides_per_element * nelements_pos
        if library_size_mean is None and mean_reads_per_gene is not None:
            library_size_mean = mean_reads_per_gene * ngenes
        elif mean_reads_per_gene is None and library_size_mean is not None:
            mean_reads_per_gene = library_size_mean / ngenes

        self.simulate_distribution = simulate_distribution
        self.ncells = ncells
        self.ngenes = ngenes
        self.nbatches = nbatches
        self.nguides_pos = nguides_pos
        self.nguides_per_element = nguides_per_element
        self.nelements_ntc = nelements_ntc
        self.ntc_element_target_method = ntc_element_target_method
        self.ncells_per_guide = ncells_per_guide
        self.guide_efficacy_values = guide_efficacy_values
        self.guide_efficacy_type = guide_efficacy_type
        self.guide_category = guide_category
        self.log2_fold_change = log2_fold_change
        self.library_size = library_size
        self.mean_reads_per_gene = mean_reads_per_gene
        self.library_size_mean = library_size_mean
        self.chunk_size = chunk_size

        pert_rate = ncells_per_guide / ncells
        lfc = np.log(2 ** (log2_fold_change))

        self.nelements_pos = nelements_pos
        self.nelements = self.nelements_pos
        self.nguides = self.nguides_pos
        if "negative_control" in guide_category:
            self.nguides_ntc = nguides_ntc
            self.nelements_ntc = nelements_ntc
            self.nguides = self.nguides_pos + self.nguides_ntc
            self.nelements = self.nelements_pos + self.nelements_ntc
        self.pert_rate = pert_rate
        self.lfc = lfc

        # set names for guides, genes, elements
        self._set_names()

        # grna modality
        self._sample_grna()
        self._get_element_targeted()
        self._get_element_targeted_uns()

        grna_modality = md.AnnData(self.grna_data)
        grna_modality.var_names = self.guide_names
        if self.guide_by_element_key is not None:
            grna_modality.varm[self.guide_by_element_key] = self.element_targeted

            grna_modality.uns[self.guide_by_element_key] = self.element_targeted_df
        grna_modality.uns["guide_efficacy"] = self.guide_efficacy_values
        grna_modality.uns["elements"] = np.array(self.element_names)

        # rna modality
        self._sample_obs()
        self._get_element_tested()
        self._get_element_tested_uns()
        self._get_logits()
        self._get_total_count()
        self._get_multiplicative_noise()
        self._get_log_mean_disp_slope()
        self._get_logits_perturb()
        # self._get_dispersion_from_curve()
        self._sample_rna()

        rna_modality = md.AnnData(self.rna_sparse, obs=self.obs, uns={"fold_change": round(np.exp(lfc), 2)})

        # if self.read_depth_key is not None:
        #     rna_modality.obs[self.read_depth_key] = rna_modality.X.sum(axis=1)  # add the library size obs
        #     log_cpm = np.log(rna_modality.obs[self.read_depth_key] / 1e6)
        # else:
        #     self.read_depth_key = self.library_size_key

        # if self.Size_Factor_key is not None:
        #     rna_modality.obs[self.Size_Factor_key] = log_cpm - np.mean(log_cpm)

        rna_modality.var["gene_mean"] = (np.exp(self.logits_corrected) * self.total_count).mean(axis=0)
        rna_modality.var["gene_total_count"] = self.total_count.squeeze()
        rna_modality.var["gene_mean_perturbed"] = (np.exp(self.logits_perturb) * self.total_count).mean(axis=0)

        rna_modality.var_names = self.gene_names
        if self.gene_by_element_key is not None:
            rna_modality.varm[self.gene_by_element_key] = self.element_tested.transpose()

            rna_modality.uns[self.gene_by_element_key] = self.element_tested_df
        rna_modality.uns["elements"] = np.array(self.element_names)

        # Construct mudata
        mdata = md.MuData({"rna": rna_modality, "grna": grna_modality})
        mdata["rna"].obs[self.library_size_key] = mdata["rna"].X.sum(axis=1)
        print(mdata)

        simulated_library_size = mdata["rna"].obs[self.library_size_key]
        print(
            f"when setting the read depth per cell as {library_size_mean}, simulated data has an average of {simulated_library_size.mean()} reads per cell."
        )

        self.mdata = mdata
        return mdata

    @staticmethod
    def extract_to_h5mu(mdata, mudata_path="simulated_data/", mudata_name="simulated_data.h5mu"):
        if not os.path.exists(mudata_path):
            os.makedirs(mudata_path)

        mdata.write(mudata_path + mudata_name)
        print("Done")

    def _set_names(self):
        """Set gene names, guide names, and element names. Distinguish between positive guides and non-targeting guides"""
        # gene names
        self.gene_names = ["gene" + str(i) for i in range(0, self.ngenes)]

        # element names
        self.element_names_pos = []
        self.element_names_pos = self.gene_names
        self.element_names = self.element_names_pos
        if "negative_control" in self.guide_category:
            self.element_names_ntc = []
            self.element_names_ntc = ["ntc" + str(i) for i in range(0, self.nelements_ntc)]
            self.element_names = self.element_names_pos + self.element_names_ntc

        # guide names
        self.guide_names_pos = [
            name + "_g" + str(k) for name in self.element_names_pos for k in range(0, self.nguides_per_element)
        ]
        self.guide_names = self.guide_names_pos
        if "negative_control" in self.guide_category:
            self.guide_names_ntc = ["ntc_g" + str(k) for k in range(self.nguides_ntc)]
            self.guide_names = self.guide_names_pos + self.guide_names_ntc

    def _create_batch_list(self):

        ncells = self.ncells
        nbatches = self.nbatches
        total_batches = self.total_batches

        # Step 1: Create a list with ncells/n occurrences of each "batch_i"
        batch_ids = np.random.choice(range(total_batches), size=nbatches, replace=True)  # randomly selected index of batches from existing batch numbers
        batch_list = []
        for i in batch_ids:
            batch_list.extend([f"batch_{i}"] * (ncells // nbatches))

        # Step 2: Handle any remaining cells (in case ncells is not perfectly divisible by n)
        remaining = ncells % nbatches
        for i in range(remaining):
            batch_list.append(f"batch_{i}")

        # Step 3: Shuffle the list to randomize the order
        np.random.shuffle(batch_list)

        return batch_list

    def _sample_obs(self):
        """Generate for .obs for RNA modality. A dataframe, each column is a obs, with ncells rows."""
        df = self.df
        ncells = self.ncells
        batch_key = self.batch_key
        continuous_covariates_keys = self.continuous_covariates_keys
        # library_size_key = self.library_size_key
        size_factor_key = self.size_factor_key

        obs = pd.DataFrame()

        # batch number
        prep_batch = self._create_batch_list()
        obs[batch_key] = prep_batch

        # other obs
        for obs_key in (continuous_covariates_keys + [size_factor_key]):

            obs_param_key = "params_" + obs_key.replace(".", "_")  # get the key for params_for_simulation
            obs_values = lognorm.rvs(
                df[obs_param_key][0],
                df[obs_param_key][1],
                df[obs_param_key][2],
                size=ncells,
            )
            # if obs_key == library_size_key:
            #     obs_values = obs_values * 1e6
            obs[obs_key] = obs_values

        # # compute "size_factor" from library_size (e.g. "umi_count")
        # if library_size_key in continuous_covariates_keys:
        #     log_cpm = np.log(obs[library_size_key].values / 1e6)
        #     obs[size_factor_key] = (log_cpm - np.mean(log_cpm)).reshape(-1, 1)

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

        grna_data = sparse_random(ncells, nguides, density=pert_rate, data_rvs=lambda n: np.ones(n), format="csr")
        self.grna_data = grna_data

    def _get_element_targeted(self):
        """Generate grna.varm["element_targeted"]. self.element_targeted, a sparse matrix, with nguides rows, nelement cols, indicate which element is targeted by which guide."""
        nguides_pos = self.nguides_pos
        guide_category = self.guide_category
        nelements_pos = self.nelements_pos
        nguides_per_element = self.nguides_per_element
        ntc_element_target_method = self.ntc_element_target_method

        # Fill the specific entries with 1s (for positive elements)
        element_targeted_pos_dense = np.zeros((nguides_pos, nelements_pos))

        for j in range(nelements_pos):
            start_row = nguides_per_element * j
            end_row = start_row + nguides_per_element
            if end_row > nguides_pos:
                break  # break the loop if the end_row exceeds the matrix size
            element_targeted_pos_dense[start_row:end_row, j] = 1

        element_targeted_dense = element_targeted_pos_dense

        # Fill the specific entries with 1s (for negative elements)
        if "negative_control" in guide_category:
            nelements_ntc = self.nelements_ntc
            nguides_ntc = self.nguides_ntc
            element_targeted_ntc_dense = np.zeros((nguides_ntc, nelements_ntc))

            if ntc_element_target_method == "random":
                # randomly assign ntc guides to ntc elements
                for j in range(nelements_ntc):
                    affecting_rows = np.random.choice(
                        nguides_ntc, nguides_per_element, replace=False
                    )  # in each column assign 4 values to 1
                    element_targeted_ntc_dense[affecting_rows, j] = 1

            elif ntc_element_target_method == "fixed":
                if nguides_ntc != nelements_ntc * nguides_per_element:
                    raise ValueError("The number of NTC guides and elements do not match.")
                else:
                    for i in range(nelements_ntc):
                        start_row = i * nguides_per_element
                        end_row = start_row + nguides_per_element
                        element_targeted_ntc_dense[start_row:end_row, i] = 1

            # Combine pos+neg.
            element_targeted_dense = np.zeros((nguides_pos + nguides_ntc, nelements_pos + nelements_ntc))
            element_targeted_dense[:nguides_pos, :nelements_pos] = element_targeted_pos_dense
            element_targeted_dense[nguides_pos:, nelements_pos:] = element_targeted_ntc_dense

        # Convert the dense matrix to a sparse matrix (CSR format)
        element_targeted = csr_matrix(element_targeted_dense)

        self.element_targeted = element_targeted

    def _get_element_targeted_uns(self):
        """Generate grna.uns["element_targeted"]. self.element_targeted_df, a dataframe, with 2 columns | grna | element |. save guide name and its targeting element."""
        element_names_pos = self.element_names_pos
        guide_names_pos = self.guide_names_pos
        nguides_per_element = self.nguides_per_element

        element_names_pos_extend = [element for element in element_names_pos for _ in range(nguides_per_element)]
        element_targeted_df = pd.DataFrame({"grna": guide_names_pos, "element": element_names_pos_extend})

        self.element_targeted_df = element_targeted_df

    def _get_element_tested(self):
        """
        Generate rna.varm["element_tested"]. self.element_tested, a sparse matrix indicating potentially related elements and genes, with nelement rows, ngenes cols.

        In "positive_control" case, it is an identity matrix, one element is only targeting one gene.
        In "negative_control" case, it is a matrix full of 1. Studying as much negative pairs as possible creates a more realistic negative control distribution.
        """
        guide_category = self.guide_category
        ngenes = self.ngenes
        nelements_ntc = self.nelements_ntc

        # for positive control elements
        element_tested_pos = eye(ngenes, dtype="float32").tocsr()

        element_tested = element_tested_pos

        # add ntc elements (if there are any)
        if "negative_control" in guide_category:
            element_tested_ntc_dense = np.zeros((nelements_ntc, ngenes))
            target_columns = np.random.randint(ngenes, size=nelements_ntc)
            element_tested_ntc_dense[np.arange(nelements_ntc), target_columns] = 1

            # turn to sparse
            element_tested_ntc = csr_matrix(element_tested_ntc_dense)
            self.element_tested_ntc = element_tested_ntc
            element_tested = vstack([element_tested_pos, element_tested_ntc])

        self.element_tested = element_tested
        self.element_tested_pos = element_tested_pos

    def _get_element_tested_uns(self):
        """Generate rna.uns["element_tested"]. self.element_tested_df, a dataframe, with 2 columns | element | gene |. save element name and its targeting gene."""
        element_names_pos = self.element_names_pos
        gene_names = self.gene_names

        element_tested_df = pd.DataFrame({"element": element_names_pos, "gene": gene_names})
        self.element_tested_df = element_tested_df

    def _get_total_perturbation_effect(self):
        """
        Get matrix of the effect of each guide on each gene according to guide_efficacy and element_effects.

        Output: self.total_perturbation_effect. An np.array, with nguides rows and ngenes cols
        """
        lfc = self.lfc
        guide_category = self.guide_category
        guide_efficacy_values = self.guide_efficacy_values
        guide_efficacy_type = self.guide_efficacy_type
        nelements = self.nelements
        nelements_pos = self.nelements_pos
        if "negative_control" in guide_category:
            # nelements_ntc = self.nelements_ntc
            nguides_ntc = self.nguides_ntc
        element_targeted = self.element_targeted
        element_tested = self.element_tested

        # for positive guides
        guide_efficacy_long_pos_mean = (
            guide_efficacy_values * nelements_pos
        )  # list of len nguides_pos, saving mean efficacy of each positive guide
        if guide_efficacy_type == "scaled":
            guide_efficacy_long_pos = guide_efficacy_long_pos_mean
        elif guide_efficacy_type == "mixed":
            guide_efficacy_long_pos = np.random.binomial(
                n=1, p=guide_efficacy_long_pos_mean
            ).tolist()  # list of len nguides_pos, saving efficacy generated from the means.

        guide_efficacy_long = guide_efficacy_long_pos

        # if there are negative control guides
        if "negative_control" in guide_category:
            guide_efficacy_long_ntc = (
                [0] * nguides_ntc
            )  # list of nguides_ntc 0s. negative control guides having no efficacy, will not effect logit.

            guide_efficacy_long = guide_efficacy_long_pos + guide_efficacy_long_ntc

        # expand from (nguides, 1) list to (nguides, nelements) matrix
        guide_efficacy_values = torch.tensor(guide_efficacy_long).reshape(-1, 1).expand(-1, nelements)
        guide_efficacy = guide_efficacy_values * (element_targeted).toarray()  # shape (nguides, nelements)

        # element_effects
        element_effects = lfc * element_tested  # shape (nelements, ngenes)

        # total_perturbation_effect: apply guide efficacy to
        total_perturbation_effect = guide_efficacy @ element_effects.toarray()  # shape (nguides, ngenes)
        self.total_perturbation_effect = total_perturbation_effect

        return total_perturbation_effect

    def _get_random_indices(self):
        """Get random indices of genes to simulate from. An np.array, with length ngenes."""
        total_genes = self.total_genes
        ngenes = self.ngenes

        gene_ids = np.random.choice(range(total_genes), size=ngenes, replace=True)
        self.gene_ids = gene_ids

        return gene_ids

    def _get_batch_effect(self):
        """Out put should have shape (ncells, ngenes)"""
        obs = self.obs
        dfs = self.dfs
        gene_ids = self._get_random_indices()
        batch_key = self.batch_key
        total_batches = self.total_batches
        # nbatches = self.nbatches
        ncells = self.ncells

        # get random batch ids for selecting corresponding batch effects from real_world data
        # batch_ids = np.random.choice(range(total_batches), size=nbatches, replace=True)

        # One-hot encode batch number
        one_hot_matrix = np.zeros((ncells, total_batches), dtype=int)
        batch_mapping = {f"batch_{i}": i for i in range(total_batches)}
        # batch_list = obs[batch_key]

        for i, batch in enumerate(obs[batch_key]):
            col_idx = batch_mapping[batch]
            one_hot_matrix[i, col_idx] = 1

        batch_effect_sizes = one_hot_matrix.reshape(-1, total_batches) @ dfs["batch_effect"].iloc[
            :, gene_ids
        ].values.reshape(total_batches, -1)

        return batch_effect_sizes

    def _get_logits(self):
        """Get the raw logits matrix. A torch.tensor, with ncells rows, ngenes cols."""
        size_factor_key = self.size_factor_key
        batch_key = self.batch_key
        continuous_covariates_keys = self.continuous_covariates_keys
        dfs = self.dfs
        gene_ids = self._get_random_indices()
        obs = self.obs
        simulate_distribution = self.simulate_distribution

        # put size_factor in the last position of continuous covariates
        # if size_factor_key not in continuous_covariates_keys:
        #    continuous_covariates_keys = continuous_covariates_keys + [size_factor_key]

        # Get covariate effects
        cov_effect_sizes = {}
        for cov_key in continuous_covariates_keys:
            # if (
            #     cov_key == "Size_Factor"
            # ):  # skip this, since this is related to the total_umis, and will later modeled by correction_term
            #     continue
            idx = continuous_covariates_keys.index(cov_key)
            cov_effect = dfs["cont_covariate_effect"].iloc[idx, gene_ids].values
            cov_effect_size = obs[cov_key].values.reshape(-1, 1) @ cov_effect.reshape(1, -1)  # torch, ncells * ngenes

            cov_effect_sizes[cov_key] = cov_effect_size

        if batch_key is not None:
            batch_effect_sizes = self._get_batch_effect()
            cov_effect_sizes[batch_key] = batch_effect_sizes

        # samples_log_gene_mean (directly choose from the original mean with the given index)
        samples_log_gene_mean = dfs["log_gene_mean"].loc[gene_ids,].values.reshape(1, -1)

        # samples_log_gene_dispersion (directly choose from the original mean with the given index)
        samples_log_gene_dispersion = dfs["log_gene_dispersion"].loc[gene_ids,].values.reshape(1, -1)

        # mutiplicative noise
        samples_multiplicative_noise = dfs["multiplicative_noise"].loc[gene_ids,].values.reshape(1, -1)

        # logit
        if simulate_distribution == "nb":
            logits = torch.from_numpy(
                samples_log_gene_mean  # base mean
                +
                obs[size_factor_key].values.reshape(-1,1) +    # size factor
                sum(cov_effect_sizes.values())  # covariate effect sizes
                - samples_log_gene_dispersion
            )  # torch.tensor, length = ngenes  # torch.tensor, length = ngenes
        elif simulate_distribution == "lnnb":
            logits = torch.from_numpy(
                samples_log_gene_mean  # base mean
                +
                obs[size_factor_key].values.reshape(-1,1) +    # size factor
                sum(cov_effect_sizes.values())  # covariate effect sizes
                - samples_log_gene_dispersion
                - samples_multiplicative_noise**2 / 2
            )  # torch.tensor, length = ngenes

        self.samples_log_gene_mean = samples_log_gene_mean
        self.samples_log_gene_dispersion = samples_log_gene_dispersion
        self.samples_multiplicative_noise = samples_multiplicative_noise
        self.cov_effect_sizes = cov_effect_sizes

        self.logits = logits
        # print(f"logit before adding size_factor {self.logits}")

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
        library_size_mean = self.library_size_mean
        # ngenes = self.ngenes
        # ncells = self.ncells
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
            orig_means = (
                (np.exp(logits + multiplicative_noise**2 / 2) * total_count).detach().cpu().numpy()
            )  # array, shape = (ncells, ngenes)

        orig_means_pert = orig_means * np.exp(log_pert_effect)

        correction_term = (library_size_mean) / (orig_means_pert.sum(axis=1).mean())
        #correction_term = (library_size_mean) / (orig_means_pert.sum(axis=1))  # array, length = ncells
        #correction_term = correction_term.reshape(
        #    -1, 1
        #)  # this correction term already account for Size_Factor in simulation (derived from total UMIs)

        self.orig_means = orig_means
        self.correction_term = correction_term
        self.log_pert_effect = log_pert_effect

        return correction_term

    def _get_log_mean_disp_slope(self):
        samples_log_gene_mean = self.samples_log_gene_mean.ravel()  # reshape array to be 1-dim for fitting
        samples_log_gene_dispersion = self.samples_log_gene_dispersion.ravel()  # reshape array to be 1-dim for fitting

        # Fit a linear model: samples_log_gene_dispersion ~ samples_log_gene_mean
        slope, intercept = np.polyfit(samples_log_gene_mean, samples_log_gene_dispersion, 1)

        self.log_mean_disp_slope = slope

    def _get_logits_corrected(self):
        """Get the logits after correction. A torch.tensor, with ncell rows, ngene columns"""
        correction_term = self._get_correction_term()
        # ncells = self.ncells
        logits = self.logits
        slope = self.log_mean_disp_slope

        # expr_spread = np.random.normal(loc=0, scale=np.sqrt(0.2), size=ncells)  # add some noise to the expression mean
        # logits_corrected = logits + (1 / (slope + 1)) * np.log(correction_term) + expr_spread.reshape(-1,1)
        logits_corrected = logits + (1 / (slope + 1)) * np.log(correction_term)
        self.logits_corrected = logits_corrected

        return logits_corrected

    def _get_total_count_corrected(self):
        slope = self.log_mean_disp_slope
        correction_term = self.correction_term
        samples_log_gene_dispersion = self.samples_log_gene_dispersion + (slope / (slope + 1)) * np.log(correction_term)

        total_count_corrected = torch.from_numpy(np.exp(samples_log_gene_dispersion))  # torch.tensor, length = ngenes

        self.total_count_corrected = total_count_corrected
        return total_count_corrected

    # def _add_size_factor_in_logit(self, logits_corrected, total_count_corrected):
    #     dfs = self.dfs
    #     gene_ids = self.gene_ids
    #     # cov_effect_sizes = self.cov_effect_sizes

    #     simulate_distribution = self.simulate_distribution
    #     # logits_corrected = self.logits_corrected
    #     # total_count_corrected = self.total_count_corrected
    #     multiplicative_noise = self.multiplicative_noise
    #     # size_factor_key = self.size_factor_key

    #     if simulate_distribution == "nb":
    #         means_corrected = (np.exp(logits_corrected) * total_count_corrected).detach().cpu().numpy()  # array, shape = (ncells, ngenes)
    #     elif simulate_distribution == "lnnb":
    #         means_corrected = (
    #             (np.exp(logits_corrected + multiplicative_noise**2 / 2) * total_count_corrected).detach().cpu().numpy()
    #         )  # array, shape = (ncells, ngenes)

    #     # deal with library_size and size_factor
    #     library_size_cell = means_corrected.sum(axis=1)  # library size for each cell
    #     log_cpm = np.log(library_size_cell / 1e5)
    #     size_factor_cell = log_cpm - np.mean(log_cpm)

    #     size_factor_effect = dfs["cont_covariate_effect"].iloc[-1, gene_ids].values
    #     size_factor_effect_size = size_factor_cell.reshape(-1,1) @ size_factor_effect.reshape(1,-1)

    #     # cov_effect_sizes[size_factor_key] = size_factor_effect_size

    #     logits_spread = logits_corrected + size_factor_effect_size + size_factor_cell.reshape(-1, 1)
    #     print(f"logit after adding size_factor {logits_spread}")
    #     return logits_spread

    def _get_logits_perturb(self):
        """Get the logits after perturbation. A torch.tensor, with ncell rows, ngene columns"""
        logits_corrected = self._get_logits_corrected()
        total_count_corrected = self._get_total_count_corrected()
        # logits_spread = self._add_size_factor_in_logit(logits_corrected, total_count_corrected)

        log_pert_effect = self.log_pert_effect

        # logits_perturb = logits_spread + log_pert_effect
        # print(f"logits_corrected {logits_corrected}")
        # print(f"logits_spread {logits_spread}")
        # print(f"logits_perturb {logits_perturb}")
        logits_perturb = logits_corrected + log_pert_effect
        self.logits_perturb = logits_perturb

    def _sample_rna(self):
        """Generate rna.X. A sparse_matrix, with ncells rows, ngene columns"""
        simulate_distribution = self.simulate_distribution
        ncells = self.ncells
        ngenes = self.ngenes
        chunk_size = self.chunk_size
        total_count_corrected = self.total_count_corrected
        #print(f"total_count size {total_count_corrected.shape}")
        logits_perturb = self.logits_perturb
        #print(f"logit size {logits_perturb.shape}")
        multiplicative_noise = self.multiplicative_noise
        #print(f"noise size {multiplicative_noise.shape}")

        #print(f"logits size {logits_perturb.shape}")
        #print(f"total count size {total_count_corrected.shape}")

        start_time = time.time()
        chunks = []
        i = 1
        print(f"The data will be simulated with {math.ceil(ncells / chunk_size)} chunks.")

        if simulate_distribution == "nb":
            print("simulate from NB")
            for start_row in range(0, ncells, chunk_size):
                # print(f"Simulate chunk {i}.")
                end_row = min(start_row + chunk_size, ncells)
                logits_perturb_chunk = logits_perturb[start_row:end_row, :]
                #total_count_chunk = total_count_corrected[start_row:end_row, :]
                simulate_chunk = dist.NegativeBinomial(
                    logits=logits_perturb_chunk, 
                    total_count=total_count_corrected)

                data_simu_chunk = csr_matrix(simulate_chunk.sample())
                chunks.append(data_simu_chunk)

                i = i + 1

        elif simulate_distribution == "lnnb":
            print("simulate from LNNB")
            for start_row in range(0, ncells, chunk_size):
                # print(f"Simulate chunk {i}.")
                end_row = min(start_row + chunk_size, ncells)
                logits_perturb_chunk = logits_perturb[start_row:end_row, :]
                #total_count_chunk = total_count_corrected[start_row:end_row, :]
                simulate_chunk = LogNormalNegativeBinomial(
                    logits=logits_perturb_chunk,
                    total_count=total_count_corrected,
                    multiplicative_noise_scale=multiplicative_noise,
                )

                data_simu_chunk = csr_matrix(simulate_chunk.sample())
                chunks.append(data_simu_chunk)

                i = i + 1

        rna_sparse = vstack(chunks)
        self.rna_sparse = rna_sparse

        end_time = time.time()
        print(
            f"Simulating the expression count matrix of {ncells} cells over {ngenes} genes takes {end_time - start_time} secs."
        )
