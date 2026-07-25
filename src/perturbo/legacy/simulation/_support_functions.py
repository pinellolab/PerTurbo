import numpy as np
import pandas as pd
from mudata import MuData
from scipy.sparse import csr_matrix, issparse
from statsmodels.stats.multitest import multipletests  # for FDR correction


def mudata_filtering(
    mdata: MuData,
    gene_by_element_key: str | None = "element_tested",
    guide_by_element_key: str | None = "element_targeted",
    rna_modality="rna",
    grna_modality="grna",
    nguides_per_element: int | None = 2,
    n_nonzero_trt_thresh: int | None = 7,
    n_nonzero_cntrl_thresh: int | None = 7,
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
    if guide_by_element_key is None:
        print("Please indicate the correct guide_by_element_key.")
        return

    rna = mdata[rna_modality].X
    # grna = mdata[grna_modality].X.toarray()
    guide_by_element = mdata[grna_modality].varm[guide_by_element_key]
    if isinstance(guide_by_element, pd.DataFrame):
        guide_by_element = guide_by_element.values
    if issparse(guide_by_element):
        guide_by_element = guide_by_element.toarray()

    element = mdata[grna_modality].X @ guide_by_element

    if gene_by_element_key is not None:
        element_tested = mdata[rna_modality].varm[gene_by_element_key]
        if isinstance(element_tested, pd.DataFrame):
            element_tested = element_tested.values
        if issparse(element_tested):
            element_tested = element_tested.toarray()

    for col_element in range(element.shape[1]):
        # change the idx of element to gene
        cols = np.nonzero(element_tested[:, col_element] > 0)[0]

        element_mask_1 = (element[:, col_element] >= 1).flatten()
        element_mask_0 = (element[:, col_element] == 0).flatten()

        for col in cols:
            # Extract the column from rna and convert to dense format
            rna_col = rna[:, col]
            if issparse(rna_col):
                rna_col = rna_col.toarray()
            rna_col = rna_col.flatten()
            # print(rna_col.shape)

            # Condition 1: Number of non-zero values in 'rna' where 'element' has entry 1 should be >= n_nonzero_trt_thresh
            condition_trt = np.sum(rna_col[element_mask_1] > 0) >= n_nonzero_trt_thresh

            # Condition 2: Number of non-zero values in 'rna' where 'element' has entry 0 should be >= n_nonzero_cntrl_thresh
            condition_cntrl = np.sum(rna_col[element_mask_0] > 0) >= n_nonzero_cntrl_thresh

            # If one of the conditions do not hold, then we remove this pair from element_tested
            if not condition_trt or not condition_cntrl:
                element_tested[col, col_element] = 0

    # Create filtered rna & grna modality
    mdata_filtered = mdata.copy()

    if gene_by_element_key is not None:
        mdata_filtered.mod[rna_modality].varm[gene_by_element_key] = csr_matrix(element_tested)

        # compare number of pairs before and after sampling
        npairs_before = (mdata[rna_modality].varm[gene_by_element_key] != 0).sum()
        npairs_after = mdata_filtered[rna_modality].varm[gene_by_element_key].nnz
        print(f"{npairs_after} element-gene pairs pass the filtering among all {npairs_before} pairs.")

    return mdata_filtered


def split_element_effects(element_effects):
    """Split the element_effect_res dataframe into two positive control group and negative control group according to gene name."""
    element_effects_negative_ctrl = element_effects[element_effects["element"].str.contains("ntc")]
    element_effects_positive_ctrl = element_effects[~element_effects["element"].str.contains("ntc")]
    element_effects_dict = {
        "negative_control": element_effects_negative_ctrl,
        "positive_control": element_effects_positive_ctrl,
    }
    return element_effects_dict


def get_n_steps_static(
    mdata,
    rna_modality="rna",
    max_steps: int | None = 400,
):
    """Get number of training steps according to sample size. training steps decrease with increasing sample size."""
    n_steps = min(
        max_steps, round(max_steps * (20000 / mdata[rna_modality].X.shape[0]))
    )  # if ncells > 20000 then n_steps decay
    n_steps = max(n_steps, 1)

    return n_steps


def get_alpha_empirical(alpha: float | None = 0.05, test_side: str | None = "both", p_val_list=None):
    """
    Get the empirical alpha from the empirical p-values of the control pairs.

    Parameters
    ----------
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
        alpha_empirical = p_val_list.quantile(alpha / 2)
    elif test_side == "single":
        alpha_empirical = p_val_list.quantile(alpha)

    return alpha_empirical


def get_alpha_empirical_for_each_pair(
    alpha: float | None = 0.05, test_side: str | None = "both", element_effects_split=None
):
    """
    Get the empirical alpha from the empirical p-values of the control pairs.

    Parameters
    ----------
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
            alpha_empirical = p_val_list.quantile(alpha / 2)
        elif test_side == "single":
            alpha_empirical = p_val_list.quantile(alpha)

        gene_id = np.where(unique_genes == gene)[0][0]
        alpha_empirical_list[gene_id] = alpha_empirical

    df_pos["alpha_empirical"] = alpha_empirical_list

    element_effects_split_modified = element_effects_split.copy()
    element_effects_split_modified["positive_control"] = df_pos

    return element_effects_split_modified


def create_empty_list(method):
    list_dict = {
        "Gene_id": [],
        "Element_id": [],
        "Gene_Mean": [],
        "Gene_Disp": [],
        "NCellsPerGRNA": [],
        "LogFoldChange": [],
        "MeanReads": [],
        "LFC_hat": [],
        "P_value": [],
        "alpha_cor": [],
        "Efficacy": [],
        "Method": [],
        "MTmethod": [],
        # "Likelihood": [],
        # "Effect_Prior": [],
        # "Merge_Guides": []
    }
    if method == "wilcoxon":
        del list_dict["LFC_hat"]

    return list_dict


def update_detailed_output(
    list_dict=None,  # a dictionary of list that we will extend onto
    element_effects=None,  # an output table of the model, saving loc/z-value/p-vlaue of each tested element-gene pair
    method="perturbo",  # glm/SECPTRE/wilcoxo, when here is wilcoxon, then do not include LFC_hat_list
    # nguides_per_element=4,
    ngenes=100,
    gene_mean=None,  # a pd.Series of gene mean values, len = ngenes
    gene_disp=None,  # a pd.Series of gene total_count values, len = ngenes
    ncells_per_guide=100,
    lfc=1,
    mean_reads_per_gene=5,
    alpha_base=0.1,  # write another function to create this
    guide_efficacy_values=None,
    MTmethod="none",  # "none"/"FDR"/"FWER"
    # likelihood="lnnb",
    # effect_prior_dist="normal",
    # merge_guides_mode="partial"
):
    if guide_efficacy_values is None:
        guide_efficacy_values = [1, 2 / 3, 1 / 3, 0]
    list_dict["Gene_id"].extend(element_effects["gene"])
    list_dict["Element_id"].extend(element_effects["element"])
    gene_id = element_effects["gene"]
    # print(f"gene_id has length {len(gene_id)}")

    list_dict["Gene_Mean"].extend(gene_mean[gene_id])
    list_dict["Gene_Disp"].extend(gene_disp[gene_id])

    # add some fixed parameters to the table
    npairs = len(element_effects["gene"])  # number of pairs passed filtering
    list_dict["NCellsPerGRNA"].extend([ncells_per_guide] * npairs)
    list_dict["LogFoldChange"].extend([lfc] * npairs)
    list_dict["MeanReads"].extend([mean_reads_per_gene] * npairs)

    if method == "perturbo" or method == "glm":
        LFC_hats = [x * np.log2(np.exp(1)) if x is not None else None for x in element_effects["loc"]]
        list_dict["LFC_hat"].extend(LFC_hats)
    elif method == "sceptre":
        LFC_hats = [x if x is not None else None for x in element_effects["loc"]]
        list_dict["LFC_hat"].extend(LFC_hats)

    list_dict["P_value"].extend(element_effects["q_value"])

    alpha_cor = get_alpha_corrected(
        alpha_base=alpha_base, MTmethod=MTmethod, element_effects=element_effects, ngenes=ngenes
    )
    list_dict["alpha_cor"].extend([alpha_cor] * npairs)
    list_dict["Efficacy"].extend([str([round(x, 2) for x in guide_efficacy_values])] * npairs)
    list_dict["Method"].extend([method] * npairs)
    list_dict["MTmethod"].extend([MTmethod] * npairs)
    # list_dict["Likelihood"].extend([likelihood] * npairs)
    # list_dict["Effect_Prior"].extend([effect_prior_dist] * npairs)
    # list_dict["Merge_Guides"].extend([merge_guides_mode] * npairs)

    return list_dict


def get_alpha_corrected(
    alpha_base=0.05,
    MTmethod="none",  # "none"/"FDR"/"FWER"/"FDR_old"
    element_effects=None,
    p_value_col="q_value",
    ngenes=100,
):
    if MTmethod == "none":
        alpha_cor = alpha_base
    elif MTmethod == "FDR_old":
        # pvals = element_effects.loc[0:ngenes, "q_value"]  # a list of p_values
        pvals = element_effects[element_effects["element"].str.contains("gene")][p_value_col]
        pvals_no_an = pvals[~np.isnan(pvals)]
        rejected, pvals_corrected, _, _ = multipletests(pvals_no_an, alpha=alpha_base, method="fdr_bh")
        alpha_cor = max(pvals_no_an[rejected]) if any(rejected) else alpha_base
    elif MTmethod == "FDR":
        pvals = element_effects[element_effects["element"].str.contains("gene")][p_value_col]
        assert not pvals.isna().any(), f"NaN values found in p-value column: '{p_value_col}'"
        rejected, pvals_corrected, _, alpha_bh = multipletests(pvals, alpha=alpha_base, method="fdr_bh")
        alpha_cor = pvals[rejected].max() if rejected.any() else alpha_bh
    elif MTmethod == "FWER":
        alpha_cor = alpha_base / ngenes

    return alpha_cor


def get_power_sum(
    power_detail,  # a dataframes of power details
    group_columns,
    test_type="fixed",  # "fixed"/"empirical"
):
    positive_pairs = power_detail[power_detail["Element_id"].str.contains("gene")]
    negative_pairs = power_detail[power_detail["Element_id"].str.contains("ntc")]

    if test_type == "fixed":
        positive_pairs["significance"] = positive_pairs["P_value"] <= positive_pairs["alpha_cor"]

    elif test_type == "empirical":
        results = []

        # Loop through each unique combination of values in the specified columns
        for _, pos_group in positive_pairs.groupby(group_columns):
            # Get the same group from negative_pairs
            neg_group = negative_pairs[
                (negative_pairs["NCellsPerGRNA"] == pos_group["NCellsPerGRNA"].iloc[0])
                & (negative_pairs["LogFoldChange"] == pos_group["LogFoldChange"].iloc[0])
                & (negative_pairs["MeanReads"] == pos_group["MeanReads"].iloc[0])
                & (negative_pairs["Efficacy"] == pos_group["Efficacy"].iloc[0])
                & (negative_pairs["Method"] == pos_group["Method"].iloc[0])
                & (negative_pairs["MTmethod"] == pos_group["MTmethod"].iloc[0])
                & (negative_pairs["alpha_cor"] == pos_group["alpha_cor"].iloc[0])
            ]

            if not neg_group.empty:
                # Step 2: Compute the k-quantile for the negative_pairs
                k = pos_group["alpha_cor"].iloc[0]  # Since alpha_cor is unique within the group
                alpha_emp = neg_group["P_value"].quantile(k)

                # Step 3: Compare P_value with alpha_emp in positive_pairs and save the result
                pos_group["significance"] = pos_group["P_value"] <= alpha_emp

                # Append the result to the list
                results.append(pos_group)

            positive_pairs = pd.concat(results)

    power_summary = positive_pairs.groupby(group_columns).agg(Power=("significance", "mean")).reset_index()

    return power_summary
