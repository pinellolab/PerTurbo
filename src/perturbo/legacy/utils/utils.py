import numpy as np
import pandas as pd
import anndata as ad
from statsmodels.stats.multitest import multipletests
from typing import Literal

def compute_empirical_pvals(
    data_real: pd.DataFrame,
    data_shuffled: pd.DataFrame,
    value_col: str ="z_value",
    adata: ad.AnnData | None = None,
    adata_count_col: str | None = None,
    pval_adj_method: str | None = None,
    group_col: str | None = None,
    two_sided: bool = True,
    bias_correction: bool = True,
    winsor: float | None = None,
    method: Literal["default", "tnull_fixed0"] = "default",
    n_quantiles: int | None = None,
):
    """
    Compute empirical p-values for real data based on null distribution from shuffled data.

    Parameters
    ----------
    data_real : pd.DataFrame
        DataFrame containing real test statistics.
    data_shuffled : pd.DataFrame
        DataFrame containing null test statistics.
    value_col : str, default "z_value"
        Column name for test statistics in both DataFrames.
    adata : AnnData, optional
        AnnData object for computing expression quantiles if n_quantiles is specified.
    adata_count_col : str, optional
        Column in adata.var to use for quantile computation.
    pval_adj_method : str or None, optional
        Method for multiple testing correction (e.g., 'fdr_bh'). If None,
        no adjustment is performed.
    group_col : str or None, optional
        Column name to group by when computing p-values. If None, compute
        p-values globally.
    two_sided : bool, default True
        If True, compute two-sided p-values; otherwise one-sided.
    bias_correction : bool, default True
        If True, apply bias correction in empirical p-value calculation.
    winsor : float in (0,0.5) or None, optional
        If specified, winsorize null statistics at these quantiles before fitting.
    method : str, default "default"
        Method for p-value computation. Options are "default" or "tnull_fixed0".
    n_quantiles : int or None, optional
        If specified, compute expression quantiles and use them as group_col.
    """

    if method == "default":
        pval_func = empirical_pvals_from_null
    elif method == "tnull_fixed0":
        pval_func = empirical_pvals_from_tnull_fixed0
    else:
        raise ValueError(f"Unknown method: {method}")

    if n_quantiles is not None:
        assert adata is not None, "adata must be provided when n_quantiles is specified."
        expression_quantiles = compute_quantiles(
            adata=adata,
            counts_col=None,
            n_quantiles=n_quantiles,
        )
        data_real = data_real.merge(expression_quantiles, left_on="gene", right_on="gene", how="left")
        data_shuffled = data_shuffled.merge(expression_quantiles, left_on="gene", right_on="gene", how="left")
        group_col = "mean_quantile"

    if group_col is None:
        pvals = pval_func(
            null_z=data_shuffled[value_col].values,
            real_z=data_real[value_col].values,
            two_sided=two_sided,
            bias_correction=bias_correction,
            winsor=winsor,
        )
        if pval_adj_method is not None:
            rej, pval_adj, _, _ = multipletests(pvals, alpha=0.05, method=pval_adj_method)
            return pval_adj
        else:
            return pvals
    else:
        pvals = data_real.groupby(group_col)[value_col].transform(
            lambda x: pval_func(
                null_z=data_shuffled.query(f"{group_col} == @x.name")[value_col],
                real_z=x,
                two_sided=two_sided,
                bias_correction=bias_correction,
                winsor=winsor,
            )
        )
        if pval_adj_method is not None:
            pval_df = data_real[[group_col]].assign(pval=pvals)
            pval_adj = pval_df.groupby(group_col)["pval"].transform(
                lambda x: multipletests(x, alpha=0.05, method=pval_adj_method)[1]
            )
            return pval_adj
        else:
            return pvals


def empirical_pvals_from_tnull_fixed0(
    null_z,
    real_z,
    two_sided: bool = True,
    winsor: float | None = None,
    return_params: bool = False,
    **kwargs,
):
    """
    Fit a Student-t(df, scale) with location fixed at 0 to null z-values,
    then compute parametric (empirical) p-values for real z-values.

    Parameters
    ----------
    null_z : array-like or pandas.Series
        Null z-values (1D).
    real_z : array-like or pandas.Series
        Observed z-values to evaluate.
    two_sided : bool, default True
        If True, two-sided p = 2 * sf(|z|); otherwise one-sided upper tail.
    winsor : float in (0,0.5), optional
        Winsorize null_z at quantiles (winsor, 1-winsor) before fitting.
    return_params : bool, default False
        If True, return (pvals, params_dict).

    Returns
    -------
    pvals : Series or ndarray
        Empirical p-values under fitted t(df, loc=0, scale).
    params : dict, optional
        {'df': df, 'scale': scale, 'n_null': B}
    """
    from scipy import stats

    # --- clean nulls
    null = pd.Series(null_z, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if null.size == 0:
        raise ValueError("No valid null statistics.")
    B = null.size

    # Optional winsorization for stability
    if winsor is not None:
        if not (0 < winsor < 0.5):
            raise ValueError("winsor must be in (0,0.5)")
        q_lo, q_hi = null.quantile([winsor, 1 - winsor])
        null = null.clip(q_lo, q_hi)

    # --- fit Student-t with loc fixed at 0
    df_hat, loc_hat, scale_hat = stats.t.fit(null.values, floc=0.0)
    # loc_hat will be 0.0 by construction

    # --- prep real values
    if isinstance(real_z, pd.Series):
        real = real_z.astype(float).replace([np.inf, -np.inf], np.nan)
        real_index = real.index
        real = real.values
        return_series = True
    else:
        real = np.asarray(real_z, dtype=float)
        real_index = None
        return_series = False

    # --- compute p-values
    z_std = real / scale_hat
    if two_sided:
        p = 2.0 * stats.t.sf(np.abs(z_std), df_hat)
    else:
        p = stats.t.sf(z_std, df_hat)
    p = np.clip(p, 0.0, 1.0)

    if return_series:
        p = pd.Series(p, index=real_index, name="p_tnull")

    if return_params:
        params = {"df": float(df_hat), "scale": float(scale_hat), "n_null": int(B)}
        return p, params
    return p


def empirical_pvals_from_null(
    null_z,
    real_z,
    two_sided: bool = True,
    bias_correction: bool = True,
    **kwargs,
):
    """
    Compute empirical p-values from a pooled null of z-like statistics.

    Parameters
    ----------
    null_z : array-like or pandas.Series
        1-D collection of null test statistics (e.g., from shuffled data).
    real_z : array-like or pandas.Series
        1-D collection of observed test statistics to evaluate.
    two_sided : bool, default True
        If True, uses |z| (two-sided). If False, uses one-sided (upper tail).
    bias_correction : bool, default True
        If True, uses (r+1)/(B+1). If False, uses r/B.

    Returns
    -------
    pvals : pandas.Series or np.ndarray
        Empirical p-values aligned to real_z (Series preserves index).
    """
    # Convert & clean
    null = pd.Series(null_z).astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
    if null.size == 0:
        raise ValueError("No valid null statistics provided.")

    if isinstance(real_z, pd.Series):
        real = real_z.astype(float).replace([np.inf, -np.inf], np.nan)
        real_index = real.index
        real = real.values
        return_series = True
    else:
        real = np.asarray(real_z, dtype=float)
        real_index = None
        return_series = False

    # Transform for sidedness
    if two_sided:
        null_t = np.abs(null)
        real_t = np.abs(real)
    else:
        null_t = null
        real_t = real

    # Sort null once (ascending)
    null_sorted = np.sort(null_t)
    B = null_sorted.size

    # For each real stat, count null >= real (upper tail)
    # searchsorted gives index of first value >= real_t (with 'left'),
    # so r = B - idx
    idx = np.searchsorted(null_sorted, real_t, side="left")
    r = B - idx  # exceedance counts

    if bias_correction:
        p = (r + 1.0) / (B + 1.0)
    else:
        # Guard against division by zero if B==0 (already caught above)
        p = r / B

    # Preserve index/type if input was a Series
    if return_series:
        return pd.Series(p, index=real_index, name="empirical_p")
    return p


def compute_quantiles(adata, counts_col=None, gene_name_col="gene", n_quantiles=10):
    if counts_col is None:
        print("computing mean counts for quantile assignment and outputting to 'mean_expression' column")
        mean_counts = adata.X.sum(axis=0) / adata.n_obs
        if isinstance(mean_counts, np.matrix):
            mean_counts = np.asarray(mean_counts).squeeze()
        adata.var["mean_expression"] = mean_counts
        counts_col = "mean_expression"

    gene_df = adata.var.reset_index(names=[gene_name_col])
    gene_df["mean_quantile"] = pd.qcut(gene_df[counts_col], n_quantiles, labels=False) + 1
    decile_df = gene_df[[gene_name_col, "mean_quantile"]]
    return decile_df


