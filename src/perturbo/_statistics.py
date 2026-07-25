"""Statistical helpers used by PerTurbo result tables.

These functions intentionally live outside the research diagnostics package so
the production result API has no dependency on applications or benchmarks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def empirical_pvals_from_null(
    null_z,
    real_z,
    *,
    two_sided: bool = True,
    bias_correction: bool = True,
):
    null = pd.Series(null_z, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if null.size == 0:
        raise ValueError("No valid null statistics provided.")

    if isinstance(real_z, pd.Series):
        real = real_z.astype(float).replace([np.inf, -np.inf], np.nan)
        real_index = real.index
        real_values = real.to_numpy()
        return_series = True
    else:
        real_values = np.asarray(real_z, dtype=float)
        real_index = None
        return_series = False

    if two_sided:
        null_t = np.abs(null)
        real_t = np.abs(real_values)
    else:
        null_t = null
        real_t = real_values

    null_sorted = np.sort(null_t)
    idx = np.searchsorted(null_sorted, real_t, side="left")
    exceedances = null_sorted.size - idx
    if bias_correction:
        pvals = (exceedances + 1.0) / (null_sorted.size + 1.0)
    else:
        pvals = exceedances / null_sorted.size

    if return_series:
        return pd.Series(pvals, index=real_index, name="empirical_p")
    return pvals


def empirical_pvals_from_tnull_fixed0(
    null_z,
    real_z,
    *,
    two_sided: bool = True,
    winsor: float | None = None,
    return_params: bool = False,
):
    null = pd.Series(null_z, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if null.empty:
        raise ValueError("No valid null statistics.")

    if winsor is not None:
        if not (0.0 < winsor < 0.5):
            raise ValueError("winsor must be in (0, 0.5).")
        q_lo, q_hi = null.quantile([winsor, 1.0 - winsor])
        null = null.clip(q_lo, q_hi)

    df_hat, _, scale_hat = stats.t.fit(null.to_numpy(), floc=0.0)

    if isinstance(real_z, pd.Series):
        real = real_z.astype(float).replace([np.inf, -np.inf], np.nan)
        real_index = real.index
        real_values = real.to_numpy()
        return_series = True
    else:
        real_values = np.asarray(real_z, dtype=float)
        real_index = None
        return_series = False

    z_std = real_values / scale_hat
    if two_sided:
        pvals = 2.0 * stats.t.sf(np.abs(z_std), df_hat)
    else:
        pvals = stats.t.sf(z_std, df_hat)
    pvals = np.clip(pvals, 0.0, 1.0)

    if return_series:
        pvals = pd.Series(pvals, index=real_index, name="p_tnull")
    if return_params:
        return pvals, {"df": float(df_hat), "scale": float(scale_hat), "n_null": int(null.size)}
    return pvals


def z_to_two_sided_pvalues(z_values) -> np.ndarray:
    z = np.nan_to_num(np.asarray(z_values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(2.0 * stats.norm.sf(np.abs(z)), 1e-12, 1.0)
