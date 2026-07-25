"""Tests for CLI/API covariate preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from perturbo.api import apply_covariate_transform, fit_covariate_transform


def test_fit_covariate_transform_count_like_uses_log1p_zscore() -> None:
    obs = pd.DataFrame({"guide_count": [0, 1, 2, 10, np.nan, 3]})
    state = fit_covariate_transform(obs, continuous_covariates=["guide_count"], batch_covariate=None)
    assert state.continuous_transforms["guide_count"] == "log1p+zscore"
    matrix, names = apply_covariate_transform(obs, state)
    assert names == ["guide_count"]
    assert matrix.shape == (obs.shape[0], 1)
    assert np.isfinite(matrix).all()


def test_fit_covariate_transform_non_count_uses_zscore() -> None:
    obs = pd.DataFrame({"percent_mito": [0.1, 0.2, 0.15, 0.4, np.nan]})
    state = fit_covariate_transform(obs, continuous_covariates=["percent_mito"], batch_covariate=None)
    assert state.continuous_transforms["percent_mito"] == "zscore"
    matrix, names = apply_covariate_transform(obs, state)
    assert names == ["percent_mito"]
    assert matrix.shape == (obs.shape[0], 1)


def test_fit_covariate_transform_missing_values_median_impute() -> None:
    obs = pd.DataFrame({"x": [1.0, np.nan, 5.0, np.nan, 3.0]})
    state = fit_covariate_transform(obs, continuous_covariates=["x"], batch_covariate=None)
    assert np.isclose(state.continuous_medians["x"], 3.0)
    matrix, _ = apply_covariate_transform(obs, state)
    assert np.isfinite(matrix).all()


def test_fit_covariate_transform_batch_one_hot_with_missing() -> None:
    obs = pd.DataFrame(
        {
            "batch": ["A", "A", "B", None, "C", "B"],
            "x": [0, 1, 2, 3, 4, 5],
        }
    )
    state = fit_covariate_transform(obs, continuous_covariates=["x"], batch_covariate="batch")
    matrix, names = apply_covariate_transform(obs, state)
    assert matrix.shape[0] == obs.shape[0]
    assert any(name.startswith("batch:batch=") for name in names)
    assert "__missing__" in state.batch_levels
    assert state.batch_reference is not None


def test_fit_covariate_transform_drops_zero_variance_columns() -> None:
    obs = pd.DataFrame(
        {
            "constant": [5, 5, 5, 5],
            "signal": [1, 2, 3, 4],
            "batch": ["x", "x", "x", "x"],
        }
    )
    state = fit_covariate_transform(obs, continuous_covariates=["constant", "signal"], batch_covariate="batch")
    assert "constant" in state.dropped_features
    assert state.feature_names == ["signal"]
    matrix, names = apply_covariate_transform(obs, state)
    assert names == ["signal"]
    assert matrix.shape == (obs.shape[0], 1)


def test_apply_covariate_transform_preserves_feature_order_shape() -> None:
    train_obs = pd.DataFrame(
        {
            "guide_count": [0, 1, 2, 3],
            "percent_mito": [0.1, 0.3, 0.2, 0.4],
            "prep_batch": ["b1", "b2", "b1", "b3"],
        }
    )
    state = fit_covariate_transform(
        train_obs,
        continuous_covariates=["percent_mito", "guide_count"],
        batch_covariate="prep_batch",
    )
    new_obs = pd.DataFrame(
        {
            "guide_count": [5, 4, 3],
            "percent_mito": [0.11, 0.29, 0.25],
            "prep_batch": ["b3", "b2", "new_batch"],
        }
    )
    matrix, names = apply_covariate_transform(new_obs, state)
    assert names == state.feature_names
    assert matrix.shape == (new_obs.shape[0], len(state.feature_names))


def test_apply_covariate_transform_missing_column_raises() -> None:
    train_obs = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    state = fit_covariate_transform(train_obs, continuous_covariates=["x"], batch_covariate=None)
    with pytest.raises(KeyError, match="x"):
        apply_covariate_transform(pd.DataFrame({"y": [1.0, 2.0]}), state)
