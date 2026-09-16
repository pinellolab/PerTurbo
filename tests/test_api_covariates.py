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


def _screen_with_an_unsampled_control_lane() -> tuple[pd.DataFrame, pd.Series]:
    """A screen whose controls miss one whole lane.

    Lane ``L3`` holds analysed cells but no control cell, which is the shape of
    the TAP-seq chr8 screen: its non-targeting guides were sequenced almost
    entirely in one lane, so the stage-one control set never sampled lane
    ``SRX6665869`` at all.
    """
    analysed = pd.Series(["L0"] * 400 + ["L1"] * 300 + ["L2"] * 200 + ["L3"] * 250)
    control_obs = pd.DataFrame(
        {
            "batch": ["L2"] * 120 + ["L0"] * 4 + ["L1"] * 2,
            "percent_mito": np.linspace(0.0, 0.2, 126),
        }
    )
    return control_obs, analysed


def test_batch_levels_come_from_the_analysed_cells_not_the_controls() -> None:
    control_obs, analysed = _screen_with_an_unsampled_control_lane()
    state = fit_covariate_transform(
        control_obs,
        continuous_covariates=["percent_mito"],
        batch_covariate="batch",
        analysed_batch_values=analysed,
        design_refit_over_analysed_cells=True,
    )
    assert state.batch_levels_source == "analysed-cells"
    # Never leave a level that exists in the data unlisted.
    assert state.batch_all_levels == ["L0", "L1", "L2", "L3"]
    assert state.batch_level_counts == {"L0": 400, "L1": 300, "L2": 200, "L3": 250}
    # The reference is the most frequent analysed level, not the most frequent
    # control level, and the lane the controls missed keeps a design column.
    assert state.batch_reference == "L0"
    assert "batch:batch=L3" in state.feature_names
    assert state.dropped_features == []
    assert state.unidentifiable_batch_levels == ["L3"]

    analysed_obs = pd.DataFrame({"batch": analysed, "percent_mito": 0.05})
    matrix, names = apply_covariate_transform(analysed_obs, state)
    assert names == state.feature_names
    lane_column = matrix[:, names.index("batch:batch=L3")]
    assert lane_column.sum() == 250
    # No analysed cell is silently coded as the reference lane.
    batch_columns = [i for i, name in enumerate(names) if name.startswith("batch:")]
    coded_as_reference = matrix[:, batch_columns].sum(axis=1) == 0
    assert set(analysed[coded_as_reference]) == {"L0"}


def test_control_anchored_warns_and_records_an_unidentifiable_batch_level(capsys) -> None:
    control_obs, analysed = _screen_with_an_unsampled_control_lane()
    state = fit_covariate_transform(
        control_obs,
        continuous_covariates=["percent_mito"],
        batch_covariate="batch",
        analysed_batch_values=analysed,
        design_refit_over_analysed_cells=False,
    )
    # The control-only null cannot identify a lane with no control cell, so the
    # column goes - but the level is still enumerated and named out loud.
    assert state.batch_all_levels == ["L0", "L1", "L2", "L3"]
    assert state.unidentifiable_batch_levels == ["L3"]
    assert "batch:batch=L3" not in state.feature_names
    assert "batch:batch=L3" in state.dropped_features
    # The reference stays the most frequent control level: the intercept has to
    # be anchored on a level the estimating cells actually populate.
    assert state.batch_reference == "L2"
    warning = capsys.readouterr().out
    assert "L3" in warning
    assert "not identified" in warning


def test_legacy_call_without_analysed_levels_is_unchanged() -> None:
    control_obs, _ = _screen_with_an_unsampled_control_lane()
    state = fit_covariate_transform(
        control_obs,
        continuous_covariates=["percent_mito"],
        batch_covariate="batch",
    )
    assert state.batch_levels_source == "fit-cells"
    assert state.batch_reference == "L2"
    assert state.batch_levels == ["L0", "L1"]
    assert state.unidentifiable_batch_levels == []


def test_polish_recovers_the_coefficient_of_a_lane_the_controls_missed() -> None:
    """The all-cells refit identifies the lane, so its true effect comes back.

    This is the defect's consequence, measured: with the lane merged into the
    reference its cells sit at the reference's fitted mean, and the residuals
    that the CRT scores inherit the whole lane effect.
    """
    import dataclasses

    from perturbo.crt import ControlNuisance, polish_baseline_to_null_mode

    control_obs, analysed = _screen_with_an_unsampled_control_lane()
    state = fit_covariate_transform(
        control_obs,
        continuous_covariates=["percent_mito"],
        batch_covariate="batch",
        analysed_batch_values=analysed,
        design_refit_over_analysed_cells=True,
    )
    analysed_obs = pd.DataFrame({"batch": analysed, "percent_mito": 0.05})
    covariates, names = apply_covariate_transform(analysed_obs, state)

    num_cells = covariates.shape[0]
    design = np.concatenate([np.ones((num_cells, 1)), covariates], axis=1).astype(np.float64)
    truth = np.zeros((design.shape[1], 1))
    truth[0, 0] = 2.5
    lane_row = 1 + names.index("batch:batch=L3")
    truth[lane_row, 0] = -1.25
    truth[1 + names.index("batch:batch=L1"), 0] = 0.4

    rng = np.random.default_rng(0)
    theta = np.array([50.0])
    mean = np.exp(design @ truth)
    counts = rng.negative_binomial(theta[0], theta[0] / (theta[0] + mean)).astype(np.float32)

    nuisance = ControlNuisance(
        counts=counts,
        nuisance_design=design.astype(np.float32),
        coefficients=np.zeros_like(truth),
        offsets=np.zeros((num_cells, 1)),
        dispersion=theta,
        nuisance_names=("intercept", *names),
        gene_names=("g0",),
    )
    polished = polish_baseline_to_null_mode(nuisance)
    fitted = np.asarray(polished.coefficients)[:, 0]
    assert fitted[lane_row] == pytest.approx(truth[lane_row, 0], abs=0.15)

    # Without the column the lane's 250 cells are pinned to the reference mean.
    merged_design = np.delete(design, lane_row, axis=1)
    merged = dataclasses.replace(
        nuisance,
        nuisance_design=merged_design.astype(np.float32),
        coefficients=np.zeros((merged_design.shape[1], 1)),
        nuisance_names=tuple(n for i, n in enumerate(("intercept", *names)) if i != lane_row),
    )
    merged_fit = np.asarray(polish_baseline_to_null_mode(merged).coefficients)[:, 0]
    lane = (analysed == "L3").to_numpy()
    merged_mean = np.exp(merged_design @ merged_fit)[lane].mean()
    assert merged_mean > 2.0 * counts[lane, 0].mean()
