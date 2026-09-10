"""Phase-0 CRT surface: configuration gating and the baseline-mode diagnostic."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from perturbo.core import ControlFit, PerTurboData
from perturbo._internal.score_resampling import fit_batched_nb_null
from perturbo.crt import (
    assemble_control_nuisance,
    check_baseline_is_null_mode,
    control_block_for_genes,
    prepare_crt_baseline,
    validate_crt_config,
    validate_offset_compatibility,
)

SUPPORTED = dict(likelihood="nb", size_factor_mode="observed", num_factors=0)


def test_supported_configuration_is_accepted() -> None:
    validate_crt_config(**SUPPORTED)


@pytest.mark.parametrize("likelihood", ["nb", "negbin"])
def test_both_spellings_of_the_nb_likelihood_are_accepted(likelihood: str) -> None:
    validate_crt_config(likelihood=likelihood, size_factor_mode="observed")


@pytest.mark.parametrize("size_factor_mode", ["observed", "none"])
def test_fixed_offset_modes_are_accepted(size_factor_mode: str) -> None:
    validate_crt_config(likelihood="nb", size_factor_mode=size_factor_mode)


def test_latent_size_factors_are_rejected_with_the_reason_and_the_remedy() -> None:
    with pytest.raises(ValueError, match="size_factor_mode") as excinfo:
        validate_crt_config(likelihood="nb", size_factor_mode="infer")
    message = str(excinfo.value)
    assert "--size-factor-mode observed" in message
    # The point is not merely that it is unsupported but that the null would
    # stop being exact; a message that omits the reason invites a workaround.
    assert "exact" in message


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"likelihood": "lnnb"}, "likelihood"),
        ({"likelihood": "mixture_nb"}, "likelihood"),
        ({"likelihood": "censored_nb"}, "likelihood"),
        ({"num_factors": 5}, "num_factors"),
        ({"guide_random_effects": True}, "guide_random_effects"),
    ],
)
def test_unsupported_options_are_rejected(kwargs: dict, expected: str) -> None:
    with pytest.raises(ValueError, match="Unsupported configuration") as excinfo:
        validate_crt_config(**{**SUPPORTED, **kwargs})
    assert expected in str(excinfo.value)


def test_every_problem_is_reported_at_once() -> None:
    # A CRT run is gated behind a full stage-one fit, so reporting one problem
    # per run would cost a training run per mistake.
    with pytest.raises(ValueError) as excinfo:
        validate_crt_config(
            likelihood="lnnb",
            size_factor_mode="infer",
            num_factors=3,
            guide_random_effects=True,
            retain_guide_structure=True,
        )
    message = str(excinfo.value)
    assert message.count("  - ") == 4


def test_an_element_map_is_accepted_by_the_control_anchored_test() -> None:
    """Guide-structured designs are collapsed to elements and cells carrying more than
    one are set aside, so the map is no longer a configuration problem."""
    validate_crt_config(**{**SUPPORTED, "retain_guide_structure": True})


def test_zero_and_none_factors_both_mean_no_factors() -> None:
    validate_crt_config(**{**SUPPORTED, "num_factors": None})
    validate_crt_config(**{**SUPPORTED, "num_factors": 0})


def _simulate_controls(
    *,
    num_cells: int = 400,
    num_genes: int = 6,
    num_covariates: int = 0,
    seed: int = 0,
) -> tuple[PerTurboData, np.ndarray, np.ndarray, np.ndarray]:
    """Control cells from a known NB null, plus the coefficients that generated them."""

    rng = np.random.default_rng(seed)
    theta = rng.uniform(2.0, 8.0, size=num_genes)
    beta_0 = rng.uniform(0.5, 2.5, size=num_genes)
    offsets = rng.normal(0.0, 0.2, size=(num_cells, 1))

    covariates = None
    covariate_coef = None
    eta = offsets + beta_0[None, :]
    if num_covariates:
        covariates = rng.normal(0.0, 1.0, size=(num_cells, num_covariates))
        covariate_coef = rng.normal(0.0, 0.3, size=(num_covariates, num_genes))
        eta = eta + covariates @ covariate_coef

    mean = np.exp(eta)
    counts = rng.negative_binomial(theta[None, :], theta[None, :] / (theta[None, :] + mean))

    data = PerTurboData(
        counts=jnp.asarray(counts, dtype=jnp.float32),
        pert_id=jnp.zeros((num_cells,), dtype=jnp.int32),
        pert_names=["__control__"],
        gene_names=[f"gene_{index}" for index in range(num_genes)],
        size_factors=jnp.asarray(offsets, dtype=jnp.float32),
        covariates=None if covariates is None else jnp.asarray(covariates, dtype=jnp.float32),
        covariate_names=None if covariates is None else [f"cov_{i}" for i in range(num_covariates)],
    )
    return data, beta_0, theta, covariate_coef


def _fitted_coefficients(data: PerTurboData, theta: np.ndarray) -> np.ndarray:
    """The control cells' actual null mode, as ``(1 + covariates, genes)``.

    Stage one fits *these* cells, so a well-trained baseline estimates their
    empirical mode - which differs from the generating coefficients by
    O(1/sqrt(n)). Using the generating values as a stand-in for a converged fit
    would conflate sampling noise with baseline error, and at a few hundred
    cells the two are the same size.
    """

    counts = np.asarray(data.counts, dtype=np.float64)
    design = [np.ones((counts.shape[0], 1))]
    if data.covariates is not None:
        design.append(np.asarray(data.covariates, dtype=np.float64))
    return fit_batched_nb_null(
        counts,
        nuisance_design=np.concatenate(design, axis=1),
        offsets=np.asarray(data.size_factors, dtype=np.float64),
        theta=theta,
    ).nuisance_mean


def _fitted_control_fit(data: PerTurboData, theta: np.ndarray) -> ControlFit:
    """A ControlFit standing exactly at the control-cell null mode."""

    coefficients = _fitted_coefficients(data, theta)
    covariate_coef = coefficients[1:] if coefficients.shape[0] > 1 else None
    return _control_fit(coefficients[0], theta, covariate_coef)


def _control_fit(beta_0: np.ndarray, theta: np.ndarray, covariate_coef: np.ndarray | None) -> ControlFit:
    return ControlFit(
        beta_0=jnp.asarray(beta_0, dtype=jnp.float32),
        theta=jnp.asarray(theta, dtype=jnp.float32),
        noise_scale=jnp.zeros_like(jnp.asarray(theta, dtype=jnp.float32)),
        factor_loadings=None,
        factor_scores=None,
        factor_center=None,
        pca_loadings=None,
        size_factors=jnp.zeros((1, 1), dtype=jnp.float32),
        losses=jnp.zeros((1,), dtype=jnp.float32),
        svi_result=None,
        covariate_coef=None if covariate_coef is None else jnp.asarray(covariate_coef, dtype=jnp.float32),
    )


def test_nuisance_assembly_stacks_intercept_above_covariates() -> None:
    data, beta_0, theta, covariate_coef = _simulate_controls(num_covariates=3)
    nuisance = assemble_control_nuisance(data, _control_fit(beta_0, theta, covariate_coef))

    assert nuisance.nuisance_design.shape == (400, 4)
    assert nuisance.coefficients.shape == (4, 6)
    assert nuisance.nuisance_names == ("intercept", "cov_0", "cov_1", "cov_2")
    # The intercept column is all ones and its coefficient row is beta_0; the
    # covariate block follows in the order the transform produced.
    np.testing.assert_array_equal(nuisance.nuisance_design[:, 0], np.ones(400))
    np.testing.assert_allclose(nuisance.coefficients[0], beta_0, rtol=1e-6)
    np.testing.assert_allclose(nuisance.coefficients[1:], covariate_coef, rtol=1e-6)


def test_nuisance_assembly_without_covariates_is_intercept_only() -> None:
    data, beta_0, theta, _ = _simulate_controls()
    nuisance = assemble_control_nuisance(data, _control_fit(beta_0, theta, None))
    assert nuisance.nuisance_design.shape == (400, 1)
    assert nuisance.nuisance_names == ("intercept",)


def test_nuisance_assembly_rejects_covariates_without_coefficients() -> None:
    data, beta_0, theta, _ = _simulate_controls(num_covariates=2)
    with pytest.raises(ValueError, match="covariate_coef is missing"):
        assemble_control_nuisance(data, _control_fit(beta_0, theta, None))


def test_nuisance_assembly_rejects_a_dispersion_of_the_wrong_length() -> None:
    data, beta_0, theta, _ = _simulate_controls()
    with pytest.raises(ValueError, match="control_fit.theta has"):
        assemble_control_nuisance(data, _control_fit(beta_0, theta[:-1], None))


@pytest.mark.parametrize("num_covariates", [0, 3])
def test_a_fitted_baseline_sits_at_the_null_mode(num_covariates: int) -> None:
    data, _, theta, _ = _simulate_controls(num_cells=800, num_covariates=num_covariates)
    check = check_baseline_is_null_mode(
        assemble_control_nuisance(data, _fitted_control_fit(data, theta))
    )

    assert check.ok, check.describe()
    # A baseline standing on the mode has essentially no step to take, whatever
    # the cell count - the residual is solver tolerance, not sampling noise.
    assert check.newton_step.max() < 1e-6


def test_a_displaced_baseline_is_caught_and_named() -> None:
    data, _, theta, _ = _simulate_controls(num_cells=800)
    fitted = _fitted_coefficients(data, theta)
    displaced = fitted[0].copy()
    displaced[2] += 0.75
    check = check_baseline_is_null_mode(assemble_control_nuisance(data, _control_fit(displaced, theta, None)))

    assert not check.ok
    np.testing.assert_array_equal(np.flatnonzero(check.failed), [2])
    assert "gene_2" in check.describe()


def test_the_step_reads_as_nats_of_displacement_near_the_mode() -> None:
    # Near the mode the NB log-likelihood is approximately quadratic, so one
    # Fisher step recovers the displacement and the reported number can be read
    # directly as "how far off, in nats of the linear predictor".
    data, _, theta, _ = _simulate_controls(num_cells=800)
    fitted = _fitted_coefficients(data, theta)
    displaced = fitted[0].copy()
    displaced[1] += 0.02
    check = check_baseline_is_null_mode(assemble_control_nuisance(data, _control_fit(displaced, theta, None)))

    assert check.newton_step[1] == pytest.approx(0.02, rel=0.05)


def test_the_step_understates_a_large_displacement() -> None:
    # Away from the mode a single step undershoots, so the diagnostic is a lower
    # bound on the true distance rather than an estimate of it. That direction is
    # the safe one for a guard - it never invents a failure - but it does mean a
    # step near tolerance should not be read as "only just off".
    data, _, theta, _ = _simulate_controls(num_cells=800)
    fitted = _fitted_coefficients(data, theta)
    displaced = fitted[0].copy()
    displaced[1] += 1.5
    check = check_baseline_is_null_mode(assemble_control_nuisance(data, _control_fit(displaced, theta, None)))

    assert 0.0 < check.newton_step[1] < 1.5
    assert not check.ok


def test_the_verdict_tightens_with_the_tolerance() -> None:
    data, _, theta, _ = _simulate_controls(num_cells=800)
    fitted = _fitted_coefficients(data, theta)
    displaced = fitted[0].copy()
    displaced[0] += 0.02
    nuisance = assemble_control_nuisance(data, _control_fit(displaced, theta, None))

    assert check_baseline_is_null_mode(nuisance, step_tolerance=0.05).ok
    assert not check_baseline_is_null_mode(nuisance, step_tolerance=0.005).ok


def test_genes_with_no_control_counts_are_excluded_rather_than_failed() -> None:
    data, beta_0, theta, _ = _simulate_controls(num_cells=500)
    counts = np.asarray(data.counts).copy()
    counts[:, 1] = 0.0
    data = PerTurboData(
        counts=jnp.asarray(counts),
        pert_id=data.pert_id,
        pert_names=data.pert_names,
        gene_names=data.gene_names,
        size_factors=data.size_factors,
    )
    check = check_baseline_is_null_mode(assemble_control_nuisance(data, _control_fit(beta_0, theta, None)))

    # No baseline can sit at a mode that does not exist, so this must not be
    # reported as a fixable failure.
    assert bool(check.degenerate[1])
    assert not bool(check.failed[1])
    assert "1 degenerate gene(s) excluded" in check.describe()


def test_check_rejects_a_nonpositive_tolerance() -> None:
    data, beta_0, theta, _ = _simulate_controls(num_cells=200)
    nuisance = assemble_control_nuisance(data, _control_fit(beta_0, theta, None))
    with pytest.raises(ValueError, match="step_tolerance"):
        check_baseline_is_null_mode(nuisance, step_tolerance=0.0)


def test_prepare_baseline_accepts_a_converged_stage_one_fit() -> None:
    data, _, theta, _ = _simulate_controls(num_cells=4000, num_covariates=2)
    baseline = prepare_crt_baseline(data, _fitted_control_fit(data, theta))

    assert baseline.null_check.ok
    assert baseline.num_genes == 6
    assert baseline.num_control_cells == 4000
    assert baseline.nuisance.nuisance_names == ("intercept", "cov_0", "cov_1")


def test_prepare_baseline_refuses_a_displaced_stage_one_fit() -> None:
    # Reuse-only means this check is the sole guard between an under-trained
    # stage one and a statistic that is no longer the efficient score.
    data, _, theta, _ = _simulate_controls(num_cells=4000)
    displaced = _fitted_coefficients(data, theta)[0] + 0.6
    with pytest.raises(ValueError, match="not at the control-cell null mode"):
        prepare_crt_baseline(data, _control_fit(displaced, theta, None))


def test_prepare_baseline_can_downgrade_the_refusal_to_a_warning() -> None:
    data, _, theta, _ = _simulate_controls(num_cells=4000)
    displaced = _fitted_coefficients(data, theta)[0] + 0.6
    with pytest.warns(RuntimeWarning, match="not at the control-cell null mode"):
        baseline = prepare_crt_baseline(data, _control_fit(displaced, theta, None), strict=False)
    assert not baseline.null_check.ok


def test_control_block_carries_a_nonzero_nuisance_score_under_a_variational_baseline() -> None:
    # The pooled null can drop Z'r because its own score equation zeroes it. A
    # reused SVI median does not, which is exactly why the term is carried.
    data, _, theta, _ = _simulate_controls(num_cells=2000)
    displaced = _fitted_coefficients(data, theta)[0] + 0.01
    baseline = prepare_crt_baseline(data, _control_fit(displaced, theta, None))
    block = control_block_for_genes(baseline)

    assert block.nuisance_score.shape == (1, 6)
    assert np.all(np.abs(block.nuisance_score) > 0.0)


def test_control_block_shapes_and_slicing_are_exact() -> None:
    data, _, theta, _ = _simulate_controls(num_cells=800, num_covariates=2)
    baseline = prepare_crt_baseline(data, _fitted_control_fit(data, theta))

    full = control_block_for_genes(baseline)
    assert full.score_residual.shape == (800, 6)
    assert full.observation_weight.shape == (800, 6)
    assert full.information.shape == (6, 3, 3)
    assert full.nuisance_score.shape == (3, 6)

    # Gene-axis slicing is a decomposition, not an approximation: a slice must
    # reproduce the full-panel columns bit for bit.
    sliced = control_block_for_genes(baseline, slice(2, 5))
    assert sliced.gene_names == ("gene_2", "gene_3", "gene_4")
    np.testing.assert_array_equal(sliced.score_residual, full.score_residual[:, 2:5])
    np.testing.assert_array_equal(sliced.observation_weight, full.observation_weight[:, 2:5])
    np.testing.assert_array_equal(sliced.information, full.information[2:5])
    np.testing.assert_array_equal(sliced.nuisance_score, full.nuisance_score[:, 2:5])


def _with_center(data: PerTurboData, center: float | None) -> PerTurboData:
    return PerTurboData(
        counts=data.counts,
        pert_id=data.pert_id,
        pert_names=data.pert_names,
        gene_names=data.gene_names,
        size_factors=data.size_factors,
        library_size_center_log_mean=center,
    )


def test_matching_offset_centering_is_accepted() -> None:
    data, *_ = _simulate_controls(num_cells=100)
    validate_offset_compatibility(_with_center(data, 7.5), _with_center(data, 7.5))
    validate_offset_compatibility(_with_center(data, None), _with_center(data, None))


def test_mismatched_offset_centering_is_rejected() -> None:
    data, *_ = _simulate_controls(num_cells=100)
    with pytest.raises(ValueError, match="centered differently"):
        validate_offset_compatibility(_with_center(data, 7.5), _with_center(data, 7.9))


def test_a_half_present_offset_centering_is_rejected() -> None:
    data, *_ = _simulate_controls(num_cells=100)
    with pytest.raises(ValueError, match="only one"):
        validate_offset_compatibility(_with_center(data, 7.5), _with_center(data, None))
