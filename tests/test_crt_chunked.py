"""Phase-3/4 CRT surface: baseline injection and perturbation-chunked execution.

The load-bearing test here is the parity one. Everything else in the lift-over
is refactoring; parity against the research path is what decides whether the
port preserved the test.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

import perturbo.crt as crt_module
from perturbo.core import ControlFit, PerTurboData
from perturbo.crt import (
    CRT_CONTROL_NAME,
    CRT_TAIL_FAMILIES,
    build_chunk_design,
    check_baseline_is_null_mode,
    fit_shared_propensity_coefficients,
    prepare_crt_baseline,
    run_crt_for_chunk,
)
from perturbo._internal.low_moi.design import prepare_low_moi_design
from perturbo._internal.score_resampling import (
    fit_batched_nb_null,
    precompute_low_moi_permutations,
    run_low_moi_score_permutations,
)


def _simulate_screen(
    *,
    num_control: int = 300,
    cells_per_target: int = 40,
    num_targets: int = 6,
    num_genes: int = 8,
    num_covariates: int = 1,
    seed: int = 3,
):
    """A small low-MOI screen, split the way production splits it.

    Controls come back as their own ``PerTurboData`` (stage one's input) and the
    targets as a second one (a stage-two chunk), because that separation - not
    the simulation - is what the chunked path has to cope with.
    """

    rng = np.random.default_rng(seed)
    theta = rng.uniform(3.0, 12.0, size=num_genes)
    beta_0 = rng.uniform(1.0, 2.5, size=num_genes)
    covariate_coef = rng.normal(0.0, 0.25, size=(num_covariates, num_genes))

    num_target_cells = cells_per_target * num_targets
    total = num_control + num_target_cells
    offsets = rng.normal(0.0, 0.2, size=(total, 1))
    covariates = rng.normal(0.0, 1.0, size=(total, num_covariates))

    codes = np.concatenate(
        [np.full(num_control, -1), np.repeat(np.arange(num_targets), cells_per_target)]
    )
    effect = np.zeros((num_targets, num_genes))
    effect[0, 2] = -1.1
    effect[3, 5] = 0.9

    eta = offsets + beta_0[None, :] + covariates @ covariate_coef
    targeted = codes >= 0
    eta[targeted] += effect[codes[targeted]]
    counts = rng.negative_binomial(theta[None, :], theta[None, :] / (theta[None, :] + np.exp(eta)))

    gene_names = [f"gene_{index}" for index in range(num_genes)]
    covariate_names = [f"cov_{index}" for index in range(num_covariates)]
    target_names = [f"target_{index}" for index in range(num_targets)]

    control_data = PerTurboData(
        counts=jnp.asarray(counts[:num_control], dtype=jnp.float32),
        pert_id=jnp.zeros((num_control,), dtype=jnp.int32),
        pert_names=["NTC"],
        gene_names=gene_names,
        size_factors=jnp.asarray(offsets[:num_control], dtype=jnp.float32),
        covariates=jnp.asarray(covariates[:num_control], dtype=jnp.float32),
        covariate_names=covariate_names,
        library_size_center_log_mean=0.0,
    )
    chunk_data = PerTurboData(
        counts=jnp.asarray(counts[num_control:], dtype=jnp.float32),
        pert_id=jnp.asarray(codes[num_control:], dtype=jnp.int32),
        pert_names=target_names,
        gene_names=gene_names,
        size_factors=jnp.asarray(offsets[num_control:], dtype=jnp.float32),
        covariates=jnp.asarray(covariates[num_control:], dtype=jnp.float32),
        covariate_names=covariate_names,
        library_size_center_log_mean=0.0,
    )
    strata = np.tile(np.asarray([0, 1]), total // 2)
    return control_data, chunk_data, theta, strata


def _fitted_control_fit(control_data: PerTurboData, theta: np.ndarray) -> ControlFit:
    """Stage one standing at the control-cell null mode, as a trained fit would."""

    counts = np.asarray(control_data.counts, dtype=np.float64)
    design = np.concatenate(
        [np.ones((counts.shape[0], 1)), np.asarray(control_data.covariates, dtype=np.float64)], axis=1
    )
    coefficients = fit_batched_nb_null(
        counts,
        nuisance_design=design,
        offsets=np.asarray(control_data.size_factors, dtype=np.float64),
        theta=theta,
    ).nuisance_mean
    return ControlFit(
        beta_0=jnp.asarray(coefficients[0], dtype=jnp.float32),
        theta=jnp.asarray(theta, dtype=jnp.float32),
        noise_scale=jnp.zeros(theta.size, dtype=jnp.float32),
        factor_loadings=None,
        factor_scores=None,
        factor_center=None,
        pca_loadings=None,
        size_factors=control_data.size_factors,
        losses=jnp.zeros((1,), dtype=jnp.float32),
        svi_result=None,
        covariate_coef=jnp.asarray(coefficients[1:], dtype=jnp.float32),
    )


def _research_design(control_data: PerTurboData, chunk_data: PerTurboData, theta: np.ndarray):
    """The same screen as one unsplit design, the way the research driver builds it."""

    num_control = int(np.asarray(control_data.counts).shape[0])
    codes = np.asarray(chunk_data.pert_id)
    labels = np.concatenate([np.zeros(num_control, dtype=np.int64), codes + 1])
    data = PerTurboData(
        counts=jnp.asarray(
            np.concatenate([np.asarray(control_data.counts), np.asarray(chunk_data.counts)], axis=0)
        ),
        pert_id=jnp.asarray(labels),
        pert_names=[CRT_CONTROL_NAME, *chunk_data.pert_names],
        gene_names=list(control_data.gene_names),
        size_factors=jnp.asarray(
            np.concatenate(
                [np.asarray(control_data.size_factors), np.asarray(chunk_data.size_factors)], axis=0
            )
        ),
        covariates=jnp.asarray(
            np.concatenate(
                [np.asarray(control_data.covariates), np.asarray(chunk_data.covariates)], axis=0
            )
        ),
        covariate_names=list(control_data.covariate_names),
    )
    return prepare_low_moi_design(
        data, control_perturbations=[CRT_CONTROL_NAME], dispersion=np.asarray(theta, dtype=np.float32)
    )


def test_injected_coefficients_reproduce_the_fit_they_replace() -> None:
    """Supplying the null coefficients must equal fitting them, given the same values.

    This isolates the injection mechanism from the question of *which* baseline
    to inject: feed back exactly what the internal fit would have produced and
    nothing may move.
    """

    control_data, chunk_data, theta, _ = _simulate_screen()
    design = _research_design(control_data, chunk_data, theta)

    fitted = run_low_moi_score_permutations(
        design, num_resamples=99, seed=4, backend="jax", null_model="control_only"
    )
    control_mask = np.asarray(design.control_mask)
    coefficients = fit_batched_nb_null(
        np.asarray(design.counts)[control_mask],
        nuisance_design=np.asarray(design.nuisance_design)[control_mask],
        offsets=np.asarray(design.offsets)[control_mask],
        theta=np.asarray(design.dispersion),
    ).nuisance_mean
    injected = run_low_moi_score_permutations(
        design,
        num_resamples=99,
        seed=4,
        backend="jax",
        null_model="control_only",
        nuisance_coefficients=coefficients,
    )

    np.testing.assert_allclose(
        np.asarray(injected.observed_score, dtype=np.float64),
        np.asarray(fitted.observed_score, dtype=np.float64),
        rtol=2e-3,
        atol=2e-4,
    )
    np.testing.assert_array_equal(np.asarray(injected.p_value), np.asarray(fitted.p_value))


def test_injection_rejects_the_wrong_shape() -> None:
    control_data, chunk_data, theta, _ = _simulate_screen()
    design = _research_design(control_data, chunk_data, theta)
    with pytest.raises(ValueError, match="nuisance_coefficients must have shape"):
        run_low_moi_score_permutations(
            design, num_resamples=9, backend="jax", nuisance_coefficients=np.zeros((3, 3))
        )


def test_injection_rejects_the_pooled_null() -> None:
    control_data, chunk_data, theta, _ = _simulate_screen()
    design = _research_design(control_data, chunk_data, theta)
    with pytest.raises(ValueError, match="requires backend='jax'"):
        run_low_moi_score_permutations(
            design,
            num_resamples=9,
            backend="batched",
            null_model="pooled",
            nuisance_coefficients=np.zeros((2, 8)),
        )


def test_the_chunk_design_stacks_controls_ahead_of_chunk_cells() -> None:
    control_data, chunk_data, theta, _ = _simulate_screen()
    baseline = prepare_crt_baseline(control_data, _fitted_control_fit(control_data, theta))
    design = build_chunk_design(baseline, chunk_data, control_data=control_data)

    codes = np.asarray(design.target_codes)
    assert design.target_names == tuple(chunk_data.pert_names)
    assert int(np.asarray(design.control_mask).sum()) == 300
    np.testing.assert_array_equal(codes[:300], np.full(300, -1))
    np.testing.assert_array_equal(codes[300:], np.repeat(np.arange(6), 40))


def test_the_chunk_design_refuses_mismatched_offset_centering() -> None:
    control_data, chunk_data, theta, _ = _simulate_screen()
    baseline = prepare_crt_baseline(control_data, _fitted_control_fit(control_data, theta))
    shifted = PerTurboData(
        counts=chunk_data.counts,
        pert_id=chunk_data.pert_id,
        pert_names=chunk_data.pert_names,
        gene_names=chunk_data.gene_names,
        size_factors=chunk_data.size_factors,
        covariates=chunk_data.covariates,
        covariate_names=chunk_data.covariate_names,
        library_size_center_log_mean=1.5,
    )
    with pytest.raises(ValueError, match="centered differently"):
        build_chunk_design(baseline, shifted, control_data=control_data)


def test_padded_chunk_cells_are_excluded_from_the_control_pool() -> None:
    """Padding must not become zero-count control cells.

    Fixed-shape chunk buffers append all-zero rows the model drops via
    ``cell_mask``. Left in, they would enter the pooled null as legitimate
    observations and dilute every target's reference distribution.
    """

    control_data, chunk_data, theta, _ = _simulate_screen()
    baseline = prepare_crt_baseline(control_data, _fitted_control_fit(control_data, theta))
    unpadded = build_chunk_design(baseline, chunk_data, control_data=control_data)

    pad = 25
    padded = PerTurboData(
        counts=jnp.asarray(
            np.concatenate([np.asarray(chunk_data.counts), np.zeros((pad, 8), dtype=np.float32)])
        ),
        pert_id=jnp.asarray(np.concatenate([np.asarray(chunk_data.pert_id), np.zeros(pad, dtype=np.int32)])),
        pert_names=chunk_data.pert_names,
        gene_names=chunk_data.gene_names,
        cell_mask=jnp.asarray(
            np.concatenate([np.ones(240, dtype=bool), np.zeros(pad, dtype=bool)])
        ),
        size_factors=jnp.asarray(
            np.concatenate([np.asarray(chunk_data.size_factors), np.zeros((pad, 1), dtype=np.float32)])
        ),
        covariates=jnp.asarray(
            np.concatenate([np.asarray(chunk_data.covariates), np.zeros((pad, 1), dtype=np.float32)])
        ),
        covariate_names=chunk_data.covariate_names,
        library_size_center_log_mean=0.0,
    )
    padded_design = build_chunk_design(baseline, padded, control_data=control_data)

    assert padded_design.num_cells == unpadded.num_cells
    np.testing.assert_array_equal(
        np.asarray(padded_design.target_codes), np.asarray(unpadded.target_codes)
    )


def test_multi_assignment_cells_are_set_aside(capsys) -> None:
    """A cell carrying two perturbations has no place in a one-label-per-cell null.
    It is set aside and counted, not reinterpreted and not fatal: a low-MOI screen at
    a realised MOI a little above one arrives with such cells in every chunk."""
    control_data, chunk_data, theta, _ = _simulate_screen()
    baseline = prepare_crt_baseline(control_data, _fitted_control_fit(control_data, theta))
    matrix = np.zeros((240, 6), dtype=np.int8)
    matrix[np.arange(240), np.repeat(np.arange(6), 40)] = 1
    matrix[0, 1] = 1  # a second assignment on one cell
    doubled = PerTurboData(
        counts=chunk_data.counts,
        pert_id=jnp.asarray(matrix),
        pert_names=chunk_data.pert_names,
        gene_names=chunk_data.gene_names,
        size_factors=chunk_data.size_factors,
        covariates=chunk_data.covariates,
        covariate_names=chunk_data.covariate_names,
        library_size_center_log_mean=0.0,
    )
    design = build_chunk_design(baseline, doubled, control_data=control_data)
    assert design is not None
    assert "setting aside 1 of 240 chunk cells" in capsys.readouterr().out
    result = run_crt_for_chunk(
        baseline, doubled, control_data=control_data, num_resamples=8, seed=0,
        tail_families=("saddlepoint",), resampling_mechanism="propensity", saddlepoint_only=True,
    )
    assert result.num_multi_assignment_cells_dropped == 1


def test_the_production_path_matches_the_research_path() -> None:
    """Parity: perturbation-chunked production vs. gene-chunked research.

    Both are handed the same cells, the same offsets, the same dispersion and
    the same baseline coefficients, so any disagreement is the port's own doing
    rather than a modelling difference. p-values must match exactly - both draw
    identical resamples, and the statistic is far from a tie at this size.
    """

    control_data, chunk_data, theta, strata = _simulate_screen()
    baseline = prepare_crt_baseline(control_data, _fitted_control_fit(control_data, theta))
    design = _research_design(control_data, chunk_data, theta)

    research = run_low_moi_score_permutations(
        design,
        num_resamples=199,
        strata=strata,
        seed=17,
        backend="jax",
        null_model="control_only",
        nuisance_coefficients=baseline.nuisance.coefficients,
    )
    production = run_crt_for_chunk(
        baseline,
        chunk_data,
        control_data=control_data,
        num_resamples=199,
        strata=strata,
        seed=17,
    )

    assert production.target_names == design.target_names
    np.testing.assert_allclose(
        production.observed_score,
        np.asarray(research.observed_score, dtype=np.float64),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(production.p_value, np.asarray(research.p_value, dtype=np.float64))


def test_the_gene_chunk_size_does_not_change_the_answer() -> None:
    """Genes are independent, so slicing them is a decomposition, not an approximation."""

    control_data, chunk_data, theta, strata = _simulate_screen()
    baseline = prepare_crt_baseline(control_data, _fitted_control_fit(control_data, theta))
    shared = dict(control_data=control_data, num_resamples=99, strata=strata, seed=5)

    whole = run_crt_for_chunk(baseline, chunk_data, **shared)
    split = run_crt_for_chunk(baseline, chunk_data, gene_chunk_size=3, **shared)

    np.testing.assert_allclose(split.observed_score, whole.observed_score, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(split.p_value, whole.p_value)


def test_a_targets_result_does_not_depend_on_which_chunk_it_landed_in() -> None:
    """The property the name-keyed seeding exists to protect, end to end.

    Splitting the perturbations across chunks changes each target's neighbours
    and its position, and must change nothing about its p-values.
    """

    control_data, chunk_data, theta, strata = _simulate_screen()
    baseline = prepare_crt_baseline(control_data, _fitted_control_fit(control_data, theta))
    num_control = int(np.asarray(control_data.counts).shape[0])

    together = run_crt_for_chunk(
        baseline, chunk_data, control_data=control_data, num_resamples=99, strata=strata, seed=8
    )

    codes = np.asarray(chunk_data.pert_id)
    keep = codes >= 3
    split_chunk = PerTurboData(
        counts=jnp.asarray(np.asarray(chunk_data.counts)[keep]),
        pert_id=jnp.asarray(codes[keep] - 3),
        pert_names=list(chunk_data.pert_names[3:]),
        gene_names=chunk_data.gene_names,
        size_factors=jnp.asarray(np.asarray(chunk_data.size_factors)[keep]),
        covariates=jnp.asarray(np.asarray(chunk_data.covariates)[keep]),
        covariate_names=chunk_data.covariate_names,
        library_size_center_log_mean=0.0,
    )
    split_strata = np.concatenate([strata[:num_control], strata[num_control:][keep]])
    apart = run_crt_for_chunk(
        baseline,
        split_chunk,
        control_data=control_data,
        num_resamples=99,
        strata=split_strata,
        seed=8,
    )

    assert apart.target_names == ("target_3", "target_4", "target_5")
    np.testing.assert_allclose(apart.observed_score, together.observed_score[3:], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(apart.p_value, together.p_value[3:])


def test_the_crt_separates_a_perturbed_gene_from_the_null_ones() -> None:
    control_data, chunk_data, theta, strata = _simulate_screen()
    baseline = prepare_crt_baseline(control_data, _fitted_control_fit(control_data, theta))
    result = run_crt_for_chunk(
        baseline, chunk_data, control_data=control_data, num_resamples=499, strata=strata, seed=2
    )

    # target_0 knocks gene_2 down and target_3 pushes gene_5 up; nothing else moves.
    assert result.p_value[0, 2] < 0.01
    assert result.p_value[3, 5] < 0.01
    untouched = np.delete(result.p_value[1], [2, 5])
    assert np.nanmin(untouched) > 0.01


def test_strata_must_cover_the_control_pool() -> None:
    control_data, chunk_data, theta, _ = _simulate_screen()
    baseline = prepare_crt_baseline(control_data, _fitted_control_fit(control_data, theta))
    with pytest.raises(ValueError, match="must cover the control cells"):
        run_crt_for_chunk(
            baseline,
            chunk_data,
            control_data=control_data,
            num_resamples=9,
            strata=np.zeros(240, dtype=np.int64),
        )


def test_every_tail_family_is_fitted_from_one_resampling_pass() -> None:
    """All three families come back, and all resolve below the empirical floor.

    That floor, ``1/(B+1)``, is the whole reason the parametric tails are here:
    over a large screen it can sit above the Benjamini-Hochberg cutoff, so no
    empirical p-value can reach a usable FDR threshold however many pairs are
    truly non-null.
    """

    control_data, chunk_data, theta, strata = _simulate_screen()
    baseline = prepare_crt_baseline(control_data, _fitted_control_fit(control_data, theta))
    result = run_crt_for_chunk(
        baseline, chunk_data, control_data=control_data, num_resamples=99, strata=strata, seed=6
    )

    assert set(result.parametric) == set(CRT_TAIL_FAMILIES)
    # Empirical p-values are accumulated in float32, so compare against the
    # floor at that precision rather than at float64's.
    floor = 1.0 / (99 + 1)
    assert np.nanmin(result.p_value) >= floor * (1.0 - 1e-6)
    for family, columns in result.parametric.items():
        finite = np.isfinite(columns["p_value"])
        assert finite.any(), family
        assert np.nanmin(columns["p_value"][finite]) < floor, family
        # The log scale has no floor at all, so it is what ranking should use.
        assert np.all(columns["log_p_value"][finite] <= 0.0)

    # Moment diagnostics travel with the fit, since the extrapolation is only
    # as good as the moments behind it.
    assert set(result.null_summaries) == {
        "crt_null_mean",
        "crt_null_variance",
        "crt_null_skewness",
        "crt_null_excess_kurtosis",
    }
    assert np.isfinite(result.null_summaries["crt_null_variance"]).any()


def test_tail_families_can_be_switched_off() -> None:
    control_data, chunk_data, theta, strata = _simulate_screen()
    baseline = prepare_crt_baseline(control_data, _fitted_control_fit(control_data, theta))
    result = run_crt_for_chunk(
        baseline,
        chunk_data,
        control_data=control_data,
        num_resamples=99,
        strata=strata,
        seed=6,
        tail_families=(),
    )
    assert result.parametric == {}


def test_cached_low_moi_selection_plan_matches_uncached_and_validates_assignments(monkeypatch) -> None:
    control_data, chunk_data, theta, strata = _simulate_screen()
    string_strata = strata.astype(str)
    control_data = dataclasses.replace(control_data, _analysis_design_token=object())
    chunk_data = dataclasses.replace(chunk_data, _analysis_design_token=object())
    baseline = prepare_crt_baseline(control_data, _fitted_control_fit(control_data, theta))
    precompute_calls = 0
    original_precompute = crt_module.precompute_low_moi_permutations

    def counting_precompute(*args, **kwargs):
        nonlocal precompute_calls
        precompute_calls += 1
        return original_precompute(*args, **kwargs)

    monkeypatch.setattr(crt_module, "precompute_low_moi_permutations", counting_precompute)
    uncached, plan = run_crt_for_chunk(
        baseline,
        chunk_data,
        control_data=dataclasses.replace(control_data),
        num_resamples=31,
        strata=string_strata,
        seed=12,
        tail_families=(),
        _return_permutations=True,
    )
    cached = run_crt_for_chunk(
        baseline,
        chunk_data,
        control_data=control_data,
        num_resamples=31,
        strata=string_strata,
        seed=12,
        tail_families=(),
        saddlepoint_screen_p_value=0.2,
        _permutations=plan,
    )

    assert precompute_calls == 1
    assert plan._validation_nuisance_design is None
    np.testing.assert_array_equal(cached.observed_score, uncached.observed_score)
    np.testing.assert_array_equal(cached.p_value, uncached.p_value)

    assignments = np.asarray(chunk_data.pert_id).copy()
    assignments[[0, 40]] = assignments[[40, 0]]
    changed = dataclasses.replace(chunk_data, pert_id=jnp.asarray(assignments))
    with pytest.raises(ValueError, match="target assignments"):
        run_crt_for_chunk(
            baseline,
            changed,
            control_data=control_data,
            num_resamples=31,
            strata=string_strata,
            seed=12,
            tail_families=(),
            _permutations=plan,
        )


def test_the_saddlepoint_only_production_path_matches_the_research_path() -> None:
    """Parity for the propensity saddlepoint with no draws at all.

    Same cells, offsets, dispersion and baseline coefficients on both sides, so
    the only thing the production wrapper adds is the perturbation-chunk
    stacking. The saddlepoint is deterministic given the fit, so the log
    p-values must agree to rounding.
    """

    control_data, chunk_data, theta, _ = _simulate_screen()
    baseline = prepare_crt_baseline(control_data, _fitted_control_fit(control_data, theta))
    design = _research_design(control_data, chunk_data, theta)

    drawn = precompute_low_moi_permutations(
        design, num_resamples=9, seed=3, resampling_mechanism="propensity", draw_resamples=False
    )
    research = run_low_moi_score_permutations(
        design,
        num_resamples=9,
        seed=3,
        backend="jax",
        null_model="control_only",
        permutations=drawn,
        nuisance_coefficients=baseline.nuisance.coefficients,
        tail_approximation="saddlepoint",
        saddlepoint_only=True,
        saddlepoint_screen_p_value=0.5,
    )
    production = run_crt_for_chunk(
        baseline,
        chunk_data,
        control_data=control_data,
        num_resamples=9,
        seed=3,
        tail_families=("saddlepoint",),
        resampling_mechanism="propensity",
        saddlepoint_only=True,
        saddlepoint_screen_p_value=0.5,
    )

    assert production.saddlepoint_only and production.resampling_mechanism == "propensity"
    assert np.isnan(production.p_value).all()
    fitted = production.parametric["saddlepoint"]
    assert fitted["valid"].all()
    np.testing.assert_allclose(
        fitted["log_p_value"],
        np.asarray(research.tail_fits["saddlepoint"]["log_p_value"]),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_array_equal(fitted["used_screen"], np.asarray(research.parametric_used_fallback))
    # The planted effects are found without a single resample.
    assert fitted["p_value"][0, 2] < 1e-4 and fitted["p_value"][3, 5] < 1e-4


def _displaced_control_fit(control_data: PerTurboData, theta: np.ndarray, shift: float = 0.3) -> ControlFit:
    """A stage-one fit that stands off the null mode, as an SVI median does."""

    fitted = _fitted_control_fit(control_data, theta)
    return ControlFit(
        beta_0=fitted.beta_0 + shift,
        theta=fitted.theta,
        noise_scale=fitted.noise_scale,
        factor_loadings=None,
        factor_scores=None,
        factor_center=None,
        pca_loadings=None,
        size_factors=fitted.size_factors,
        losses=fitted.losses,
        svi_result=None,
        covariate_coef=fitted.covariate_coef * 0.5,
    )


def test_polishing_moves_a_displaced_baseline_onto_the_null_mode() -> None:
    control_data, _, theta, _ = _simulate_screen()
    displaced = _displaced_control_fit(control_data, theta)
    with pytest.raises(ValueError, match="not at the control-cell null mode"):
        prepare_crt_baseline(control_data, displaced)

    baseline = prepare_crt_baseline(control_data, displaced, polish=True)
    assert baseline.pre_polish_check is not None and not baseline.pre_polish_check.ok
    assert baseline.null_check.ok
    assert float(np.nanmax(baseline.null_check.newton_step)) < 1e-3
    # Dispersion, offsets and design are stage one's; only the coefficients moved.
    np.testing.assert_allclose(baseline.nuisance.dispersion, theta, rtol=1e-6)


def test_the_polished_production_path_matches_the_research_fit() -> None:
    """Polished production against the research driver fitting its own null.

    The research path fits the control-only NB null itself; polishing brings a
    displaced stage-one baseline onto that same mode, so the two must agree on
    the statistic and, sharing draws, on the p-values up to float32 rounding.
    """

    control_data, chunk_data, theta, strata = _simulate_screen()
    baseline = prepare_crt_baseline(control_data, _displaced_control_fit(control_data, theta), polish=True)
    design = _research_design(control_data, chunk_data, theta)

    research = run_low_moi_score_permutations(
        design,
        num_resamples=199,
        strata=strata,
        seed=17,
        backend="jax",
        null_model="control_only",
    )
    production = run_crt_for_chunk(
        baseline,
        chunk_data,
        control_data=control_data,
        num_resamples=199,
        strata=strata,
        seed=17,
    )
    np.testing.assert_allclose(
        production.observed_score,
        np.asarray(research.observed_score, dtype=np.float64),
        rtol=1e-4,
        atol=1e-4,
    )
    assert np.abs(production.p_value - np.asarray(research.p_value, dtype=np.float64)).max() <= 2.0 / 200


def _screen_with_a_selection_covariate(seed: int = 3):
    """The same screen, but with a covariate that genuinely predicts selection.

    Targets 0-2 sit high on the covariate and targets 3-5 low, so which targets
    a chunk holds decides what a selection model fitted inside that chunk sees.
    Without a shift like this the estimator's chunk dependence is invisible.
    """

    control_data, chunk_data, theta, strata = _simulate_screen(seed=seed)
    codes = np.asarray(chunk_data.pert_id)
    covariates = np.asarray(chunk_data.covariates).copy()
    covariates[codes < 3, 0] += 1.5
    covariates[codes >= 3, 0] -= 1.5
    chunk_data = dataclasses.replace(chunk_data, covariates=jnp.asarray(covariates))
    return control_data, chunk_data, theta, strata


def _split_off_last_three_targets(chunk_data: PerTurboData) -> PerTurboData:
    """Targets 3-5 alone, laid out as the production chunk loader hands them over."""

    codes = np.asarray(chunk_data.pert_id)
    keep = codes >= 3
    return PerTurboData(
        counts=jnp.asarray(np.asarray(chunk_data.counts)[keep]),
        pert_id=jnp.asarray(codes[keep] - 3),
        pert_names=list(chunk_data.pert_names[3:]),
        gene_names=chunk_data.gene_names,
        size_factors=jnp.asarray(np.asarray(chunk_data.size_factors)[keep]),
        covariates=jnp.asarray(np.asarray(chunk_data.covariates)[keep]),
        covariate_names=chunk_data.covariate_names,
        library_size_center_log_mean=0.0,
    )


def _selection_model_over(control_data: PerTurboData, chunk_data: PerTurboData) -> np.ndarray:
    """Fit the selection model over exactly these cells, as the CLI does per screen."""

    control_covariates = np.asarray(control_data.covariates)
    chunk_covariates = np.asarray(chunk_data.covariates)
    num_control = control_covariates.shape[0]
    num_target = chunk_covariates.shape[0]
    # [intercept, covariates], controls first: the layout build_chunk_design
    # gives the CRT, which is the parametrisation the coefficients travel in.
    nuisance = np.concatenate(
        [
            np.ones((num_control + num_target, 1), dtype=np.float32),
            np.concatenate([control_covariates, chunk_covariates], axis=0),
        ],
        axis=1,
    )
    targeting = np.concatenate([np.zeros(num_control), np.ones(num_target)])
    return fit_shared_propensity_coefficients(nuisance, targeting)


def _saddlepoint_log_p(baseline, chunk, control_data, shared) -> np.ndarray:
    result = run_crt_for_chunk(
        baseline,
        chunk,
        control_data=control_data,
        num_resamples=9,
        seed=8,
        tail_families=("saddlepoint",),
        resampling_mechanism="propensity",
        saddlepoint_only=True,
        saddlepoint_screen_p_value=0.5,
        shared_propensity_coefficients=shared,
    )
    return result.parametric["saddlepoint"]["log_p_value"]


def test_screen_wide_slopes_survive_a_change_of_chunk_size() -> None:
    """A target's p-value must not move when the chunk-size flag regroups it.

    Slopes fitted once over the whole screen are the same numbers in every
    chunk, so targets 3-5 get the same saddlepoint tail whether they are tested
    beside targets 0-2 or on their own.
    """

    control_data, chunk_data, theta, _ = _screen_with_a_selection_covariate()
    baseline = prepare_crt_baseline(control_data, _fitted_control_fit(control_data, theta))
    shared = _selection_model_over(control_data, chunk_data)

    together = _saddlepoint_log_p(baseline, chunk_data, control_data, shared)
    apart = _saddlepoint_log_p(
        baseline, _split_off_last_three_targets(chunk_data), control_data, shared
    )

    np.testing.assert_allclose(apart, together[3:], rtol=1e-6, atol=1e-8)


def test_slopes_fitted_inside_the_chunk_move_with_the_chunk_size() -> None:
    """The same run under the estimator the screen-wide fit replaces.

    Fitting the shared slopes on the cells the chunk happens to hold is the old
    in-chunk behaviour. Here it makes targets 3-5 answer differently depending
    on whether targets 0-2 shared their chunk, which is the bug.
    """

    control_data, chunk_data, theta, _ = _screen_with_a_selection_covariate()
    baseline = prepare_crt_baseline(control_data, _fitted_control_fit(control_data, theta))
    split = _split_off_last_three_targets(chunk_data)

    together = _saddlepoint_log_p(
        baseline, chunk_data, control_data, _selection_model_over(control_data, chunk_data)
    )
    apart = _saddlepoint_log_p(
        baseline, split, control_data, _selection_model_over(control_data, split)
    )

    assert np.max(np.abs(apart - together[3:])) > 1e-2


def test_screen_wide_slopes_are_the_in_design_fit_when_nothing_is_chunked() -> None:
    """Unchunked, "over the whole screen" and "inside the chunk" are the same cells.

    The CLI passes the screen-wide coefficients on the unchunked path too, so
    that one piece of code serves both, and this is the claim that makes that
    free. It also checks the assembly the CLI does by hand - controls first,
    ``[intercept, covariates]``, that column order - against the design the CRT
    actually builds, which is the thing the coefficients are interpreted in.
    """

    control_data, chunk_data, theta, _ = _screen_with_a_selection_covariate()
    baseline = prepare_crt_baseline(control_data, _fitted_control_fit(control_data, theta))
    assembled_by_the_caller = _selection_model_over(control_data, chunk_data)

    design = build_chunk_design(baseline, chunk_data, control_data=control_data)
    nuisance = np.asarray(design.nuisance_design, dtype=np.float64)
    fitted_in_the_design = fit_shared_propensity_coefficients(
        nuisance, (np.asarray(design.target_codes) >= 0).astype(np.float32)
    )

    # Compared as predictors, not coefficients: the design carries no unique
    # coefficient vector when a batch one-hot sits beside the intercept, and
    # only the predictor is identified.
    np.testing.assert_allclose(
        nuisance @ assembled_by_the_caller, nuisance @ fitted_in_the_design, rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        _saddlepoint_log_p(baseline, chunk_data, control_data, assembled_by_the_caller),
        _saddlepoint_log_p(baseline, chunk_data, control_data, fitted_in_the_design),
        rtol=1e-6,
        atol=1e-8,
    )


def test_shared_coefficients_must_match_the_nuisance_columns() -> None:
    control_data, chunk_data, theta, _ = _screen_with_a_selection_covariate()
    baseline = prepare_crt_baseline(control_data, _fitted_control_fit(control_data, theta))
    with pytest.raises(ValueError, match="one coefficient per nuisance column"):
        _saddlepoint_log_p(baseline, chunk_data, control_data, np.zeros(7))
