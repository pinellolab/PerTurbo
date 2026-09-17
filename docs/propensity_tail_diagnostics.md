# Propensity CRT tail diagnostics

The equal-tail Bernoulli-propensity CRT validates each evaluated interior
saddlepoint approximation before reporting its probability. This policy was
ported from the frozen Xaira `full11194_newton_finite_bound_v4` research wrapper.
It does not change the fitted assignment law, efficient score, or screening
threshold. It can change p-values and discoveries for failed approximations.

The root residual is `abs(K'(t) - observed) / null_standard_deviation` and must
be at most `1e-6`. The raw Lugannani–Rice correction must be finite and define a
probability; linear underflow alone is not a failure when the log probability
is valid. A failing approximation uses `min(1, 2 * exp(K(t) - t * observed))`
when the bound guards pass. A finite, nonnegative, nonoptimal tilt still gives
a Chernoff bound: root-residual-only failures may therefore use that bound.
Multiple failures do not get this relaxation. If no guarded bound is usable,
the result is p=1. The checks are applied to the selected observed-side tail.
Exact support-boundary cases are handled directly.

The symmetric convention retains its previous implementation. These diagnostics
refer to the equal-tail policy, not the stratified or permutation approximations.

| Column | Meaning |
| --- | --- |
| `crt_tail_failure_reason` | Bitwise combination of failures below; 0 means evaluated with no failure, -1 means not evaluated by this policy. |
| `crt_used_chernoff` | A failed approximation was replaced with a usable Chernoff bound. |
| `crt_used_conservative_one` | A failed approximation had no usable guarded bound and was replaced with p=1. |
| `crt_root_residual_null_sd` | Root residual in null standard deviations; NaN if unavailable or at a support boundary. |

Screened-out or untested pairs have reason -1, false fallback flags, and NaN
residuals. The existing `crt_saddlepoint_used_screen` keeps its meaning: it
indicates retention of the cheaper screening approximation, not a Chernoff
fallback. `crt_low_information` remains a detection-count annotation and does
not determine the tail policy or filter the multiple-testing family.

| Bit | Failure |
| --- | --- |
| 1 | Root not bracketed |
| 2 | Nonfinite root or cumulant-generating-function quantity |
| 4 | Nonpositive second derivative |
| 8 | Negative LR radicand beyond rounding tolerance |
| 16 | Root residual exceeds 1e-6 null standard deviations |
| 32 | LR correction ratio at or below -1 |
| 64 | Nonpositive LR tail in the log-domain check |
| 128 | Raw LR tail exceeds 1 |
| 256 | Wrong-sign saddlepoint |
| 512 | Nonfinite LR correction quantity |

Failure flags record why the approximation was replaced; they do not mark the
replacement p-value invalid. Existing nonfinite-observation validity checks
remain in force.
