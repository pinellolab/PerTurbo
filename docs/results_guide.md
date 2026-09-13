# Reading PerTurbo results

The main CLI result is `element_effects.parquet`, with one row per analyzed
element–gene pair. Read it with pandas:

```python
import numpy as np
import pandas as pd

effects = pd.read_parquet("perturbo_outputs/run/element_effects.parquet")
effects["count_ratio"] = np.exp(effects["posterior_mean"])
```

Start by checking `crt_metadata.json`, the run log, and the loss trajectory. They
record which CRT pool ran, how the design was classified, whether cells were set
aside, and whether optimization warrants a longer-run comparison.

## Element table fields

| Field | Meaning |
| --- | --- |
| `method` | Producer name; the CLI writes `perturbo`. |
| `element` | Biological target from the element mapping. |
| `gene` | Measured expression feature. |
| `posterior_mean` | Variational posterior location for the element's log count effect. `exp(posterior_mean)` is the fitted count ratio in the model's reference frame. |
| `posterior_scale` | Scale of the variational posterior approximation. Larger values mean less precise fitted effects. |
| `z_value` | `posterior_mean / posterior_scale`. |
| `posterior_prob` | Despite its historical name, the current table computes a two-sided standard-normal tail area from `abs(z_value)`. It is not the posterior probability that an effect exists, is positive, or exceeds a biological threshold. |
| `empirical_p_value` | A two-sided diagnostic obtained by fitting a zero-centered Student-t null to `z_value` values from control-matched element rows. It is missing when no such rows exist. It is distinct from the CRT. |

The standard table does not apply Benjamini–Hochberg correction to
`posterior_prob` or `empirical_p_value`. Use the CRT q-value columns for the
implemented frequentist screen-wide correction.

When CRT output is present, the table adds:

| Field pattern | Meaning |
| --- | --- |
| `crt_z_value` | Observed CRT score standardized by its null spread. It is a test statistic, not the fitted log effect. |
| `crt_p_value` | Empirical randomization p-value. Missing in saddlepoint-only runs. |
| `crt_q_value` | Benjamini–Hochberg adjustment of finite `crt_p_value` values across the full tested element–gene family. |
| `crt_<family>_p_value` | Tail p-value for `<family>`, where configured families can be `skew_normal`, `student_t`, or `saddlepoint`. |
| `crt_<family>_log_p_value` | Natural logarithm of that tail p-value, retained for very small probabilities. |
| `crt_<family>_q_value` | Benjamini–Hochberg adjustment for that family's p-values across the full tested family. Families are corrected separately. |
| `crt_<family>_valid` | Whether that tail calculation was valid for the row. Check this before filtering on its p- or q-value. |
| `crt_saddlepoint_used_screen` | Whether the saddlepoint path used its screening approximation for that row. Present when applicable. |
| `crt_null_mean`, `crt_null_variance`, `crt_null_skewness`, `crt_null_excess_kurtosis` | Diagnostics describing the randomization null for the score. Some fields can be missing for modes that do not compute them. |

`NaN` means the quantity was unavailable or the pair was not validly tested. It
does not mean zero effect or non-significance.

## A practical filtering recipe

Choose the CRT family before looking at discoveries, require a valid finite test,
then combine q-value, direction, and a biologically meaningful magnitude. For a
propensity saddlepoint-only run:

```python
tested = effects.loc[
    effects["crt_saddlepoint_valid"].astype(bool)
    & effects["crt_saddlepoint_q_value"].notna()
].copy()

hits = tested.loc[
    (tested["crt_saddlepoint_q_value"] < 0.05)
    & (tested["posterior_mean"].abs() >= np.log(1.25))
].sort_values(["crt_saddlepoint_q_value", "element", "gene"])

hits[
    [
        "element",
        "gene",
        "posterior_mean",
        "count_ratio",
        "posterior_scale",
        "crt_saddlepoint_p_value",
        "crt_saddlepoint_q_value",
    ]
].head(20)
```

The `1.25` ratio is only an example; choose a threshold suited to the assay and
scientific question. Do not filter `posterior_prob` as though it were a posterior
inclusion probability. Do not use `crt_z_value` as an effect size.

For a resampling run, use `crt_q_value` with finite `crt_p_value`. Remember that
the empirical p-value has a resolution floor. If many rows lie at that floor,
their ordering and adjusted values contain limited information; inspect the
configured tail-family columns and their validity diagnostics rather than
interpreting the floor as identical biological evidence.

## Requested pairs and multiple-testing families

If the run used `--pairs-to-test`, PerTurbo still fits and tests the full grid. It
also writes `element_effects_requested_pairs.parquet`. Effect estimates and
p-values for a pair are the same in both files. Every column named `q_value` or
ending in `_q_value` is recomputed within the requested set when its matching
p-value column exists.

This makes the two files answers to two declared families:

- `element_effects.parquet`: correction over all tested element–gene pairs.
- `element_effects_requested_pairs.parquet`: correction over the supplied,
  deduplicated `element,gene` pairs that were present in the analyzed grid.

Choose the family that matches the question specified before examining results.
A smaller requested-family q-value does not mean that the underlying estimate or
p-value improved.

```python
all_pairs = pd.read_parquet("perturbo_outputs/run/element_effects.parquet")
requested = pd.read_parquet(
    "perturbo_outputs/run/element_effects_requested_pairs.parquet"
)

key = ["element", "gene"]
comparison = requested.merge(
    all_pairs[key + ["crt_saddlepoint_p_value", "crt_saddlepoint_q_value"]],
    on=key,
    suffixes=("_requested", "_screen"),
    validate="one_to_one",
)

# The p-values agree; the q-values may differ because the families differ.
assert np.allclose(
    comparison["crt_saddlepoint_p_value_requested"],
    comparison["crt_saddlepoint_p_value_screen"],
    equal_nan=True,
)
```

## Guide-level outputs

The primary biological table is element-level. With the `relative` guide-effect
strategy, the CLI can also write `guide_efficiency.parquet`, which summarizes
guide efficiencies relative to their parent element. Trained-model bundles may
contain `guide_effects.parquet` with `guide`, `element`, `gene`, `loc`, `scale`,
`z_value`, and a historically named `q_value`; in that table, `q_value` is the
two-sided standard-normal tail area computed from `abs(z_value)`, not a
Benjamini–Hochberg-adjusted CRT q-value. Keep guide diagnostics separate from the
screen-wide element CRT when reporting discoveries.

## What to report

For each reported result, include the element and gene, `posterior_mean` and its
exponentiated ratio, `posterior_scale`, the chosen CRT p- and q-value columns, the
CRT pool and assignment mechanism from `crt_metadata.json`, the multiple-testing
family, and the relevant validity flag. Also report the number of cells and
elements analyzed, control definition, covariates, guide-to-element mapping,
likelihood, training budget, and whether a longer-run comparison supported SVI
stability.
