# What PerTurbo estimates and tests

PerTurbo analyzes a Perturb-seq count matrix together with the perturbations
observed in each cell. Its main unit of analysis is an **element–gene pair**:
an element is the biological target of a perturbation, and a gene is a measured
expression feature. For each pair, PerTurbo can estimate the direction and size
of the expression change and run a conditional randomization test (CRT) of the
association.

These answer related but different questions:

- The effect fit asks, “How large is the fitted expression difference, in which
  direction, after accounting for the supplied cell-level adjustment terms?”
- The CRT asks, “Would an association this strong be unusual if perturbation
  assignment were redrawn according to the fitted assignment mechanism, while
  holding the observed expression and covariates fixed?”

The estimate is useful for biological ranking and magnitude. The CRT supplies a
frequentist p-value that can be corrected across a declared family of tested
pairs. Agreement between a sizable estimate and a small CRT q-value is stronger
evidence than either number alone. Neither output by itself proves causality;
unmeasured technical or biological differences between assigned and unassigned
cells can still affect interpretation.

## Vocabulary

**Cell**
: One profiled cell and one row of the expression and perturbation matrices.

**Gene**
: One measured expression feature. PerTurbo models raw gene counts, not a
  log-normalized expression matrix.

**Guide**
: An observed guide barcode. Several guides can be designed against the same
  biological target.

**Element**
: The target to which one or more guides are mapped, such as a promoter, enhancer,
  or gene. The guide-to-element map defines this grouping. In the shared-effect
  model, two guides for the same element in one cell count as one active element,
  rather than adding the same element effect twice.

**MOI**
: Multiplicity of infection, used here operationally as the number of detected
  guides per cell. Low-MOI designs usually have zero or one targeting element per
  cell. High-MOI designs contain co-occurring elements, so each element's fitted
  coefficient is adjusted for the other represented elements.

Control guides and control elements must be identified correctly. They set the
stage-one baseline and provide the null element rows used for one of the output
diagnostics. A guide-to-element mapping is also required for guide-aware models
and the high-MOI all-cells CRT.

## Count model and effect scale

The standard model is a negative binomial (NB) count model. For cell $i$ and
gene $g$, its log mean contains a gene baseline, the cell's size-factor offset,
optional measured covariates, and the sum of active element effects:

$$
\log \mu_{ig} = \beta_{0g} + s_i + x_i^T\gamma_g + a_i^T\beta_g.
$$

The NB dispersion allows count variance to exceed the mean. Stage one estimates
the baseline and nuisance quantities from control cells. Stage two estimates the
element effects while reusing that baseline. With co-occurring perturbations,
all active element columns are retained in the joint stage-two model.

`posterior_mean` is on the natural-log expression scale. Holding the other model
terms fixed, exponentiating it gives a fitted count ratio:

- `posterior_mean = 0` means a ratio of `exp(0) = 1`.
- `posterior_mean = -0.69` means approximately half as many expected counts.
- `posterior_mean = 0.69` means approximately twice as many expected counts.

The reference is a cell with that element inactive, conditional on the other
included perturbations, size factor, and covariates. It is not necessarily an
untreated bulk sample. For a high-MOI cell, the coefficient is the additive log
effect of one element within the fitted multi-element model; PerTurbo does not
fit pairwise epistasis terms in this model.

## Shrinkage and approximate uncertainty

Element coefficients receive the configured prior (`normal` by default, with a
`cauchy` option). The prior shrinks weakly informed estimates toward zero. This
stabilizes a large element-by-gene fit, but it can also pull rare-element or
poorly converged estimates toward zero. `posterior_scale` summarizes uncertainty
from the fitted variational approximation; it is not a standard error from an
unpenalized maximum-likelihood fit.

PerTurbo fits the posterior with stochastic variational inference (SVI) and an
`AutoNormal` guide. This is an approximation to the posterior. A completed run
does not itself demonstrate convergence. Inspect the loss trajectory, compare
estimates after a larger training budget, and pay special attention to large
effects and sparse elements. The step size and number of steps work together;
changing either changes how far optimization can move. Cell minibatching also
changes the optimization regime and is not recommended as a first response to a
slow or memory-limited fit.

## Conditional randomization tests

The CRT uses an NB score under a null model and varies the perturbation labels
while holding the observed expression fixed. The central CRT assumption concerns
the **assignment distribution given covariates**. With that distribution known
and the randomization distribution evaluated exactly, CRT validity need not
require a correctly specified expression model; the expression model supplies
a useful statistic. This robustness motivates the conditional-randomization
approach introduced for these screens by
[SCEPTRE](https://pmc.ncbi.nlm.nih.gov/articles/PMC8686614/).

PerTurbo uses estimated nuisance quantities and, in its default production path,
a saddlepoint tail approximation. The general CRT principle does not make these
estimated p-values exact or automatically calibrated on a new screen. Missing
assignment predictors, guide-detection artifacts, inappropriate controls, and
numerical or approximation failures still require attention. Evaluate negative
controls at the small p-values relevant to your testing family, and inspect
diagnostics across genes with different detection rates. A distant element–gene
pair is not necessarily a negative control: perturbations can have real trans
effects.

### Control-anchored pool

`control-anchored` is intended for a design with at most one targeting element
per analyzed cell and a usable pool of control cells. For each element, the test
pool consists of the control cells and cells carrying that element. The null
baseline is learned from controls. If a mapped low-MOI input contains cells with
more than one targeting element, this path sets those cells aside and reports
their count.

This test asks whether that element's cells differ from the control pool beyond
what the selected assignment mechanism and covariates predict. It does not
estimate a high-MOI marginal association.

### All-cells pool

`all-cells` is intended for high-MOI designs. It uses every analyzed cell and
tests each element as a **marginal association** in the all-cells population. The
null outcome model omits all element effects. Each element has a logistic
propensity model based on the nuisance design; by default the propensity also
includes standardized `log1p` detected-guide count because cells with more
detected guides are more likely to contain any particular element.

This estimand is not a pairwise interaction or epistasis effect. Co-occurring
perturbations can be associated with the tested element through the assignment
process, so the propensity covariates must capture the assignment structure
needed for the conditional null. Review `crt_metadata.json` to confirm which pool
was selected and what the software measured.

### Empirical and tail p-values

With resampling enabled, `crt_p_value` is the empirical randomization p-value. It
cannot be smaller than `1 / (number_of_resamples + 1)`. PerTurbo can also report
tail approximations named `skew_normal`, `student_t`, and `saddlepoint`. These are
alternative approximations for the same score statistic and each receives its
own multiple-testing correction. Their disagreement is a diagnostic, especially
in the far tail.

In propensity `saddlepoint-only` mode, no resamples are drawn:
`crt_p_value` and `crt_q_value` are missing, and the primary columns are
`crt_saddlepoint_p_value` and `crt_saddlepoint_q_value`. Check the corresponding
`crt_saddlepoint_valid` field before interpreting a row.
