# PerTurbo: from perturbations to gene-expression effects

You have measured RNA in cells carrying CRISPR perturbations. Which genes
responded to each perturbation? How large were those changes, and which
associations are convincing enough to follow up?

PerTurbo analyzes Perturb-seq count data with two complementary tools: a
Bayesian model that estimates expression effects and uncertainty, and a
conditional randomization test (CRT) that supplies frequentist p-values for
screen-wide testing. Both use JAX for CPU or NVIDIA GPU computation. Inputs
use the AnnData and MuData formats familiar from single-cell analysis in Python.

## When to choose PerTurbo

PerTurbo is designed for screens where you want an **element-by-gene table**:
for example, how targeting an enhancer changes a nearby gene, or how targeting
a transcription factor changes a broader expression program. An element can
be a targeted gene, enhancer, or another experimental target, with one or more
guides assigned to it.

- In a **low-MOI screen**, most cells carry one perturbation. PerTurbo can
  compare each target with negative-control cells.
- In a **high-MOI screen**, cells carry several perturbations. The effect model
  retains co-occurring element predictors; gene blocks let you fit parts of the
  transcriptome while keeping those predictors together.
- Effect estimates and statistical evidence are available together, so you
  can prioritize changes that are both biologically substantial and supported
  by the test, rather than ranking solely by a p-value.

The two outputs answer different questions, especially in high-MOI data:
the joint effect model adjusts for other fitted elements, whereas the
all-cells CRT tests marginal element–gene associations given its covariates.
Read [How the analysis works](method.md) before treating those quantities as
interchangeable. Calibration depends on the experimental design and model
assumptions; the software does not make every dataset automatically calibrated.

PerTurbo starts after read processing, guide calling, and basic cell quality
control. It is not an alignment tool, a cell-type annotation workflow, or an
explicit interaction/epistasis model. Assess its suitability for a particular
screen with negative controls, fit diagnostics, and relevant comparisons.

## Start here

| What you want to do | Where to go |
| --- | --- |
| Install PerTurbo and check whether it sees your GPU | [Installation](installation.md) |
| Complete a small analysis without downloading a large dataset | [Your first analysis](quickstart.md) |
| Convert your existing AnnData or guide assignments | [Prepare your data](data_preparation.md) |
| Choose the important command-line settings | [Running analyses](running_analyses.md) |
| Understand shrinkage, the CRT, and model assumptions | [How the analysis works](method.md) |
| Read the result table and select follow-up candidates | [Interpreting results](results_guide.md) |
| Work with fitted models in a notebook | [Using PerTurbo from Python](python_api.md) |
| Diagnose a failed, slow, or poorly calibrated run | [Troubleshooting](troubleshooting.md) |

These pages describe the v2 release candidate. The first tutorial uses small
synthetic data to teach the workflow; it is not a biological benchmark. Use the
[installation instructions](installation.md) for the currently available
distribution, and record the version and settings used for a real analysis.

```{toctree}
:hidden:
:caption: Getting started
:maxdepth: 1

installation.md
quickstart.md
data_preparation.md
```

```{toctree}
:hidden:
:caption: User guide
:maxdepth: 1

running_analyses.md
method.md
results_guide.md
python_api.md
troubleshooting.md
crt_quickstart.md
pipeline_integration.md
```

```{toctree}
:hidden:
:caption: Reference and development
:maxdepth: 1

api.md
changelog.md
contributing.md
references.md
scverse_readiness.md
high_moi_padding_handoff.md
```
