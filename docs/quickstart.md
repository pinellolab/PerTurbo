# Your first analysis

This tutorial creates a small synthetic screen, runs the full command-line
workflow, and opens the results in Python. It needs no external dataset and
can be run on CPU. You will see both expression-effect estimates and CRT
p-values. The planted changes teach the workflow; they are not evidence about
PerTurbo's performance on real biology.

Complete [Installation](installation.md) first. Run Python blocks in a notebook
or Python session, and shell blocks in a terminal in the same working folder.

## 1. Create a small screen

We will use 1,800 cells, 20 genes, and six guides. Two guides represent negative
controls, two target element A, and two target element B. Each cell receives
one guide, making this a low-MOI example. Element A reduces the simulated
mean of gene 0; element B increases that of gene 1.

```python
import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import scipy.sparse as sp

rng = np.random.default_rng(7)
n_cells, n_genes = 1800, 20
group = np.repeat(np.arange(3), [1000, 400, 400])
guide_column = 2 * group + rng.integers(0, 2, n_cells)
barcodes = pd.Index([f"cell_{i}" for i in range(n_cells)])

baseline = np.exp(rng.uniform(0.5, 2.0, n_genes))
depth = rng.lognormal(mean=0, sigma=0.2, size=n_cells)
mean = depth[:, None] * baseline[None, :]
mean[group == 1, 0] *= 0.4
mean[group == 2, 1] *= 1.8
dispersion = 10.0
counts = rng.negative_binomial(
    dispersion, dispersion / (dispersion + mean)
).astype(np.int32)

rna = ad.AnnData(
    X=sp.csr_matrix(counts),
    obs=pd.DataFrame(index=barcodes),
    var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
)
rna.obs["library_size"] = counts.sum(axis=1)

guide_names = [
    "non-targeting_1", "non-targeting_2",
    "target_A_1", "target_A_2", "target_B_1", "target_B_2",
]
assignments = sp.csr_matrix(
    (np.ones(n_cells, dtype=np.float32), (np.arange(n_cells), guide_column)),
    shape=(n_cells, 6),
)
grna = ad.AnnData(
    X=assignments,
    obs=pd.DataFrame(index=barcodes.copy()),
    var=pd.DataFrame(index=guide_names),
)
grna.varm["element_targeted"] = sp.csr_matrix(
    (np.ones(6, dtype=np.float32), (np.arange(6), [0, 0, 1, 1, 2, 2])),
    shape=(6, 3),
)
grna.uns["element_names"] = np.array(
    ["non-targeting", "target_A", "target_B"]
)

assert rna.obs_names.equals(grna.obs_names)
mdata = md.MuData({"rna": rna, "grna": grna})
mdata.write_h5mu("quickstart.h5mu")
```

The map's rows correspond to guide names; its columns correspond to element
names. Both guide identities for A share one element effect. For your own
experiment, preserve this alignment and replace the simulated measurements
with raw counts and called guide assignments. See [Prepare your data](data_preparation.md).

## 2. Run PerTurbo

```bash
perturbo \
  --input quickstart.h5mu \
  --out-dir quickstart_results \
  --modality-key rna \
  --perturbation-modality-key grna \
  --perturbation-element-varm-key element_targeted \
  --perturbation-element-names-uns-key element_names \
  --control-substring non-targeting \
  --library-size-key library_size \
  --size-factor-mode observed \
  --likelihood negbin \
  --num-steps-control 500 \
  --num-steps-betas 500 \
  --step-size 0.01 \
  --crt \
  --crt-pool control-anchored \
  --crt-mechanism propensity \
  --crt-tail-families saddlepoint \
  --crt-saddlepoint-only \
  --device cpu
```

Most statistical settings here are already CLI defaults; spelling them out
makes the example easier to reproduce. The data flags identify the storage
layout. `--control-substring` identifies negative controls. The library-size
column contains raw totals, which PerTurbo transforms into fixed offsets.

The remaining settings request the ordinary negative-binomial model, 500
optimizer updates per stage, and a control-anchored CRT with a saddlepoint
tail calculation. That calculation avoids drawing a large number of
randomized screens. `--crt` explicitly requests testing, so incompatible
model choices produce an error instead of silently leaving you with effects
only. Each optimizer update uses all cells in its fitting pool; no cell
minibatching is requested.

For this example, expect the startup log to identify a median of one guide
per cell and 1,000 control-only cells. The first stage fits the control
expression model. The effect stage estimates element–gene changes; the CRT
supplies a separate measure of statistical evidence. Compilation can make
the initial progress update slower than later updates.

## 3. Open the result table

```python
import numpy as np
import pandas as pd

effects = pd.read_parquet("quickstart_results/element_effects.parquet")
columns = [
    "element", "gene", "posterior_mean", "posterior_scale",
    "crt_saddlepoint_p_value", "crt_saddlepoint_q_value",
]
targets = effects.loc[effects["element"].isin(["target_A", "target_B"])].copy()
targets["expression_ratio"] = np.exp(targets["posterior_mean"])
print(targets[columns + ["expression_ratio"]].head())

planted = targets.loc[
    ((targets["element"] == "target_A") & (targets["gene"] == "gene_0"))
    | ((targets["element"] == "target_B") & (targets["gene"] == "gene_1"))
]
print(planted[columns + ["expression_ratio"]])
```

`posterior_mean` is an effect on the natural-log scale. A value near `-0.69`
corresponds to a model-implied expression ratio of about `0.5`, while `0.69`
corresponds to about `2`. The ratio is conditional on the model's normalization
and covariates; it is not an absolute RNA-molecule ratio. Sampling noise,
normalization, shrinkage, and finite training mean the inferred changes need
not equal the planted multipliers exactly.

The CRT p-value describes evidence against its null model. The corresponding
q-value adjusts for testing many element–gene pairs. A small q-value is not
an effect size, and a large q-value does not establish that an effect is zero.
The historically named `posterior_prob` column is **not** a posterior
probability that a perturbation has an effect; use the explicit CRT columns
for this testing workflow. See [Interpreting results](results_guide.md).

## 4. Inspect the run before selecting candidates

Open the loss-curve images and `crt_metadata.json` in `quickstart_results`.
Check that the intended controls and CRT pool were used, that losses are
finite, and that the requested tail calculation has valid outputs. Compare
effect estimates with a longer fitting run when moving to real data.
500 steps is a starting budget, not a guarantee of convergence.

In saddlepoint-only mode, the empirical resampling p-value columns may be
missing values because no resamples were drawn; inspect the
`crt_saddlepoint_*` columns. Do not replace missing statistical results with
zero. The [Results guide](results_guide.md) explains the diagnostics and
multiple-testing families.

## Move to your own screen

Start with [Prepare your data](data_preparation.md), then use
[Running analyses](running_analyses.md) to choose controls, covariates,
the CRT pool, and memory settings. For high-MOI data, gene blocking preserves
the co-occurring perturbation predictors. For a notebook workflow focused on
Bayesian estimates and fitted-model bundles, see [Using PerTurbo from Python](python_api.md).
