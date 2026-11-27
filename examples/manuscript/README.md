# Full-screen analysis commands

Run a script with the prepared count file and a fresh output directory. Each example runs association testing and fits effect sizes, then writes `fitted.h5mu`. These are GPU recipes for the six experimental screens in the manuscript. For a small CPU example with example data, use the [analysis notebooks](../../docs/tutorials/index.md).

| Screen | Script | Prepared input | Cells in the rc13 baseline |
| --- | --- | --- | ---: |
| Gasperini K562 at-scale | [gasperini-k562.sh](gasperini-k562.sh) | GSE120861_at_scale_screen.h5mu | 207,324 (all cells) |
| Replogle K562 genome-wide | [replogle-k562-gwps.sh](replogle-k562-gwps.sh) | K562_gwps_raw_singlecell_01.h5ad | 75,328 controls |
| Replogle K562 essential | [replogle-k562-essential.sh](replogle-k562-essential.sh) | K562_essential_raw_singlecell_01.h5ad | 10,691 controls |
| Replogle RPE-1 | [replogle-rpe1.sh](replogle-rpe1.sh) | rpe1_raw_singlecell_01.h5ad | 11,485 controls |
| Nadig HepG2 | [nadig-hepg2.sh](nadig-hepg2.sh) | GSE264667_hepg2_raw_singlecell_01.h5ad | 4,976 controls |
| Nadig Jurkat | [nadig-jurkat.sh](nadig-jurkat.sh) | GSE264667_jurkat_raw_singlecell_01.h5ad | 12,013 controls |

For example, from the repository root:

```bash
examples/manuscript/replogle-k562-essential.sh K562_essential_raw_singlecell_01.h5ad results/k562-essential
examples/manuscript/gasperini-k562.sh GSE120861_at_scale_screen.h5mu results/gasperini
```

## Gasperini

The all-cells testing pool and the number of cells used to train the baseline are separate settings. The historical whole-screen association analysis used the all-cells pool with a 10,000-cell training baseline and skipped the effect fit. The retained empirical guide-efficiency pools also depend on a 10,000-cell baseline and targeted relative-effect fits. A separate rc12 baseline fit used all 207,324 cells and all 13,135 genes, with 4,000 steps and 20,000 cells per minibatch. The script uses that schedule, the original full-library sizes, total UMIs and mitochondrial fraction as continuous covariates, and preparation batch. It fits the baseline anew.

The script fits shared element effects in 32-gene blocks. It passes the bundled `gasperini-cis-candidates.parquet` to `--cis-pairs`. This table contains all 84,725 declared cis pairs from the original comparison, including 130 pairs from ten named positive-control groups. Its two columns contain element names and Ensembl gene IDs from the full input. PerTurbo applies one BH correction to the finite p-values among these pairs and records the declared, matched and finite counts. Other pairs retain their screen-wide and within-element adjustments.

Figure 3 compared all five methods on 82,434 common finite pairs. To use that historical comparison family, pass `--cis-pairs examples/manuscript/gasperini-cis-figure3-common.parquet`. Both files contain only identifiers, selected independently of significance. Their 2,291-row difference reflects unavailable comparator p-values. A new fit may itself have unsupported tests.

To estimate guide efficiencies with the full candidate family:

```bash
examples/manuscript/gasperini-k562.sh GSE120861_at_scale_screen.h5mu results/gasperini-relative \
  --guide-effect-strategy relative
```

The full joint run above is an rc13 recipe. The frozen all-cell stage-one fit and historical selected-gene effect fits are separate artifacts.

## Replogle and Nadig

The five low-MOI scripts use the deposited construct labels and non-targeting controls. They retain the original library-size field, batch covariate, control-anchored tests, and 1,000-step shared effect fit. GWPS uses 4,000 baseline steps with 16,384 cells per minibatch and partitions stage two by 300 constructs. The four essential-scale screens use 2,500 baseline steps and 256-gene effect blocks. The K562 essential rc12 run used 10,000 of 10,691 controls. Its rc13 command uses all 10,691. The other four retained low-MOI runs already included all eligible controls.

## Historical settings and sparse genes

The retained rc12 whole-screen commands explicitly permitted unconverged baseline coefficients. These rc13 commands use the package's strict baseline policy and request low-information annotations. Unsupported tests can therefore have missing p-values and adjusted p-values. The historical and rc13 discovery counts can differ. Extra arguments at the end of a command override scalar settings or add optional inputs. For example, `--output-format bundle` writes the compact bundle.

The low-MOI settings were transcribed from the saved rc12 launch records for the K562 genome-wide screen and the four gene-blocked essential-scale runs. To recover their baseline and test policy, use `--crt-allow-unconverged-baseline` and, for K562 essential only, `--max-control-cells 10000`. Exact reproduction also requires the original software, inputs and output settings. The frozen Gasperini stage-one checkpoint has SHA-256 `8864c2b7cdc245eeaba10238a31d67289c5f31dae57d659a69d5c458a2414551`.
