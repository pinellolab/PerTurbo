#!/usr/bin/env bash
set -euo pipefail

input=${1:?Supply the prepared GSE120861_at_scale_screen.h5mu input}
output=${2:?Supply a fresh output directory}
shift 2
example_dir=$(cd -- "$(dirname -- "$0")" && pwd)

exec perturbo \
  --input "$input" \
  --out-dir "$output" \
  --device gpu \
  --no-progress-bar \
  --modality-key gene \
  --perturbation-modality-key guide \
  --perturbation-element-varm-key guide_intended_target_pairs \
  --perturbation-element-names-uns-key intended_targets \
  --library-size-key total_umis \
  --size-factor-mode observed \
  --continuous-covariates total_umis percent_mito \
  --batch-covariate prep_batch \
  --max-control-cells 207324 \
  --max-chunk-size 50000 \
  --num-steps-control 4000 \
  --num-steps-betas 1000 \
  --step-size 0.01 \
  --cis-pairs "$example_dir/gasperini-cis-candidates.parquet" \
  --crt \
  --crt-pool all-cells \
  --crt-mechanism propensity \
  --crt-tail-families saddlepoint \
  --crt-saddlepoint-only \
  --crt-two-sided equal-tail \
  --crt-polish-baseline \
  --crt-gene-chunk-size 500 \
  --crt-max-gather-gib 1 \
  --save-model-params \
  --prior normal \
  --guide-effect-strategy shared \
  --gene-chunk-size 32 \
  --report-low-information \
  --minibatch-size-control 20000 \
  --likelihood negbin \
  "$@"
