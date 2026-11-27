#!/usr/bin/env bash
set -euo pipefail

input=${1:?Supply the prepared GSE264667_hepg2_raw_singlecell_01.h5ad input}
output=${2:?Supply a fresh output directory}
shift 2

exec perturbo \
  --input "$input" \
  --out-dir "$output" \
  --device gpu \
  --no-progress-bar \
  --perturbation-key gene_transcript \
  --control-substring non-targeting \
  --batch-covariate gem_group \
  --library-size-key UMI_count \
  --size-factor-mode observed \
  --likelihood negbin \
  --max-control-cells 1000000 \
  --max-chunk-size 50000 \
  --num-steps-control 2500 \
  --num-steps-betas 1000 \
  --step-size 0.01 \
  --crt \
  --crt-pool control-anchored \
  --crt-mechanism propensity \
  --crt-tail-families saddlepoint \
  --crt-saddlepoint-only \
  --crt-two-sided equal-tail \
  --crt-polish-baseline \
  --crt-gene-chunk-size 500 \
  --crt-max-gather-gib 1 \
  --crt-seed 0 \
  --save-model-params \
  --prior normal \
  --guide-effect-strategy shared \
  --gene-chunk-size 256 \
  --report-low-information \
  "$@"
