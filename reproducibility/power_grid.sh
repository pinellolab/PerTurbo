#!/bin/bash

OUTPUT_DIR=power_tests
DATA_DIR=/Users/ljb80/Data/gasperini_pilot

mkdir -p $OUTPUT_DIR
echo '*' > $OUTPUT_DIR/.gitignore

SCRIPT=reproducibility/gasperini_power.py

FRACS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# FRACS=(1.0)
for FRAC_SUBSAMPLE in "${FRACS[@]}"
do
  for FRAC_DOWNSAMPLE in "${FRACS[@]}"
  do
    echo testing subsample_frac: $FRAC_SUBSAMPLE downsample_frac: $FRAC_DOWNSAMPLE
    python $SCRIPT $DATA_DIR $OUTPUT_DIR $FRAC_SUBSAMPLE $FRAC_DOWNSAMPLE
  done
done
