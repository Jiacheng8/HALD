#!/bin/bash

export CUDA_VISIBLE_DEVICES=

# Configuration
img_mode= # choose from fadrm, sre2l, lpld, and rded
ipc=
SLC=

wpg=1
teacher=ResNet18

# Relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$PARENT_DIR")"

# Load constants (e.g., $Generated_Path, $Dataset_Name)
source $SCRIPT_DIR/constants.sh

# Loop over IPC and SLC values

echo "🔁 Running for ipc=$ipc, SLC=$SLC"
python $PARENT_DIR/relabel_slc.py \
  --img-mode $img_mode \
  --workers-per-gpu $wpg \
  --syn-data-path $Generated_Path/generated_data/syn_data/$Dataset_Name/${img_mode}_ipc${ipc} \
  --fkd-path $Generated_Path/generated_data/new_labels/$Dataset_Name/${img_mode} \
  --model-choice $teacher \
  --online \
  -b 16 \
  -j 10 \
  --dataset-name $Dataset_Name \
  --SLC $SLC \
  --fkd-seed 42 \
  --min-scale-crops 0.08 \
  --max-scale-crops 1 \
  --mode 'fkd_save' \
  --mix-type 'cutmix'
echo "✅ Done ipc=$ipc, SLC=$SLC"

