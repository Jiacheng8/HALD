#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$PARENT_DIR")"

source "$SCRIPT_DIR/constants.sh"

mode= # choose from fadrm, sre2l, lpld, and rded
Model_Name= # choose from ResNet18, ResNet50, and  ResNet101

hyper_loc="$PARENT_DIR/hyper.yaml"

T=20
bs=16

mkdir -p "$SCRIPT_DIR/logs"

for slc in 100
do
  for ipc in 10
  do
    # Set soft_epochs for the given slc, total should be 300
    soft_epochs=150
    hard_epochs=$((300 - soft_epochs))

    n=$((slc / ipc))

    # Path settings
    ODP="${Generated_Data_Path}/syn_data/${Dataset_Name}/${mode}_ipc${ipc}"
    FKD="${Generated_Data_Path}/new_labels/${Dataset_Name}/${mode}_bs16_slc${slc}_ipc${ipc}"
    OPD="${Generated_Data_Path}/validate_output"

    EXP_NAME="${mode}_slc${slc}_ipc${ipc}_${Model_Name}"
    WANDB_PROJECT="${Dataset_Name}_${Model_Name}"

    echo ">>> Running experiment: slc=${slc}, ipc=${ipc}, soft_epochs=${soft_epochs}, hard_epochs=${hard_epochs}, n=${n}"

    python "$PARENT_DIR/train_fkd.py" \
        --config-path "$hyper_loc" \
        --model "$Model_Name" \
        --ipc "$ipc" \
        --wandb-project "$WANDB_PROJECT" \
        --exp-name "$EXP_NAME" \
        --original-data-path "$ODP" \
        --fkd-path "$FKD" \
        --output-dir "$OPD" \
        --batch-size "$bs" \
        --hard-epochs "$hard_epochs" \
        --soft-epochs "$soft_epochs" \
        --n "$n" \
        --dataset-name "$Dataset_Name" \
        --gradient-accumulation-steps 1 \
        --mix-type "cutmix" \
        --cos \
        -j 16 \
        -T "$T" \
        --val-dir "$val_dir"
  done
done
