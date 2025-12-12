export CUDA_VISIBLE_DEVICES=5
python semantic_consistency.py \
  --data-root /data/hulk/jiacheng/SC_data/generated_data/syn_data/imagenet1k/fadrm_ipc10 \
  --checkpoint /data/hulk/jiacheng/SC_data/generated_data/new_validate/imagenet1k/fadrm_ours_slc50_ipc10_ResNet18_semantic-validation/after_stageB/checkpoint.pth.tar \
  --arch resnet18 \
  --num-classes 1000 \
  --n-pairs-per-class 500 \
  --crop-size 224 \
  --normalize \
  --batch-size 64
