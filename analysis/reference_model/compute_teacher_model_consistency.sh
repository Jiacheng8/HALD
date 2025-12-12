export CUDA_VISIBLE_DEVICES=0
python reference_model_consistency.py \
  --arch resnet18 \
  --num-classes 1000 \
  --ref-torchvision \
  --student-ckpt /data/hulk/jiacheng/SC_data/generated_data/new_validate/imagenet1k/fadrm_ours_slc100_ipc10_ResNet18_semantic-validation/after_stageB/checkpoint.pth.tar\
  --data-dir /data/hulk/jiacheng/Common/test_data/imagenet1k \
  --num-crops 10