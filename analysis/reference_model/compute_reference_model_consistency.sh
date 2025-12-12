export CUDA_VISIBLE_DEVICES=3
python reference_model_consistency.py \
  --arch resnet18 \
  --num-classes 1000 \
  --ref-ckpt /data/hulk/jiacheng/SC_data/generated_data/new_validate/imagenet1k/reference_model/model_epoch_300.pth \
  --student-ckpt /data/hulk/jiacheng/SC_data/generated_data/new_validate/imagenet1k/fadrm_ours_slc100_ipc10_ResNet18_semantic-validation/final_model/checkpoint.pth.tar \
  --data-dir /data/hulk/jiacheng/Common/test_data/imagenet1k \
  --num-crops 10 \
  --input-size 224
