export CUDA_VISIBLE_DEVICES=5

img_mode=fadrm
python compute_loss_grid.py \
  --model /data/hulk/jiacheng/SC_data/generated_data/new_validate/imagenet1k/${img_mode}_full/checkpoint.pth.tar\
  --ref /data/hulk/jiacheng/SC_data/generated_data/new_validate/imagenet1k/${img_mode}_ours_slc50_ipc10_ResNet18/checkpoint.pth.tar \
  --trainset /data/hulk/jiacheng/SC_data/generated_data/syn_data/imagenet1k/${img_mode}_ipc10 \
  --testset /data/hulk/jiacheng/Common/test_data/imagenet1k \
  --prefix ideal_${img_mode}
