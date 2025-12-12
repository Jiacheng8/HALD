export CUDA_VISIBLE_DEVICES=5
python LVSD.py \
  --data-root /data/hulk/jiacheng/SC_data/generated_data/syn_data/imagenet1k/fadrm_ipc10 \
  --model shufflenet_v2_x1_0 \
  --max-images 1000 \
  --samples-per-image 8 \
  --num-workers 8
