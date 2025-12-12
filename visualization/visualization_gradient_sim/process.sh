export CUDA_VISIBLE_DEVICES=0
python process.py \
    --train-dir /data/hulk/jiacheng/SC_data/generated_data/syn_data/imagenet1k/fadrm_ipc10 \
    --save-path ./fadrm.npy\