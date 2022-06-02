#!/bin/sh
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate my_env
# export CUDA_VISIBLE_DEVICES=0,1,2

data_root=/ssd/bj1103/flickr8k
raw_audio_base_path=/ssd/bj1103/flickr_audio
fb_w2v2_weights_fn=/ssd/bj1103/fb_w2v/wav2vec_small.pt
exp_dir=/ssd/bj1103/exp_flickr

python3 \
../run_flickr8k.py \
--data_root ${data_root} \
--raw_audio_base_path ${raw_audio_base_path} \
--fb_w2v2_weights_fn ${fb_w2v2_weights_fn} \
--exp_dir ${exp_dir} \
--num_workers 4 \
--batch_size 32 \
--val_batch_size 32 \
--val_cross_batch_size 32 \
--n_epochs 250 \
--n_print_steps 234 \
--n_val_steps 937 \
--lr 0.0001 \
--warmup_fraction 0.1 \
--xtrm_layers 2 \
--trm_layers 6 \
--fine_matching_weight 0 \
--coarse_matching_weight 1.0 \
--coarse_to_fine_retrieve \
--feature_grad_mult 0. \
--layer_use 7 \