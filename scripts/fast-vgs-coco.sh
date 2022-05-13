#!/bin/sh
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate my_env
export CUDA_VISIBLE_DEVICES=0

data_root=/work/vjsalt22/poheng/coco
raw_audio_base_path=/work/vjsalt22/dataset/coco/SpokenCOCO
fb_w2v2_weights_fn=/work/vjsalt22/poheng/fb_w2v2/wav2vec_small.pt
exp_dir=/work/vjsalt22/poheng/exp_coco

python \
../run_spokencoco.py \
--data_root ${data_root} \
--raw_audio_base_path ${raw_audio_base_path} \
--fb_w2v2_weights_fn ${fb_w2v2_weights_fn} \
--exp_dir ${exp_dir} \
--num_workers 4 \
--batch_size 64 \
--val_batch_size 64 \
--val_cross_batch_size 200 \
--n_epochs 20 \
--n_print_steps 200 \
--n_val_steps 6400 \
--lr 0.05 \
--warmup_fraction 0.1 \
--xtrm_layers 2 \
--trm_layers 6 \
--fine_matching_weight 0 \
--coarse_matching_weight 1 \
--coarse_to_fine_retrieve \
--feature_grad_mult 0. \
--layer_use 7 \
--solo_loss VICReg
