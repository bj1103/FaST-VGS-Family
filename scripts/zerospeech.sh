#!/bin/sh
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate my_env
export CUDA_VISIBLE_DEVICES=0

data_root=/work/vjsalt22/poheng/flickr
raw_audio_base_path=/work/vjsalt22/dataset/flickr/flickr_audio
fb_w2v2_weights_fn=/work/vjsalt22/poheng/fb_w2v2/wav2vec_small.pt
exp_dir=/work/vjsalt22/poheng/2nd_best_InfoNCE_BT

python \
../run_semantic.py \
--exp_dir ${exp_dir} \
--num_workers 4 \
--batch_size 1 \
--val_batch_size 1 \
--val_cross_batch_size 1 \
--n_epochs 250 \
--n_print_steps 312 \
--n_val_steps 624 \
--lr 0.0001 \
--warmup_fraction 0.1 \
--xtrm_layers 2 \
--trm_layers 6 \
--fine_matching_weight 0 \
--coarse_matching_weight 1.0 \
--coarse_to_fine_retrieve \
--feature_grad_mult 0. \
--layer_use 7 \
--validate \
--task_input_dir /work/vjsalt22/dataset/zerospeech2021/semantic/ \
--output_result_dir /work/vjsalt22/poheng/zerospeech2021 \
--task_name semantic
