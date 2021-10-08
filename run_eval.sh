#!/bin/zsh
export CUDA_VISIBLE_DEVICES=0

CONFIG_FILE='configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb.py'
CHECKPOINT_FILE='models/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb_20210630-8df9c358.pth'
# CHECKPOINT_FILE='work_dirs/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb/epoch_1.pth'
# CHECKPOINT_FILE='work_dirs/tsm_r50_1x1x16_25e_ucf101_rgb/epoch_25.pth'
RESULT_FILE='results/result-epoch1.pkl'
EVAL_METRICS='top_k_accuracy'


python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out ${RESULT_FILE} --eval ${EVAL_METRICS}