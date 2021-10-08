#!/bin/zsh
# export CUDA_VISIBLE_DEVICES=0

GPUS=2
CONFIG_FILE=configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb.py
# CONFIG_FILE=configs/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_diving48_rgb.py
# RESUME=

./tools/dist_train.sh ${CONFIG_FILE} ${GPUS}
# --resume-from ${RESUME}
