#!/bin/zsh
# export CUDA_VISIBLE_DEVICES=0

GPUS=8
CONFIG_FILE=configs/recognition/swin/swin_base_patch244_window877_diving48_22k.py
RESUME=pretrained/swin_base_patch244_window877_kinetics400_22k.pth

./tools/dist_train.sh ${CONFIG_FILE} ${GPUS} \
    --cfg-options load_from=${RESUME}
