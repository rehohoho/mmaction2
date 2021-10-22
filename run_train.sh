#!/bin/zsh
export CUDA_VISIBLE_DEVICES=1,2

GPUS=2
CONFIG_FILE=configs/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_diving48_rgb.py
RESUME=pretrained/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb-44362b55.pth

./tools/dist_train.sh ${CONFIG_FILE} ${GPUS}
  # --cfg-options model.backbone.pretrained=${RESUME} model.backbone.use_checkpoint=True
  # --resume-from ${RESUME}
  
