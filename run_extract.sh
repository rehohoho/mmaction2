#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

CONFIG_FILE=configs/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_diving48_rgb.py
CHECKPOINT_FILE=work_dirs/tpn_slowonly_r50_8x8x1_150e_diving48_rgb/epoch_149.pth
RESULT_FILE=work_dirs/tpn_slowonly_r50_8x8x1_150e_diving48_rgb/epoch_149_backbone_features
NUM_GPU=2

./tools/dist_extract.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPU} \
   --out ${RESULT_FILE} \
2>&1 | tee ${RESULT_FILE}.log
