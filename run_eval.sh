#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

CONFIG_FILE=configs/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_diving48_rgb.py
CHECKPOINT_FILE=work_dirs/tpn_slowonly_r50_8x8x1_150e_diving48_rgb/epoch_135.pth
RESULT_FILE=work_dirs/tpn_slowonly_r50_8x8x1_150e_diving48_rgb/epoch_135.pkl
EVAL_METRICS=top_k_accuracy
NUM_GPU=2

# ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPU} \
#    --out ${RESULT_FILE} \
#    --eval ${EVAL_METRICS} \
# 2>&1 | tee ${RESULT_FILE}.log

for i in $(seq 130 150)
do
    CKPT_FILE=work_dirs/tpn_slowonly_r50_8x8x1_150e_diving48_rgb/epoch_${i}.pth
    R_FILE=work_dirs/tpn_slowonly_r50_8x8x1_150e_diving48_rgb/epoch_${i}_results.pkl
    echo $CKPT_FILE $R_FILE
    ./tools/dist_test.sh ${CONFIG_FILE} ${CKPT_FILE} ${NUM_GPU} \
        --out ${R_FILE} \
        --eval ${EVAL_METRICS} \
    2>&1 | tee ${R_FILE}.log
done
