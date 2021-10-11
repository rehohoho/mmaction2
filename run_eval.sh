#!/bin/zsh
export CUDA_VISIBLE_DEVICES=0,1

CONFIG_FILE=configs/recognition/swin/swin_base_patch244_window877_diving48_22k.py
CHECKPOINT_FILE=work_dirs/swin_base_patch244_window877_diving48_22k/latest.pth
RESULT_FILE=work_dirs/swin_base_patch244_window877_diving48_22k/latest_results.pkl
EVAL_METRICS=top_k_accuracy

# ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} 2 \
#    --out ${RESULT_FILE} \
#    --eval ${EVAL_METRICS} \
# 2>&1 | tee ${RESULT_FILE}.log

for i in $(seq 20 30)
do
    CKPT_FILE=work_dirs/swin_base_patch244_window877_diving48_22k/epoch_${i}.pth
    R_FILE=work_dirs/swin_base_patch244_window877_diving48_22k/epoch_${i}_results.pkl
    echo $CKPT_FILE $R_FILE
    ./tools/dist_test.sh ${CONFIG_FILE} ${CKPT_FILE} 2 \
        --out ${R_FILE} \
        --eval ${EVAL_METRICS} \
    2>&1 | tee ${R_FILE}.log
done
