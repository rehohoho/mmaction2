#!/bin/zsh
# source activate swin
./tools/dist_test.sh configs/recognition/swin/swin_base_patch244_window1677_sthv2_hdf5.py pretrained_models/swin_base_patch244_window1677_sthv2.pth 2 --eval top_k_accuracy
# python tools/test.py configs/recognition/swin/swin_base_patch244_window1677_sthv2_hdf5.py pretrained_models/swin_base_patch244_window1677_sthv2.pth --eval top_k_accuracy