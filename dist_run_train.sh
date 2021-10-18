#!/bin/zsh
# multi-gpu training
bash tools/dist_train.sh configs/recognition/swin/swin_base_patch244_window1677_sthv2_hdf5.py \
                         2 \
                        #  --cfg-options model.backbone.pretrained=pretrained_models/swin_base_patch244_window1677_sthv2.pth \
                        #  model.backbone.use_checkpoint=True
