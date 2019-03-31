#!/usr/bin/env bash

model_name="e2e_faster_rcnn_R-50-FPN_1x"

CUDA_VISIBLE_DEVICES=0  python tools/test_net.py --dataset coco2014 \
        --cfg configs/baselines/$model_name.yaml \
        --load_ckpt  ./Outputs/e2e_faster_rcnn_R-50-FPN_1x/Jan01-20-49-39_sedlight_step/ckpt/model_step109999.pth \
2>&1 | tee -a  ./logs/$model_name/${DATE}-testlogs.out 
