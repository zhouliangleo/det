#!/usr/bin/env bash

model_name="e2e_faster_rcnn_R-50-C4_1x"
model_name="e2e_faster_rcnn_R-50-FPN_1x"
mkdir -p ./logs/$model_name
DATE=`date '+%Y-%m-%d-%H-%M-%S'`

CUDA_VISIBLE_DEVICES=0,1  python tools/train_net_step.py --dataset car\
        --cfg configs/baselines/$model_name.yaml \
        --load_detectron ./models/$model_name.pkl   \
	--use_tfboard --bs 4 --nw 4 --disp_interval 100 \
2>&1 | tee -a  ./logs/$model_name/${DATE}-logs.out 
