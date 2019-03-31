#!/usr/bin/env bash

model_name="e2e_faster_rcnn_R-50-FPN_1x"

CUDA_VISIBLE_DEVICES=0  python tools/infer_simple.py --dataset coco2014 \
        --cfg configs/baselines/$model_name.yaml \
        --load_ckpt Outputs/e2e_faster_rcnn_R-50-FPN_1x/Mar29-10-09-24_sedlight_step/ckpt/model_step29999.pth  \
	--image_dir /home/leo/mask-rcnn.pytorch/data/cartest/img/Loc1_1 --output_dir /home/leo/data/coco/images/visval/cartest
2>&1 | tee -a  ./logs/$model_name/${DATE}-testlogs.out 
