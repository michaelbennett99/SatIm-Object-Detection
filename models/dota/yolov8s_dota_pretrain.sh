#!/usr/bin/env bash

# Train from pre-trained weights

python ../yolo_dota_train.py \
    --weights 'yolov8s-obb.pt' \
    --batch-size 32 \
    --name 'yolov8s-obb-pretrain' \
    --pretrained
