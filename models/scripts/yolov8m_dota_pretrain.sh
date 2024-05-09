#!/usr/bin/env bash

# Train from pre-trained weights

python ../yolo_dota_train.py \
    --weights 'yolov8m-obb.pt' \
    --batch-size 16 \
    --name 'yolov8m-obb-pretrain' \
    --pretrained
