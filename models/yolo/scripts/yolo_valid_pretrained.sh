#!/usr/bin/env bash

# Continue training from model already pretrained on DOTA

python ../yolo_dota_train.py \
    --data './datasets/valid.yaml' \
    --weights 'yolov8s-obb.pt' \
    --batch-size 32 \
    --pretrained \
    --name 'yolov8s-obb-valid-pretrained'
