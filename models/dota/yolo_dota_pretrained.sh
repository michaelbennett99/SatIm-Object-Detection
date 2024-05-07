#!/usr/bin/env bash

# Continue training from model already pretrained on DOTA

python ../yolo_dota_train.py \
    --weights 'yolov8n-obb.pt' \
    --batch-size 64 \
    --pretrained
