#!/usr/bin/env bash

# Train from scratch

python ../yolo_dota_train.py \
    --weights 'yolov8n-obb.yaml' \
    --batch-size 32 \
    --name 'yolov8n-obb-fresh'
