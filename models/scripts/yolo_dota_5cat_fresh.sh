#!/usr/bin/env bash

# Train from scratch

python ../yolo_dota_train.py \
    --data './datasets/DOTAv1_5cat.yaml' \
    --weights 'yolov8s-obb.yaml' \
    --batch-size 32 \
    --name 'yolov8s-obb-dota5cat-fresh'
