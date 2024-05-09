#!/usr/bin/env bash

# Continue training from model already pretrained on DOTA

python ../yolo_dota_train.py \
    --data './datasets/DOTAv1_5cat.yaml' \
    --weights './runs/obb/yolov8s-obb-valid-fresh/weights/best.pt' \
    --batch-size 32 \
    --pretrained \
    --name 'yolov8s-obb-dota5cat-pretrained-valid'
