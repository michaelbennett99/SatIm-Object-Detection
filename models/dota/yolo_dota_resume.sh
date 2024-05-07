#!/usr/bin/env bash

# Resume training from a specified checkpoint
# takes a single argument: the path to the checkpoint file

python ../yolo_dota_train.py --weights $1 \
    --resume \
    --batch-size 64
