#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1

python3 /home/user/LZX/DenseFusion-30/tools/train_m.py --dataset ycb\
  --dataset_root /home/user/LZX/YCB_Video_Dataset