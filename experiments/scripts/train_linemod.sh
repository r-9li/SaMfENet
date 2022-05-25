#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1

python3 /home/user/LZX/DenseFusion-22/tools/train_m.py --dataset linemod\
  --dataset_root /home/user/LZX/Linemod_preprocessed