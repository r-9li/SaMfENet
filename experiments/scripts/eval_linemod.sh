#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 /home/user/LZX/DenseFusion-test1/tools/eval_linemod_1.py --dataset_root /home/user/LZX/Linemod_preprocessed\
  --model /home/user/LZX/DenseFusion-24/trained_models/linemod/pose_model_8_0.008757849160650142.pth