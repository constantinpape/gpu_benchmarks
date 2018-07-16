#! /bin/bash

export CUDA_VISIBLE_DEVICES=$1
# python -u inference_single_gpu.py $@
./inference_single_gpu.py $@
