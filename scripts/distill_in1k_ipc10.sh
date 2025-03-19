#!/bin/bash
# 激活dd环境
eval "$(conda shell.bash hook)"
conda activate dd

cd distill

CFG="../configs/ImageNet/Imagenette/ConvIN/IPC10.yaml"

python3 edf_distill.py --cfg $CFG