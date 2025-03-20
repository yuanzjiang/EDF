#!/bin/bash
# 激活dd环境
eval "$(conda shell.bash hook)"
conda activate dd

cd buffer
python3 buffer.py \
--dataset=Tiny \
--model=ConvNetD5 \
--train_epochs=100 \
--num_experts=100 \
--buffer_path="../buffer_storage_1/in1k" \
--data_path="/home/fangyuanchen/code/dataset/imagenet" \
--rho_max=0.01 \
--rho_min=0.01 \
--alpha=0.3 \
--lr_teacher=0.01 \
--mom=0. \
--batch_train=256