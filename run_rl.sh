#!/usr/bin/env bash
set -e

#######################################
# 基础环境
#######################################
# ROS 环境（你现在用的）
source /opt/ros/noetic/setup.bash

# 如果你用 conda / venv，这里也可以加
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env

#######################################
# 训练参数
#######################################
SCRIPT=train_HOPE_sac.py

# 使用 GPU 数量（单卡=1，多卡=4）
NUM_GPUS=1

# 指定可见 GPU（可选）
export CUDA_VISIBLE_DEVICES=0,1,2,3

#######################################
# 启动逻辑
#######################################
echo "[INFO] Training script: ${SCRIPT}"
if [ "$NUM_GPUS" -le 1 ]; then
    echo "[INFO] Running new_single-GPU training"
    python ${SCRIPT}
else
    echo "[INFO] Running DDP training with ${NUM_GPUS} GPUs"
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        ${SCRIPT}
fi
