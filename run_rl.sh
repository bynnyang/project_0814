#!/usr/bin/env bash
set -e

#######################################
# 基础环境
#######################################
source /opt/ros/noetic/setup.bash

#######################################
# 训练参数
#######################################
SCRIPT=train_HOPE_sac.py
# PREPARE_SCRIPT=prepare.py

NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0   # 单卡明确指定

#######################################
# 启动 MPS（如已启动，不会重复启动）
#######################################
# if ! pgrep -f nvidia-cuda-mps-control >/dev/null; then
#     echo "[INFO] Starting CUDA MPS daemon"
#     sudo nvidia-cuda-mps-control -d
# else
#     echo "[INFO] CUDA MPS already running"
# fi

#######################################
# 启动 prepare.py（后台，限 20% 算力）
#######################################
# echo "[INFO] Starting prepare.py with MPS limit 20%"
# CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 \
# python ${PREPARE_SCRIPT} &

# PREPARE_PID=$!
# echo "[INFO] prepare.py PID = ${PREPARE_PID}"

#######################################
# 确保脚本退出时清理后台进程
#######################################
# cleanup() {
#     echo "[INFO] Stopping prepare.py (PID=${PREPARE_PID})"
#     kill ${PREPARE_PID} 2>/dev/null || true
# }
# trap cleanup EXIT

#######################################
# 启动训练
#######################################
echo "[INFO] Training script: ${SCRIPT}"

if [ "$NUM_GPUS" -le 1 ]; then
    echo "[INFO] Running single-GPU training"
    python ${SCRIPT}
else
    echo "[INFO] Running DDP training with ${NUM_GPUS} GPUs"
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        ${SCRIPT}
fi
