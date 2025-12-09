#!/bin/bash

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=1

# 设置基本参数
SUBJECT="zhang_1111"
SEQUENCE="EMO-1"
ITER=100000
timeout_duration="10m"

# 设置 points_per_face 数组
ppfs=(1 5 10 50 100)

# 循环遍历 ppfs 数组
for ppf in "${ppfs[@]}"; do
  OUTPUT="GA_${SUBJECT}_${SEQUENCE}_${ppf}"

  # 训练命令
  echo "Starting training with points_per_face=${ppf} and output=${OUTPUT}"

  timeout $timeout_duration python train.py \
  -s "../output/export/${SUBJECT}_${SEQUENCE}" \
  -m "../output/gaussian/${OUTPUT}" \
  --iterations ${ITER} --interval 1000 \
  --eval --bind_to_mesh --white_background \
  --port 60000 \
  --points_per_face ${ppf}

  echo "Training for points_per_face=${ppf} stopped or completed after ${timeout_duration}"
done

echo "All training tasks have been completed."
