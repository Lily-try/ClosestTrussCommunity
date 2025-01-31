#!/bin/bash

# 设置默认参数
ROOT="../data"
METHODS=(6 12 18)  # 噪声类型编号
PTB_RATES=(0.1 0.2 0.3 0.4 0.5 0.6)  # 扰动率
DATASETS=("cora" "citeseer")  # 数据集列表

# 创建输出目录
mkdir -p logs

# 循环执行加噪操作
for dataset in "${DATASETS[@]}"; do
    echo "Processing dataset: $dataset"
    for method in "${METHODS[@]}"; do
        for rate in "${PTB_RATES[@]}"; do
            echo "Applying method $method with perturbation rate $rate on $dataset"

            # 执行 Python 脚本
            python comm_noise.py \\
                --root "$ROOT" \\
                --dataset "$dataset" \\
                --method "$method" \\
                --ptb_rate "$rate" \\
                >> logs/${dataset}_method${method}_rate${rate}.log 2>&1

            echo "Completed: $dataset, Method: $method, Rate: $rate"
        done
    done
done

echo "All tasks completed."
