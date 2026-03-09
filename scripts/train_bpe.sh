#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

DATA_DIR="$PROJECT_ROOT/data"
RESULTS_DIR="$PROJECT_ROOT/results"
TRAIN_PY="$PROJECT_ROOT/train.py"

# 确保输出目录存在
mkdir -p "$RESULTS_DIR"

# 获取实验类型并转换为小写
EXP_TYPE=$(echo "$1" | tr '[:upper:]' '[:lower:]')

# --- 训练逻辑 ---
# Typer 默认会将 --data_path 映射为 --data-path
run_tinystory() {
    echo "开始训练 TinyStories (10K Vocab)..."
    python "$TRAIN_PY" train_bpe \
        --data-path "$DATA_DIR/TinyStoriesV2-GPT4-train.txt" \
        --vocab-size 10000 \
        --save-path "$RESULTS_DIR/TinyStories"
}

run_owt() {
    echo "开始训练 OpenWebText (32K Vocab)..."
    python "$TRAIN_PY" train_bpe \
        --data-path "$DATA_DIR/owt_train.txt" \
        --vocab-size 32000 \
        --save-path "$RESULTS_DIR/owt_train"
}

case "$EXP_TYPE" in
    "ts"|"tinystory") run_tinystory ;;
    "owt") run_owt ;;
    *) echo "使用方法: $0 {ts|owt}"; exit 1 ;;
esac