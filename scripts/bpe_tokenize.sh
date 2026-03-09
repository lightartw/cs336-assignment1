#!/bin/bash

set -e

# --- 1. 环境初始化 ---
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
DATA_DIR="$PROJECT_ROOT/data"
RESULTS_DIR="$PROJECT_ROOT/results"
TRAIN_PY="$PROJECT_ROOT/train.py"

# --- 2. 映射表 ---
declare -A VOCAB_DIRS=( ["owt"]="owt_train" ["ts"]="TinyStories" )
declare -A DATA_FILES=( ["owt"]="owt_train.txt" ["ts"]="TinyStoriesV2-GPT4-train.txt" )

# --- 3. 核心函数 ---
run_single_compare() {
    local v_key=$1
    local d_key=$2

    # 校验输入是否存在于映射表中
    if [[ -z "${VOCAB_DIRS[$v_key]}" || -z "${DATA_FILES[$d_key]}" ]]; then
        echo "Error: Invalid arguments. Use 'owt' or 'ts'."
        exit 1
    fi

    local v_dir="${RESULTS_DIR}/${VOCAB_DIRS[$v_key]}"
    local d_file="${DATA_DIR}/${DATA_FILES[$d_key]}"

    echo ">>> Running Compare: Vocab=$v_key, Data=$d_key"
    
    python "$TRAIN_PY" compare-ratio \
        "$v_dir/vocab.json" \
        "$v_dir/merges.txt" \
        "$d_file" \
        --num-samples 10
}

run_single_encode() {
    local key=$1

    if [[ -z "${VOCAB_DIRS[$key]}" ]]; then
        echo "Error: Invalid argument '$key'. Use 'owt' or 'ts'."
        exit 1
    fi

    local v_dir="${RESULTS_DIR}/${VOCAB_DIRS[$key]}"
    local d_file="${DATA_DIR}/${DATA_FILES[$key]}"
    local out_file="${v_dir}/encoded_data.bin"

    echo ">>> Running Encode: Dataset=$key"
    echo ">>> Target Bin: $out_file"
    
    # 调用你的 Python encode-file 命令
    python "$TRAIN_PY" encode-file \
        --vocab-file "$v_dir/vocab.json" \
        --merges-file "$v_dir/merges.txt" \
        --data-file "$d_file" \
        --output-file "$out_file"
}

# --- 4. 路由逻辑 ---
COMMAND=$(echo "$1" | tr '[:upper:]' '[:lower:]')
shift 

case "$COMMAND" in
    "ratio"|"compare")
        # Usage: ./run.sh ratio <vocab_name> <data_name>
        run_single_compare "$1" "$2"
        ;;
    "encode")
        # Usage: ./run.sh encode <dataset_name>
        run_single_encode "$1"
        ;;
    "all")
        run_single_compare "ts" "ts"
        run_single_encode "ts"
        ;;
    *)
        echo "Usage: $0 {ratio|encode|all} [args...]"
        echo "Example (Ratio):  $0 ratio owt ts"
        echo "Example (Encode): $0 encode owt"
        exit 1
        ;;
esac

echo ">>> Done."