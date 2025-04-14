#!/bin/bash
# Usage:
#   ./tools/checkpoint_convert/mapping_language_weight.sh <start> <end> <step> <exp_name> <model_name>
#
# 例:
#   ./tools/checkpoint_convert/mapping_language_weight.sh 1 10 1 gemma3-4b gemma3-4b

export HF_HOME=/mnt/filestore/.cache
export HUGGINGFACE_HUB_CACHE=/mnt/filestore/.cache
export JAX_PLATFORMS=cpu
export JAX_THREADED_P_MAP=false

# 引数の数をチェック
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <start> <end> <step> <exp_name> <model_name>"
    exit 1
fi

# 引数の取得
start=$1
end=$2
step=$3
exp_name=$4
model_name=$5


# model_name に応じた target-model の設定
if [ "${model_name}" = "gemma3-4b" ]; then
    target_model="google/gemma-3-4b-pt"
elif [ "${model_name}" = "gemma3-12b" ]; then
    target_model="google/gemma-3-12b-pt"
elif [ "${model_name}" = "gemma3-27b" ]; then
    target_model="google/gemma-3-27b-pt"
else
    echo "Unknown model_name: ${model_name}"
    exit 1
fi

# ソースディレクトリと出力ディレクトリの設定
SOURCE_DIR="/mnt/filestore/checkpoints_hf/${exp_name}/checkpoints"
OUTPUT_DIR="/mnt/filestore/checkpoints_hf/${exp_name}_multimodal/checkpoints"

# # 出力ディレクトリの作成
mkdir -p "${OUTPUT_DIR}"

echo "言語ウェイトの変換を開始します..."

# ループ: 指定された start から end まで step ごとに処理
for (( iter = start; iter <= end; iter += step ))
do
    echo "------------------------------"
    echo "チェックポイント ${iter} の変換を開始"
    echo "------------------------------"

    python tools/checkpoint_convert/gemma3/language_weight_map.py \
        --source-dir "${SOURCE_DIR}" \
        --target-model "${target_model}" \
        --output-dir "${OUTPUT_DIR}" \
        --checkpoint ${iter} \
        --dtype "bfloat16"
done
