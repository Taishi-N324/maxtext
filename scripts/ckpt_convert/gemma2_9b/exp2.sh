#!/bin/bash

export HF_HOME=/mnt/filestore/.cache
export HUGGINGFACE_HUB_CACHE=/mnt/filestore/.cache
export JAX_PLATFORMS=cpu
export JAX_THREADED_P_MAP=false

# 変換するチェックポイント番号のリスト
CHECKPOINTS=(2500 5000 7500 10000 12500 15000 17500)

# ベースディレクトリ
BASE_DIR="/mnt/filestore"
CHECKPOINT_DIR="${BASE_DIR}/checkpoints/gemma2_9b_exp2/checkpoints"
OUTPUT_DIR="${BASE_DIR}/checkpoints_hf/gemma2_9b_exp2/checkpoints"

echo "チェックポイント変換を開始します..."

for CP in "${CHECKPOINTS[@]}"; do
    echo "=============================="
    echo "チェックポイント ${CP} の変換を開始"
    echo "=============================="
    
    # MaxTextの変換スクリプトを実行
    python3 MaxText/llama_mistral_mixtral_orbax_to_hf.py \
        MaxText/configs/base.yml \
        base_output_directory=${BASE_DIR}/checkpoints_hf \
        load_parameters_path=${CHECKPOINT_DIR}/${CP}/items/ \
        run_name=gemma9_run \
        model_name=gemma2-9b \
        hardware=${OUTPUT_DIR}/${CP}
    
    # トークナイザーをコピー
    if [ ! -f "${OUTPUT_DIR}/${CP}/tokenizer.model" ]; then
        echo "トークナイザーをコピーします: ${CP}"
        mkdir -p "${OUTPUT_DIR}/${CP}"
        cp -r ${BASE_DIR}/checkpoints/tokenizer.model ${OUTPUT_DIR}/${CP}/tokenizer.model
    else
        echo "トークナイザーは既に存在します: ${CP}"
    fi
    
    echo "チェックポイント ${CP} の変換が完了しました"
    echo ""
done

echo "すべてのチェックポイント変換が完了しました"
