#!/bin/bash

# 設定
REPO_USERNAME="tokyotech-llm"
EXPERIMENT_NAME="gemma2_2b_exp2"
BASE_DIR="/mnt/filestore/checkpoints_hf/gemma2_2b_exp2/checkpoints"
UPLOAD_SCRIPT="scripts/ckpt_convert/upload_to_hf.py"
CHECKPOINTS=(2500 5000 7500 10000 12500 15000)

echo "Hugging Faceへのチェックポイントアップロードを開始します..."

# スクリプトの存在確認
if [ ! -f "${UPLOAD_SCRIPT}" ]; then
    echo "エラー: アップロードスクリプトが見つかりません: ${UPLOAD_SCRIPT}"
    exit 1
fi

# 各チェックポイントをアップロード
for CP in "${CHECKPOINTS[@]}"; do
    echo "=============================="
    echo "チェックポイント ${CP} のアップロードを開始"
    echo "=============================="
    
    CHECKPOINT_PATH="${BASE_DIR}/${CP}"
    REPO_ID="${REPO_USERNAME}/${EXPERIMENT_NAME}-checkpoint-${CP}"
    
    # チェックポイントが存在するか確認
    if [ ! -d "${CHECKPOINT_PATH}" ]; then
        echo "エラー: チェックポイントディレクトリが見つかりません: ${CHECKPOINT_PATH}"
        continue
    fi
    
    # 既存のスクリプトを実行
    python3 "${UPLOAD_SCRIPT}" "${CHECKPOINT_PATH}" "${REPO_ID}" "${CP}"
    
    if [ $? -eq 0 ]; then
        echo "チェックポイント ${CP} のアップロードが成功しました"
    else
        echo "チェックポイント ${CP} のアップロードが失敗しました"
    fi
    
    echo ""
done

echo "すべてのチェックポイントのアップロードが完了しました"
