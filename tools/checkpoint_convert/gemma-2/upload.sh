#!/bin/bash
# Usage:
#   tools/checkpoint_convert/upload.sh <start> <end> <step> <exp_name> <model_name>
#
# 例:
#   tools/checkpoint_convert/upload.sh 1 10 1 gemma3-4b gemma3-4b

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

# ソースディレクトリと出力ディレクトリの設定
OUTPUT_DIR="/mnt/filestore/checkpoints_hf/${exp_name}/checkpoints"

###############################################################################
# ここから Hugging Face へのアップロード処理
###############################################################################

# アップロード用設定
REPO_USERNAME="tokyotech-llm"
UPLOAD_SCRIPT="scripts/ckpt_convert/upload_to_hf.py"

echo "Hugging Faceへのチェックポイントアップロードを開始します..."

# ループ: 指定された start から end まで step ごとに処理（変換したチェックポイントのアップロード）
for (( iter = start; iter <= end; iter += step ))
do
    echo "=============================="
    echo "チェックポイント ${iter} のアップロードを開始"
    echo "=============================="

    CHECKPOINT_PATH="${OUTPUT_DIR}/${iter}"
    REPO_ID="${REPO_USERNAME}/${exp_name}-checkpoint-${iter}"

    # チェックポイントが存在するか確認
    if [ ! -d "${CHECKPOINT_PATH}" ]; then
        echo "エラー: チェックポイントディレクトリが見つかりません: ${CHECKPOINT_PATH}"
        continue
    fi

    # モデルごとに HF 用の config ディレクトリを場合分けしてコピー
    case "${model_name}" in
        "gemma2-2b")
            cp -r tools/hf/configs/gemma-2-2b-it/* "${CHECKPOINT_PATH}"
            ;;
        "gemma2-9b")
            cp -r tools/hf/configs/gemma-2-9b-it/* "${CHECKPOINT_PATH}"
            ;;
        "gemma2-27b")
            cp -r tools/hf/configs/gemma-2-27b-it/* "${CHECKPOINT_PATH}"
            ;;
    esac

    # アップロードスクリプトの実行
    python3 "${UPLOAD_SCRIPT}" "${CHECKPOINT_PATH}" "${REPO_ID}" "${iter}"

    if [ $? -eq 0 ]; then
        echo "チェックポイント ${iter} のアップロードが成功しました"
    else
        echo "チェックポイント ${iter} のアップロードが失敗しました"
    fi

    echo ""
done

echo "すべてのチェックポイントのアップロードが完了しました"
