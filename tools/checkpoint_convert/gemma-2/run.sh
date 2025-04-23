#!/bin/bash
# Usage:
#   tools/checkpoint_convert/run.sh <start> <end> <step> <exp_name> <model_name>
#
# 例:
#   tools/checkpoint_convert/run.sh 5000 10000 5000 gemma3_4b_exp7 gemma3-4b

set -euo pipefail

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
export HF_HOME=/mnt/filestore/.cache
export HUGGINGFACE_HUB_CACHE=/mnt/filestore/.cache
export PYTHONPATH=/mnt/filestore/gemma2_sft


echo "----- MaxText -> HF 変換処理を開始 -----"
bash tools/checkpoint_convert/gemma-2/maxtext_to_hf.sh "${start}" "${end}" "${step}" "${exp_name}" "${model_name}"

echo "----- Hugging Face へのアップロード処理を開始 -----"
bash tools/checkpoint_convert/gemma-2/upload.sh "${start}" "${end}" "${step}" "${exp_name}" "${model_name}"

echo "----- すべての処理が完了しました -----"
