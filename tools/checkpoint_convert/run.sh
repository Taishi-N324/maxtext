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

echo "----- MaxText -> HF 変換処理を開始 -----"
bash tools/checkpoint_convert/maxtext_to_hf.sh "${start}" "${end}" "${step}" "${exp_name}" "${model_name}"

echo "----- 言語ウェイトマッピング処理を開始 -----"
bash tools/checkpoint_convert/mapping_language_weight.sh "${start}" "${end}" "${step}" "${exp_name}" "${model_name}"

echo "----- Hugging Face へのアップロード処理を開始 -----"
bash /mnt/filestore/gemma3/maxtext/tools/checkpoint_convert/upload.sh "${start}" "${end}" "${step}" "${exp_name}" "${model_name}"

echo "----- すべての処理が完了しました -----"
