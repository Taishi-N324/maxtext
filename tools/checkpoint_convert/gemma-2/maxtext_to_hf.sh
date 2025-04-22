#!/bin/bash
# Usage:
#   tools/checkpoint_convert/maxtext_to_hf.sh <start> <end> <step> <exp_name> <model_name>
#
# 例:
#   tools/checkpoint_convert/maxtext_to_hf.sh 1 10 1 gemma3-4b gemma3-4b

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


# ベースディレクトリの設定（出力先はローカル）
BASE_DIR="/mnt/filestore"
OUTPUT_DIR="${BASE_DIR}/checkpoints_hf/${exp_name}/checkpoints"

echo "チェックポイント変換を開始します..."

# 出力ディレクトリの作成（必要ならさらにサブディレクトリも）
mkdir -p "${OUTPUT_DIR}"

# ループ: 指定された start から end まで step ごとに処理
for (( iter = start; iter <= end; iter += step ))
do
    echo "=============================="
    echo "チェックポイント ${iter} の変換を開始"
    echo "=============================="

    # gs://のパスを利用してチェックポイントディレクトリを指定
    CHECKPOINT_DIR="gs://swallow-asia-b2/checkpoints/${exp_name}/checkpoints/${iter}/items/"

    # MaxTextの変換スクリプトの実行
    python3 MaxText/llama_mistral_mixtral_orbax_to_hf.py \
        MaxText/configs/base.yml \
        base_output_directory="${BASE_DIR}/checkpoints_hf" \
        load_parameters_path="${CHECKPOINT_DIR}" \
        run_name=gemma2_run \
        model_name="${model_name}" \
        hardware="${OUTPUT_DIR}/${iter}"
done
