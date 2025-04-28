#!/bin/bash

# hfのcache場所 (Filestore上)
export HF_HOME=/mnt/filestore/.cache
export HUGGINGFACE_HUB_CACHE=/mnt/filestore/.cache

# === 基本設定 ===
# GRPO用の出力ディレクトリ (Filestore上)
export BASE_OUTPUT_DIRECTORY="/mnt/filestore/checkpoints_grpo"
# 実行名をGRPO+9B+HFロード用に設定
export RUN_NAME="gemma2_9b_grpo_exp1_hf"
# MaxText内部のモデルアーキテクチャ名を指定
export MODEL_NAME="gemma2-9b"

# === GRPO用データセット設定 ===
export DATASET_TYPE="hf"
# 使用するデータセットパスを指定
export HF_DATASET_PATH="open-r1/OpenR1-Math-220k"
export TRAIN_SPLIT="train"

# === トークナイザー設定 ===
# 読み込むHFモデルと同じパスを指定 (推奨)
export TOKENIZER_PATH="tokyotech-llm/gemma2_9b_sft_exp5-checkpoint-1495"

# === TPU設定 (9B継続事前学習の設定を流用) ===
VMEM_LIMIT=114688
REMAT_POLICY="full"
BLOCK_SIZE=2048
export LIBTPU_INIT_ARGS="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_scoped_vmem_limit_kib=${VMEM_LIMIT} --xla_tpu_enable_async_collective_fusion=true --xla_tpu_assign_all_reduce_scatter_layout --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

# MaxTextのディレクトリに移動 (Filestore上のパス)
cd /mnt/filestore/maxtext


echo "Starting GRPO training script..."
echo "Using Dataset: ${HF_DATASET_PATH}"
echo "Using Tokenizer: ${TOKENIZER_PATH}"
echo "Loading Model from HF: tokyotech-llm/gemma2_9b_sft_exp5-checkpoint-1495"
echo "Run Name: ${RUN_NAME}"
echo "Output Directory: ${BASE_OUTPUT_DIRECTORY}"
echo "Working Directory: $(pwd)"
echo "Active Python: $(which python)" # 仮想環境が有効か確認用


python3 -u MaxText/train.py \
    MaxText/configs/base.yml \
    MaxText/experimental/rl/grpo.yml \
    load_from_hf_repo="tokyotech-llm/gemma2_9b_sft_exp5-checkpoint-1495" \
    dataset_type=${DATASET_TYPE} \
    dataset_path=${HF_DATASET_PATH} \
    train_split=${TRAIN_SPLIT} \
    tokenizer_path=${TOKENIZER_PATH} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    run_name=${RUN_NAME} \
    model_name=${MODEL_NAME} \
    # === 並列化・最適化設定 (9B継続事前学習の設定を流用) ===
    remat_policy=${REMAT_POLICY} \
    ici_fsdp_transpose_parallelism=256 \
    ici_fsdp_parallelism=-1 \
    use_iota_embed=true \
    attention=flash sa_block_q=${BLOCK_SIZE} \
    sa_block_q_dkv=${BLOCK_SIZE} \
    sa_block_q_dq=${BLOCK_SIZE} \
    train_data_columns="problem" # データセットのカラム名を指定

echo "GRPO training script finished." 