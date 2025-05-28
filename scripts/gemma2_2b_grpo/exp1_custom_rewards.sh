#!/bin/bash

# ===== 環境設定 =====
export HF_HOME=~/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=~/.cache/huggingface
# export FILESTORE_IP=10.12.183.170 # Removed as per user direction

# ===== 基本設定 =====
RUN_NAME="gemma2_2b_grpo_exp1_custom_rewards_test" # Modified RUN_NAME
MODEL_NAME="gemma2-2b"
DATASET_PATH="open-r1/OpenR1-Math-220k"
TOKENIZER_PATH="tokyotech-llm/Gemma-2-Llama-Swallow-2b-it-v0.1"
BASE_OUTPUT_DIRECTORY="/mnt/filestore/checkpoints_grpo_custom"

# ===== TPU/実行設定 =====
VMEM_LIMIT=114688
export LIBTPU_INIT_ARGS="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_scoped_vmem_limit_kib=${VMEM_LIMIT} --xla_tpu_enable_async_collective_fusion=true --xla_tpu_assign_all_reduce_scatter_layout --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
REMAT_POLICY="full"
ENABLE_DATA_SHUFFLING=true
DATA_SHUFFLE_SEED=42
STEPS=10
LOG_PERIOD=2
CHECKPOINT_PERIOD=50

cd /home/shige/maxtext || exit 1

export PYTHONPATH=/home/shige/maxtext:$PYTHONPATH

# 実行情報表示
echo "===== Starting GRPO Training (Custom Rewards) ====="
echo "Run Name: ${RUN_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Output Directory: ${BASE_OUTPUT_DIRECTORY}"
echo "Steps: ${STEPS}"
echo "================================"

python MaxText/experimental/rl/grpo_trainer.py \
    MaxText/experimental/rl/grpo_exp1.yml \
    run_name=${RUN_NAME}

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "===== GRPO training script (Custom Rewards) finished successfully. ====="
else
    echo "===== GRPO training script (Custom Rewards) failed with exit code ${EXIT_CODE}. ====="
fi

exit $EXIT_CODE 