#!/bin/bash

# ===== 環境設定 =====
export HF_HOME=/mnt/filestore/.cache
export HUGGINGFACE_HUB_CACHE=/mnt/filestore/.cache

# ===== 基本設定 =====
RUN_NAME="gemma2_2b_grpo_exp1_test"
MODEL_NAME="gemma2-2b"
DATASET_PATH="open-r1/OpenR1-Math-220k"
TOKENIZER_PATH="tokyotech-llm/Gemma-2-Llama-Swallow-2b-it-v0.1"
BASE_OUTPUT_DIRECTORY="/mnt/filestore/checkpoints_grpo"

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

PYTHON_PATH="/home/shige/maxtext/venv/bin/python3"
export PYTHONPATH="/home/shige/maxtext:${PYTHONPATH}"

# 実行情報表示
echo "===== Starting GRPO Training ====="
echo "Run Name: ${RUN_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Output Directory: ${BASE_OUTPUT_DIRECTORY}"
echo "Steps: ${STEPS}"
echo "================================"

${PYTHON_PATH} -u MaxText/experimental/rl/grpo_trainer.py \
    MaxText/experimental/rl/grpo_problem.yml \
    run_name=${RUN_NAME} \
    model_name=${MODEL_NAME} \
    load_from_hf_repo=tokyotech-llm/Gemma-2-Llama-Swallow-2b-it-v0.1 \
    hf_path=${DATASET_PATH} \
    tokenizer_path=${TOKENIZER_PATH} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    steps=${STEPS} \
    log_period=${LOG_PERIOD} \
    checkpoint_period=${CHECKPOINT_PERIOD} \
    remat_policy=${REMAT_POLICY} \
    enable_data_shuffling=${ENABLE_DATA_SHUFFLING} \
    data_shuffle_seed=${DATA_SHUFFLE_SEED} \
    use_vertex_tensorboard=false \
    use_iota_embed=true

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "===== GRPO training script finished successfully. ====="
else
    echo "===== GRPO training script failed with exit code ${EXIT_CODE}. ====="
fi

exit $EXIT_CODE