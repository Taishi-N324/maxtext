#!/bin/bash

# hfのcache場所
export HF_HOME=/mnt/filestore/.cache
export HUGGINGFACE_HUB_CACHE=/mnt/filestore/.cache

export DATASET_TYPE="grain"
export GRAIN_TRAIN_FILES="/mnt/filestore/gemma_tpu_grain/**/*.arecord"
export GRAIN_EVAL_FILES="/mnt/filestore/gemma_tpu_grain/**/*.arecord"
export TOKENIZER_PATH="/mnt/filestore/checkpoints/gemma3_tokenizer/tokenizer.model" #modelファイルのパス
export BASE_OUTPUT_DIRECTORY="gs://swallow-asia-b2/checkpoints/"
export RUN_NAME="gemma3_4b_exp3"
export MODEL_NAME="gemma3-4b"      
export CONVERTED_CHECKPOINT="/mnt/filestore/checkpoints_maxtext/gemma3-4b/0/items"



export LIBTPU_INIT_ARGS="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_enable_async_collective_fusion=true --xla_tpu_assign_all_reduce_scatter_layout --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

cd /mnt/filestore/gemma3/maxtext

python3 -u MaxText/train.py MaxText/configs/base.yml \
    dataset_type=${DATASET_TYPE} \
    grain_train_files=${GRAIN_TRAIN_FILES} \
    grain_eval_files=${GRAIN_EVAL_FILES} \
    tokenizer_path=${TOKENIZER_PATH} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    run_name=${RUN_NAME} \
    model_name=${MODEL_NAME} \
    per_device_batch_size=2 \
    gradient_accumulation_steps=2 \
    grain_worker_count=8 \
    max_target_length=8192 \
    steps=50001 \
    learning_rate_schedule_steps=50001 \
    learning_rate=2.5e-5 \
    cosine_learning_rate_final_fraction=0.1 \
    warmup_steps_fraction=0.04 \
    checkpoint_period=500 \
    enable_checkpointing=true \
    ici_fsdp_transpose_parallelism=128 \
    ici_fsdp_parallelism=-1
