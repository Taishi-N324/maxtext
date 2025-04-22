#!/bin/bash
# hfのcache場所
export HF_HOME=/mnt/filestore/.cache
export HUGGINGFACE_HUB_CACHE=/mnt/filestore/.cache
export DATASET_TYPE="hf"
export HF_PATH="/mnt/filestore/sft_datasets/llama-3.1-swallow-instruct-v0.3-data"
export TOKENIZER_PATH="/mnt/filestore/checkpoints_hf/gemma-2-2b-it"
export BASE_OUTPUT_DIRECTORY="gs://swallow-asia-b2/checkpoints/"
export RUN_NAME="gemma2_2b_sft_exp9"
export MODEL_NAME="gemma2-2b"
export CONVERTED_CHECKPOINT="/mnt/filestore/checkpoints_maxtext/gemma2_2b_exp2/checkpoints/50000/50000/items/"
export LIBTPU_INIT_ARGS="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_enable_async_collective_fusion=true --xla_tpu_assign_all_reduce_scatter_layout --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
export PYTHONPATH=$PYTHONPATH:/mnt/filestore/gemma2_sft


cd /mnt/filestore/gemma2_sft/maxtext
python3 -u MaxText/train.py MaxText/configs/base.yml \
    dataset_type=${DATASET_TYPE} \
    hf_path=${HF_PATH} \
    train_split="train" \
    hf_eval_split="train" \
    train_data_columns="['messages']" \
    eval_data_columns="['messages']" \
    tokenizer_path=${TOKENIZER_PATH} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    run_name=${RUN_NAME} \
    model_name=${MODEL_NAME} \
    per_device_batch_size=2 \
    gradient_accumulation_steps=1 \
    max_target_length=8192 \
    steps=5989 \
    learning_rate_schedule_steps=5989 \
    learning_rate=1.0e-5 \
    cosine_learning_rate_final_fraction=0.1 \
    warmup_steps_fraction=0.1 \
    checkpoint_period=5988 \
    enable_checkpointing=true \
    ici_fsdp_transpose_parallelism=64 \
    ici_fsdp_parallelism=-1 \
    use_sft=true \
    sft_train_on_completion_only=true \
    packing=false
