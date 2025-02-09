#!/bin/bash

# hfのcache場所
export HF_HOME=/mnt/filestore/.cache
export HUGGINGFACE_HUB_CACHE=/mnt/filestore/.cache

export DATASET_TYPE="grain"
export GRAIN_TRAIN_FILES="/mnt/filestore/gemma_tpu_grain/ja_wiki_merged.arecord"
export GRAIN_EVAL_FILES="/mnt/filestore/gemma_tpu_grain/ja_wiki_merged.arecord"
export TOKENIZER_PATH="/mnt/filestore/tokenizers/gemma-2b/tokenizer.model" #modelファイルのパス
export BASE_OUTPUT_DIRECTORY="/mnt/filestore/checkpoints"  
export RUN_NAME="profiler"  
export MODEL_NAME="gemma2-2b"      
export CONVERTED_CHECKPOINT="/mnt/filestore/checkpoints_maxtext/gemma2-2b/0/items"


export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"

# はやくならない
 --xla_enable_async_collective_permute=true
 --xla_tpu_enable_async_collective_fusion=true
 --xla_tpu_overlap_compute_collective_tc=true


# 他の参考
# https://cloud.google.com/tpu/docs/v5p-training?hl=ja
# https://github.com/AI-Hypercomputer/maxtext/blob/9ca273807c4f40184e415386f865265167dd6e59/end_to_end/tpu/test_tflops_64b_params.sh#L30
# https://github.com/AI-Hypercomputer/maxtext/blob/9ca273807c4f40184e415386f865265167dd6e59/MaxText/configs/README.md

# /home/$USER/maxtext にパスがあると仮定しています
cd maxtext

# すでに実行中のプロセスがあれば終了させる
pkill -9 python

python3 MaxText/train.py MaxText/configs/base.yml \
    dataset_type=${DATASET_TYPE} \
    grain_train_files=${GRAIN_TRAIN_FILES} \
    grain_eval_files=${GRAIN_EVAL_FILES} \
    tokenizer_path=${TOKENIZER_PATH} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    run_name=${RUN_NAME} \
    model_name=${MODEL_NAME} \
    per_device_batch_size=2 \
    gradient_accumulation_steps=8 \
    grain_worker_count=8 \
    max_target_length=8192 \
    steps=10000 \
    checkpoint_period=10000 \
    enable_checkpointing=true \
    ici_tensor_parallelism=1 \
    dcn_tensor_parallelism=1 \
    dcn_data_parallelism=-1 \
    ici_data_parallelism=1 \
    ici_fsdp_parallelism=-1 \
    profiler="xplane" \
    upload_all_profiler_results=False \
    skip_first_n_steps_for_profiler=20 \
    profiler_steps=10 \
    profile_cleanly=true \
    profile_periodically_period=-1
