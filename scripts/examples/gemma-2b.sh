#!/bin/bash

# hfのcache場所

export HF_HOME=/mnt/filestore/.cache
export HUGGINGFACE_HUB_CACHE=/mnt/filestore/.cache

export DATASET_TYPE="hf"
export HF_PATH="arrow"
export HF_TRAIN_FILES="/mnt/filestore/datasets/*/train/*.arrow"
export HF_EVAL_FILES="/mnt/filestore/datasets/*train/*.arrow"
export TOKENIZER_PATH="/mnt/filestore/tokenizers/gemma-2b"        
export BASE_OUTPUT_DIRECTORY="/mnt/filestore/checkpoints"  
export RUN_NAME="hf_streaming_test_$(date +%Y-%m-%d-%H-%M)"  
export MODEL_NAME="gemma2-2b"      
export CONVERTED_CHECKPOINT="/mnt/filestore/checkpoints_maxtext/gemma2-2b/0/items"
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"

# /home/$USER/maxtext にパスがあると仮定しています
cd maxtext

python3 MaxText/train.py MaxText/configs/base.yml \
    dataset_type=${DATASET_TYPE} \
    hf_path=${HF_PATH} \
    hf_train_files=${HF_TRAIN_FILES} \
    hf_eval_files=${HF_EVAL_FILES} \
    tokenizer_path=${TOKENIZER_PATH} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    run_name=${RUN_NAME} \
    model_name=${MODEL_NAME} \
    per_device_batch_size=2 \
    gradient_accumulation_steps=8 \
    max_target_length=8192 \
    steps=10000 \
    checkpoint_period=10000 \
    enable_checkpointing=true \
    ici_tensor_parallelism=1 \
    dcn_tensor_parallelism=1 \
    dcn_data_parallelism=-1 \
    ici_data_parallelism=1 \
    ici_fsdp_parallelism=-1 
