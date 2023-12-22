#!/bin/bash

torchrun --nproc_per_node=2 finetune.py \
    --model_path "allenai/longformer-base-4096" \
    --experiment_tag "longformer" \
    --max_length 1024 \
    --batch_size 8 \
    --gradient_accumulation 2
    