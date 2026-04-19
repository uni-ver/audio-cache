#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
export CUDA_LAUNCH_BLOCKING=1 
torchrun \
    --nproc_per_node=1 \
    run_demo_avatar_single_audio_to_video.py \
    --context_parallel_size=1 \
    --checkpoint_dir=./weights/LongCat-Video-Avatar \
    --stage_1=at2v \
    --input_json=assets/avatar/single_example_1.json \
    2>&1 | tee ./log/run_avatar_$(date +%Y%m%d_%H%M%S).log