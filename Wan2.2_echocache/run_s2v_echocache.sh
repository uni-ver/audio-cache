#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
# CUDA_LAUNCH_BLOCKING=1 
python Wan-s2v-echocache.py \
    --task s2v-14B \
    --size 1024*704 \
    --ckpt_dir ./Wan2.2-S2V-14B/ \
    --offload_model True \
    --convert_model_dtype \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  \
    --image "examples/i2v_input.JPG" \
    --audio "examples/talk.wav"\
    --teacache_thresh 0.2 \
    --quantize_ratio 0.9\
    --selected_ratio 0.15\
    --enable_teacache False\
    --need_ffn True
    2>&1 | tee ./log/run_Wan_s2v_$(date +%Y%m%d_%H%M%S).log