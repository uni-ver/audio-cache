export CUDA_VISIBLE_DEVICES=7
export CUDA_LAUNCH_BLOCKING=1 

torchrun \
    --nproc_per_node=1 \
    --master_port=29520 \
    run_longcat_echocache.py \
    --context_parallel_size=1 \
    --checkpoint_dir=./weights/LongCat-Video-Avatar \
    --stage_1=at2v \
    --input_json=assets/avatar/single_example_1.json \
    --teacache_thresh 0.05 \
    --use_ret_steps \
    2>&1 | tee ./log/run_avatar_echocache_$(date +%Y%m%d_%H%M%S).log