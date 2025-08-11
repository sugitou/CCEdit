#!/bin/bash

export ACCELERATE_USE_DEEPSPEED=0
export TRITON_CACHE_DIR=/parallel_scratch/rs02358/.triton/autotune
export PYTHONPATH="/parallel_scratch/rs02358/CCEdit/src/taming-transformers:${PYTHONPATH}"

python scripts/sampling/sampling_tv2v.py \
    --config_path configs/inference_ccedit/keyframe_no2ndca_depthmidas.yaml \
    --ckpt_path models/tv2v-no2ndca-depthmidas.ckpt \
    --H 192 --W 256 \
    --original_fps 25 --target_fps 8 \
    --num_keyframes 120 --batch_size 1 --num_samples 1 \
    --sample_steps 100 --sampler_name DPMPP2SAncestralSampler  --cfg_scale 12 \
    --prompt 'Two focused fencers in white uniforms duel on a strip inside a large sports hall' \
    --video_path /parallel_scratch/rs02358/Reference_Videos/v_RockClimbingIndoor_g13_c03.mp4 \
    --add_prompt 'anime style' \
    --save_path outputs/tv2v/Fencing-Longvideo \
    --disable_check_repeat \
    --prior_coefficient_x 0.3 \
    --basemodel_path models/base/revAnimated_v2Rebirth.safetensors \
    # --lora_path models/lora/PixelArtRedmond15V-PixelArt-PIXARFK.safetensors \
    # --lora_strength 0.8 \
