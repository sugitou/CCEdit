#!/bin/bash

# export PYTHONPATH=/scratch/rs02358/ved_dissertation/CCEdit
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

python scripts/sampling/sampling_tv2v.py \
    --config_path configs/inference_ccedit/keyframe_no2ndca_depthmidas.yaml \
    --ckpt_path models/tv2v-no2ndca-depthmidas.ckpt \
    --H 192 --W 256 \
    --original_fps 25 --target_fps 8 \
    --num_keyframes 24 --batch_size 1 --num_samples 1 \
    --sample_steps 100 --sampler_name DPMPP2SAncestralSampler  --cfg_scale 12 \
    --prompt 'Two focused fencers in white uniforms duel on a strip inside a large sports hall' \
    --video_path /scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_Fencing_g01_c05.mp4 \
    --add_prompt 'anime style' \
    --save_path outputs/tv2v/test \
    --disable_check_repeat \
    --prior_coefficient_x 0.3 \
    --basemodel_path models/base/revAnimated_v2Rebirth.safetensors \
    # --lora_path models/lora/PixelArtRedmond15V-PixelArt-PIXARFK.safetensors \
    # --lora_strength 0.8 \
