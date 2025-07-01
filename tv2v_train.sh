#!/bin/bash

export PYTHONPATH=/scratch/rs02358/ved_dissertation/CCEdit
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

python scripts/sampling/sampling_tv2v.py \
    --config_path configs/inference_ccedit/keyframe_no2ndca_depthmidas.yaml \
    --ckpt_path models/tv2v-no2ndca-depthmidas.ckpt \
    --H 192 --W 256 \
    --original_fps 8 --target_fps 6 \
    --num_keyframes 17 --batch_size 1 --num_samples 1 \
    --sample_steps 30 --sampler_name DPMPP2SAncestralSampler  --cfg_scale 7.5 \
    --prompt 'A person is playing golf.' \
    --video_path assets/ucf101/Golf/v_GolfSwing_g01_c01_8fps.mp4 \
    --add_prompt 'Van Gogh style' \
    --save_path outputs/tv2v/playinggolf-VanGogh \
    --disable_check_repeat