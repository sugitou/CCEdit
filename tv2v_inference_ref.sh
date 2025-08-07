#!/bin/bash

# export PYTHONPATH=/scratch/rs02358/ved_dissertation/CCEdit
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

python scripts/sampling/sampling_tv2v_ref.py \
    --seed 201574 \
    --config_path configs/inference_ccedit/keyframe_ref_cp_no2ndca_add_cfca_depthzoe.yaml \
    --ckpt_path models/tv2v-no2ndca-depthmidas.ckpt \
    --H 32 --W 32 \
    --original_fps 8 --target_fps 6 --num_keyframes 17 --batch_size 1 --num_samples 1 \
    --sample_steps 50 --sampler_name DPMPP2SAncestralSampler --cfg_scale 7 \
    --prompt 'A person walks on the grass, the Milky Way is in the sky, night' \
    --add_prompt 'masterpiece, best quality,' \
    --video_path assets/Samples/tshirtman_8fps.mp4 \
    --reference_path assets/Samples/tshirtman-milkyway.png \
    --save_path outputs/tv2v/tshirtman-MilkyWay \
    --disable_check_repeat \
    --prior_coefficient_x 0.03 \
    --prior_type ref \