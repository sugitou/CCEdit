#!/bin/bash

#export ACCELERATE_USE_DEEPSPEED=0
#export TRITON_CACHE_DIR=/parallel_scratch/rs02358/.triton/autotune
#export PYTHONPATH="/parallel_scratch/rs02358/CCEdit/src/taming-transformers:${PYTHONPATH}"

LOG_FILE="outputs/tv2v_appearance/experiment_$(date '+%Y%m%d_%H%M%S').log"

# 1. The information for each video
prompts=(
    "A knight equipping armor practices boxing, punching a red heavy bag in his garage home gym"
    "A crystal skeleton is brushing its teeth in a bathroom"
    "Two focused women in kimonos duel on a strip inside a large sports hall"
    "A wizard is practising his golf swing on the green front lawn of a brick house"
    "A man riding a zebra navigates an obstacle course in a sandy arena with trees and hills"
    "An emperor penguin with a red backpack is carving on a paved road next to a snowy mountain"
    "A woman in a maid's costume ascends a challenging indoor climbing wall with a follow cinematic shot"
    "A small robot carefully applies eyeliner with a thin brush in her bathroom"
    "A muscular alien performs an impressive snatch, lifting a heavy barbell overhead in a gym"
    "A silver robot holds a steady handstand on a grassy hill under a vast blue sky"
)
video_paths=(
    "/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_BoxingPunchingBag_g01_c03.mp4"
    "/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_BrushingTeeth_g01_c04.mp4"
    "/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_Fencing_g01_c05.mp4"
    "/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_GolfSwing_g02_c02_25fps.mp4"
    "/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_HorseRiding_g01_c06_25fps.mp4"
    "/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_SkateBoarding_g01_c03.mp4"
    "/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_RockClimbingIndoor_g13_c03.mp4"
    "/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_ApplyEyeMakeup_g12_c05.mp4"
    "/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_CleanAndJerk_g25_c01.mp4"
    "/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_HandstandWalking_g14_c03.mp4"
)
save_paths=(
    "outputs/tv2v_appearance/BoxingPunchingBag"
    "outputs/tv2v_appearance/BrushingTeeth"
    "outputs/tv2v_appearance/Fencing"
    "outputs/tv2v_appearance/GolfSwing"
    "outputs/tv2v_appearance/HorseRiding"
    "outputs/tv2v_appearance/SkateBoarding"
    "outputs/tv2v_appearance/RockClimbingIndoor"
    "outputs/tv2v_appearance/ApplyEyeMakeup"
    "outputs/tv2v_appearance/CleanAndJerk"
    "outputs/tv2v_appearance/HandstandWalking"
)

# 2. add_prompt/basemodel pairs
#add_prompts=("anime style" "anime style" "mecha style" "pixel art style")
#basemodel_paths=("revAnimated_v2Rebirth.safetensors" "toonyou_alpha3.safetensors" "hellomecha_V12fvae.safetensors" "Counterfeit-V3.0.safetensors")

# 3. cfg_scale/prior_coefficient_x
cfg_scales=(5 9 12)
prior_coeffs=(0.3 0.1)

# 各動画ごと
for idx in "${!prompts[@]}"; do
  prompt="${prompts[$idx]}"
  video_path="${video_paths[$idx]}"
  save_path="${save_paths[$idx]}"

  # cfg_scale × prior_coefficient_x の全組み合わせ
  for cfg in "${cfg_scales[@]}"; do
    for prior in "${prior_coeffs[@]}"; do
      
      echo "==== Experiment Start ====" >> "$LOG_FILE"
      echo "datetime: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
      echo "prompt: $prompt" >> "$LOG_FILE"
      echo "video_path: $video_path" >> "$LOG_FILE"
      #echo "add_prompt: $add_prompt" >> "$LOG_FILE"
      #echo "basemodel_path: $basemodel_path" >> "$LOG_FILE"
      echo "cfg_scale: $cfg" >> "$LOG_FILE"
      echo "prior_coefficient_x: $prior" >> "$LOG_FILE"
      echo "save_path: $save_path" >> "$LOG_FILE"

      python scripts/sampling/sampling_tv2v.py \
        --config_path configs/inference_ccedit/keyframe_no2ndca_depthmidas.yaml \
        --ckpt_path models/tv2v-no2ndca-depthmidas.ckpt \
        --H 192 --W 256 \
        --original_fps 25 --target_fps 8 \
        --num_keyframes 24 --batch_size 1 --num_samples 1 \
        --sample_steps 100 --sampler_name DPMPP2SAncestralSampler  --cfg_scale "$cfg" \
        --prompt "$prompt" \
        --video_path "$video_path" \
        --save_path "$save_path" \
        --disable_check_repeat \
        --prior_coefficient_x "$prior" \
        --basemodel_path models/base/revAnimated_v2Rebirth.safetensors

      echo "==== Experiment End ====" >> "$LOG_FILE"
      echo "" >> "$LOG_FILE"

    done
  done
done
