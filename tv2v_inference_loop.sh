#!/bin/bash

LOG_FILE="outputs/tv2v/experiment_$(date '+%Y%m%d_%H%M%S').log"

# 1. The information for each video
prompts=(
    "An athletic person holds a steady handstand on a grassy hill under a vast blue sky"
    "A strong man performs an impressive snatch, lifting a heavy barbell overhead in a gym"
)
video_paths=(
    "/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_HandstandWalking_g14_c03.mp4"
    "/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_CleanAndJerk_g25_c01.mp4"
)
save_paths=(
    "outputs/tv2v/HandstandWalking"
    "outputs/tv2v/CleanAndJerk"
)

# 2. add_prompt/basemodel pairs
add_prompts=("anime style" "anime style" "mecha style" "pixel art style")
basemodel_paths=("revAnimated_v2Rebirth.safetensors" "toonyou_alpha3.safetensors" "hellomecha_V12fvae.safetensors" "Counterfeit-V3.0.safetensors")

# 3. cfg_scale/prior_coefficient_x
cfg_scales=(5 9 12)
prior_coeffs=(0.3 0.1)

# 各動画ごと
for idx in "${!prompts[@]}"; do
  prompt="${prompts[$idx]}"
  video_path="${video_paths[$idx]}"
  save_path="${save_paths[$idx]}"

  # add_prompt/basemodelペアごと
  for style_idx in "${!add_prompts[@]}"; do
    add_prompt="${add_prompts[$style_idx]}"
    basemodel_path="${basemodel_paths[$style_idx]}"

    # cfg_scale × prior_coefficient_x の全組み合わせ
    for cfg in "${cfg_scales[@]}"; do
      for prior in "${prior_coeffs[@]}"; do
        
        echo "==== Experiment Start ====" >> "$LOG_FILE"
        echo "datetime: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
        echo "prompt: $prompt" >> "$LOG_FILE"
        echo "video_path: $video_path" >> "$LOG_FILE"
        echo "add_prompt: $add_prompt" >> "$LOG_FILE"
        echo "basemodel_path: $basemodel_path" >> "$LOG_FILE"
        echo "cfg_scale: $cfg" >> "$LOG_FILE"
        echo "prior_coefficient_x: $prior" >> "$LOG_FILE"
        echo "save_path: $save_path" >> "$LOG_FILE"

        # LoRAが必要な場合のみ分岐
        if [[ "$add_prompt" == "pixel art style" && "$basemodel_path" == "Counterfeit-V3.0.safetensors" ]]; then
          python scripts/sampling/sampling_tv2v.py \
            --config_path configs/inference_ccedit/keyframe_no2ndca_depthmidas.yaml \
            --ckpt_path models/tv2v-no2ndca-depthmidas.ckpt \
            --H 192 --W 256 \
            --original_fps 25 --target_fps 8 \
            --num_keyframes 24 --batch_size 1 --num_samples 1 \
            --sample_steps 100 --sampler_name DPMPP2SAncestralSampler  --cfg_scale "$cfg" \
            --prompt "$prompt" \
            --video_path "$video_path" \
            --add_prompt "$add_prompt" \
            --save_path "$save_path" \
            --disable_check_repeat \
            --prior_coefficient_x "$prior" \
            --basemodel_path models/base/"$basemodel_path" \
            --lora_path models/lora/PixelArtRedmond15V-PixelArt-PIXARFK.safetensors \
            --lora_strength 0.8
          echo "lora: ON" >> "$LOG_FILE"
        
        else
          python scripts/sampling/sampling_tv2v.py \
            --config_path configs/inference_ccedit/keyframe_no2ndca_depthmidas.yaml \
            --ckpt_path models/tv2v-no2ndca-depthmidas.ckpt \
            --H 192 --W 256 \
            --original_fps 25 --target_fps 8 \
            --num_keyframes 24 --batch_size 1 --num_samples 1 \
            --sample_steps 100 --sampler_name DPMPP2SAncestralSampler  --cfg_scale "$cfg" \
            --prompt "$prompt" \
            --video_path "$video_path" \
            --add_prompt "$add_prompt" \
            --save_path "$save_path" \
            --disable_check_repeat \
            --prior_coefficient_x "$prior" \
            --basemodel_path models/base/"$basemodel_path"
          echo "lora: OFF" >> "$LOG_FILE"
          
        fi

        echo "==== Experiment End ====" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"

      done
    done
  done
done
