#!/bin/bash

export ACCELERATE_USE_DEEPSPEED=0
export TRITON_CACHE_DIR=/parallel_scratch/rs02358/.triton/autotune
export PYTHONPATH="/parallel_scratch/rs02358/CCEdit/src/taming-transformers:${PYTHONPATH}"

LOG_FILE="outputs/tv2v/experiment_$(date '+%Y%m%d_%H%M%S').log"

# 1. The information for each video
prompts=(
    #"A man wearing white tank top practices boxing, punching a red heavy bag in his garage home gym"
    #"A young girl with long blonde hair is brushing her teeth in a bathroom"
    #"Two focused fencers in white uniforms duel on a strip inside a large sports hall"
    #"A man is practising his golf swing on the green front lawn of a brick house"
    #"A rider on horseback navigates an obstacle course in a sandy arena with trees and hills"
    #"A skater with a red backpack is carving on a paved road next to a snowy mountain"
    "A climber ascends a challenging indoor climbing wall with a follow cinematic shot"
)
video_paths=(
    #"/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_BoxingPunchingBag_g01_c03.mp4"
    #"/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_BrushingTeeth_g01_c04.mp4"
    #"/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_Fencing_g01_c05.mp4"
    #"/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_GolfSwing_g02_c02_25fps.mp4"
    #"/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_HorseRiding_g01_c06_25fps.mp4"
    #"/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_SkateBoarding_g01_c03.mp4"
    "/parallel_scratch/rs02358/Reference_Videos/v_RockClimbingIndoor_g13_c03.mp4"
)
save_paths=(
    #"outputs/tv2v/BoxingPunchingBag"
    #"outputs/tv2v/BrushingTeeth"
    #"outputs/tv2v/Fencing"
    #"outputs/tv2v/GolfSwing"
    #"outputs/tv2v/HorseRiding"
    #"outputs/tv2v/SkateBoarding"
    "outputs/tv2v/RockClimbingIndoor"
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
