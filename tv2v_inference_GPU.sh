#!/bin/bash
set -euo pipefail

# ====== 監視の設定 ======
INTERVAL=1  # 監視間隔(秒)
#OUTDIR="/scratch/rs02358/ved_dissertation/CCEdit/outputs/Lookout/Boxing_160f"
OUTDIR="/parallel_scratch/rs02358/CCEdit/outputs/Lookout/Boxing_80f"
mkdir -p "$OUTDIR"

GPU_CSV="$OUTDIR/gpu.csv"             # GPU全体: index, usedMiB, totalMiB, util%, memUtil%
GPROC_CSV="$OUTDIR/gpu_procs.csv"     # GPUプロセス: pid, name, usedMiB
RAM_CSV="$OUTDIR/ram.csv"             # ノードRAM: usedMiB, totalMiB
PRSS_CSV="$OUTDIR/proc_rss.csv"       # 対象プロセスのRSS: pid, rssMiB

echo "timestamp,index,mem_used_MiB,mem_total_MiB,util_gpu,util_mem" > "$GPU_CSV"
echo "timestamp,pid,process_name,used_mem_MiB" > "$GPROC_CSV"
echo "timestamp,ram_used_MiB,ram_total_MiB" > "$RAM_CSV"
echo "timestamp,pid,rss_MiB" > "$PRSS_CSV"

ts() { date "+%F %T"; }

gpu_monitor() {
  while :; do
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits \
        | awk -v t="$(ts)" -F, '{gsub(/^ +| +$/,""); printf "%s,%s,%s,%s,%s,%s\n", t,$1,$2,$3,$4,$5}' >> "$GPU_CSV"
      nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -v t="$(ts)" -F, '{gsub(/^ +| +$/,""); printf "%s,%s,%s,%s\n", t,$1,$2,$3}' >> "$GPROC_CSV" || true
    fi
    sleep "$INTERVAL"
  done
}

ram_monitor() {
  while :; do
    free -m | awk -v t="$(ts)" '/^Mem:/ {printf "%s,%s,%s\n", t,$3,$2}' >> "$RAM_CSV"
    sleep "$INTERVAL"
  done
}

proc_rss_monitor() {
  local pid="$1"
  while kill -0 "$pid" 2>/dev/null; do
    rss_kb=$(ps -o rss= -p "$pid" 2>/dev/null || echo 0)
    rss_mb=$(( (rss_kb+999)/1000 ))
    echo "$(ts),$pid,$rss_mb" >> "$PRSS_CSV"
    sleep "$INTERVAL"
  done
}

cleanup() {
  kill ${GPU_MON_PID:-0} ${RAM_MON_PID:-0} ${PRSS_MON_PID:-0} 2>/dev/null || true
}

trap cleanup EXIT INT TERM

# ====== あなたの元の環境変数 ======
export ACCELERATE_USE_DEEPSPEED=0
export TRITON_CACHE_DIR=/parallel_scratch/rs02358/.triton/autotune
export PYTHONPATH="/parallel_scratch/rs02358/CCEdit/src/taming-transformers:${PYTHONPATH:-}"

# ====== あなたの元の実行コマンド ======
python scripts/sampling/sampling_tv2v.py \
    --config_path configs/inference_ccedit/keyframe_no2ndca_depthmidas.yaml \
    --ckpt_path models/tv2v-no2ndca-depthmidas.ckpt \
    --H 192 --W 256 \
    --original_fps 25 --target_fps 8 \
    --num_keyframes 80 --batch_size 1 --num_samples 1 \
    --sample_steps 100 --sampler_name DPMPP2SAncestralSampler --cfg_scale 9 \
    --prompt 'A man wearing white tank top practices boxing, punching a red heavy bag in his garage home gym' \
    --video_path /parallel_scratch/rs02358/Reference_Videos/v_BoxingPunchingBag_g01_c03.mp4 \
    --add_prompt 'pixel art style' \
    --save_path outputs/tv2v/LongVideo/Boxing \
    --disable_check_repeat \
    --prior_coefficient_x 0.3 \
    --basemodel_path models/base/Counterfeit-V3.0.safetensors \
    --lora_path models/lora/PixelArtRedmond15V-PixelArt-PIXARFK.safetensors \
    --lora_strength 0.8 &

TARGET_PID=$!

# ====== 監視開始 ======
gpu_monitor &  GPU_MON_PID=$!
ram_monitor &  RAM_MON_PID=$!
proc_rss_monitor "$TARGET_PID" & PRSS_MON_PID=$!

# ====== ターゲット終了待ち ======
wait "$TARGET_PID"
EXIT_CODE=$?

# ====== サマリー ======
cleanup
awk -F, 'NR>1{if($3>m)m=$3} END{if(m!="") printf "GPU peak used: %.1f MiB\n", m+0}' "$GPU_CSV"
awk -F, 'NR>1{if($3>m)m=$3} END{if(m!="") printf "Process RSS peak: %.1f MiB\n", m+0}' "$PRSS_CSV"
awk -F, 'NR>1{if($2>m)m=$2; t=$3} END{if(m!="") printf "System RAM peak used: %.1f / %.1f MiB\n", m+0, t+0}' "$RAM_CSV"
exit "$EXIT_CODE"
