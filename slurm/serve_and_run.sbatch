#!/bin/bash
#SBATCH --job-name=bw-serve-run
#SBATCH --partition=aiscii
#SBATCH --account=aiscii
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# --gres, --time, --output are given on the sbatch command line (see below).
#
# One job = N identical model servers (one per GPU, all the same GPU type) + the driver that uses them. The job exits
# when the driver exits (its work queue is empty), so no GPU is held idle. Usage:
#
#   sbatch --gres=gpu:N --time=HH:MM:SS --output=runs/slurm/%x-%j.out slurm/serve_and_run.sbatch \
#          scripts/precheck.py --config configs/workflow_d5.yaml --out runs/precheck_d5
#
# The driver gets --set llm.api_base=<comma list of the N servers>; requests are routed by item (workflow.llm.RoutedLLM),
# so no request is generated twice. Server settings (identical on every server, user plan of 2026-11-27):
# Qwen3-14B bf16, --max-model-len 40960, --kv-cache-dtype fp8; per request: temperature 0.7, max_tokens 30000,
# thinking on (chat_template_kwargs.enable_thinking) - set in configs/workflow_d5.yaml.
# Env: uv venv, official vLLM 0.19.0 wheel + torch cu128 (aiscii: H200, driver 570 / CUDA 12.8, RHEL 9).
# Weights: local HF cache, offline. No credentials are read: models/env.sh is not sourced.
set -uo pipefail
REPO=/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/projects/boostrsi
E=${VLLM_ENV:-/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/envs/boostrsi-vllm-0.19.0}
export HF_HUB_CACHE=/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/models/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_CACHE_ROOT=/scratch/pioneer/users/yxs1432/boostrsi/vllm-cache
mkdir -p "$VLLM_CACHE_ROOT"
# torch inductor runs `nvcc --version` on PATH entries: drop missing / non-searchable directories (marsi notes)
export PATH=$E/bin:/usr/bin:/bin
MODEL=$HF_HUB_CACHE/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18
HOST=${SLURMD_NODENAME}
NGPU=$(nvidia-smi -L | wc -l)
BASE=$((20000 + (SLURM_JOB_ID % 2000) * 8))
OUT=$REPO/runs/slurm/job-$SLURM_JOB_ID
mkdir -p "$OUT"
cd "$REPO"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; }
trap cleanup EXIT
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | tee "$OUT/gpus.txt"
if [ "$(cut -d, -f2 "$OUT/gpus.txt" | sort -u | wc -l)" != 1 ]; then echo "mixed GPU types"; exit 2; fi

COMMON=(--served-model-name qwen3-14b --host 0.0.0.0 --dtype bfloat16 --max-model-len 40960 --kv-cache-dtype fp8
        --gpu-memory-utilization 0.90 --enable-prefix-caching --seed 0)
CONFIGS=("" "--enforce-eager")   # the same choice is applied to every server; recorded in server_config.txt
ok=0
for cfg in "${CONFIGS[@]}"; do
  PIDS=()
  echo "=== starting $NGPU servers, extra args: '${cfg}'"
  for i in $(seq 0 $((NGPU - 1))); do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$i vllm serve "$MODEL" "${COMMON[@]}" --port $((BASE + i)) $cfg > "$OUT/vllm-$i.log" 2>&1 &
    PIDS+=($!)
  done
  for t in $(seq 1 180); do                       # up to 30 min to load
    up=0; dead=0
    for i in $(seq 0 $((NGPU - 1))); do
      kill -0 "${PIDS[$i]}" 2>/dev/null || dead=1
      curl -sf "http://127.0.0.1:$((BASE + i))/health" >/dev/null && up=$((up + 1))
    done
    if [ "$dead" = 1 ]; then echo "=== a server exited during startup"; break; fi
    if [ "$up" = "$NGPU" ]; then ok=1; break; fi
    sleep 10
  done
  [ "$ok" = 1 ] && break
  cleanup
done
if [ "$ok" != 1 ]; then echo "=== no configuration started"; exit 1; fi
{ echo "gpu_type=$(cut -d, -f2 "$OUT/gpus.txt" | sort -u | xargs)"; echo "n_servers=$NGPU"; echo "node=$HOST";
  echo "vllm_args=${COMMON[*]} ${cfg}"; "$E/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"; } \
  | tee "$OUT/server_config.txt"
API=$(for i in $(seq 0 $((NGPU - 1))); do printf "http://127.0.0.1:%d/v1," $((BASE + i)); done); API=${API%,}
echo "=== READY $API"
start=$(date +%s)
"$E/bin/python" "$@" --set "llm.api_base=$API" 2>&1 | tee "$OUT/driver.log"
rc=${PIPESTATUS[0]}
echo "=== driver exited rc=$rc after $(( $(date +%s) - start ))s; stopping servers"
exit "$rc"
