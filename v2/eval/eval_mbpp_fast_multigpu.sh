#!/usr/bin/env bash

# eval_mbpp_fast_multigpu.sh
#
# 使用方法：
# bash eval_mbpp_fast_multigpu.sh [METHOD] [THRESHOLD] [USE_BLOCK_CACHE] [REGENERATE] [NUM_GPUS] [LIMIT]
#
# 示例：
# bash eval_mbpp_fast_multigpu.sh Fast
# bash eval_mbpp_fast_multigpu.sh Fast 0.9 true false
# bash eval_mbpp_fast_multigpu.sh Fast 0.9 false false
# bash eval_mbpp_fast_multigpu.sh Fast 0.9 true true
# bash eval_mbpp_fast_multigpu.sh Qwen3 0.9 true false
# bash eval_mbpp_fast_multigpu.sh Qwen3 0.9 true false 4
# bash eval_mbpp_fast_multigpu.sh Fast 0.9 true true 2 20
# bash eval_mbpp_fast_multigpu.sh Qwen2.5_LoRA
# 
# bash eval_mbpp_fast_multigpu.sh Qwen3_Ori
# bash eval_mbpp_fast_multigpu.sh Qwen3_Large
# bash eval_mbpp_fast_multigpu.sh Qwen3_Large 0.9 false
#
# 参数说明：
# METHOD: Fast / Qwen2.5 / Qwen2.5_LoRA / Qwen3
# THRESHOLD: 默认 0.9
# USE_BLOCK_CACHE: true / false，默认 true
# REGENERATE: true / false，默认 false
# NUM_GPUS: 默认使用当前可见的全部 GPU
# LIMIT: 可选，只评测前 N 个 MBPP 任务
#
# 也可以用 CUDA_VISIBLE_DEVICES 限制 GPU：
# CUDA_VISIBLE_DEVICES=0,2 bash eval_mbpp_fast_multigpu.sh Fast 0.9 true false

set -euo pipefail

cd /home/u-shengbf/Codes/Fast-dLLM/v2/eval

normalize_bool() {
  local value="${1,,}"

  case "$value" in
    true|1|yes|y|on)
      echo "true"
      ;;
    false|0|no|n|off)
      echo "false"
      ;;
    *)
      echo "Invalid boolean value: $1" >&2
      echo "Use one of: true/false, 1/0, yes/no, y/n, on/off" >&2
      exit 1
      ;;
  esac
}

discover_gpus() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' '
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' '
    return
  fi

  python - <<'PY'
import torch
print(" ".join(str(i) for i in range(torch.cuda.device_count())))
PY
}

print_gpu_info() {
  echo "当前 GPU 配置："
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu \
      --format=csv,noheader
  else
    python - <<'PY'
import torch
print(f"torch.cuda.is_available={torch.cuda.is_available()}")
print(f"torch.cuda.device_count={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"{i}: {torch.cuda.get_device_name(i)}")
PY
  fi
}

METHOD="${1:-Fast}"
THRESHOLD="${2:-0.9}"
USE_BLOCK_CACHE="$(normalize_bool "${3:-true}")"
REGENERATE="$(normalize_bool "${4:-false}")"
REQUESTED_NUM_GPUS="${5:-}"
LIMIT="${6:-}"

DATASET="mbpp"

case "$METHOD" in
  Fast)
    MODEL_PATH="Efficient-Large-Model/Fast_dLLM_v2_7B"
    ;;

  Qwen2.5)
    MODEL_PATH="/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_fast_dLLM_7B_20260521_222723"
    ;;

  Qwen2.5_LoRA)
    MODEL_PATH="/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_fast_dLLM_7B_merged_20260524_172813"
    ;;

  Qwen3_Ori)
    MODEL_PATH="/home/u-shengbf/Codes/Fast-dLLM/v2/base_models/Model-Qwen-3-8B"
    ;;

  Qwen3)
    MODEL_PATH="/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_full_20260525_234355"
    ;;

  Qwen3_Large)
    MODEL_PATH="/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_full_20260527_235410"
    ;;

  *)
    echo "Unknown METHOD: ${METHOD}" >&2
    echo "Supported METHOD values: Fast, Qwen2.5, Qwen2.5_LoRA, Qwen3_Ori, Qwen3, Qwen3_Large" >&2
    exit 1
    ;;
esac

# mask_id 逻辑：
# Qwen3 使用 151669，其余模型使用 151665。
if [ "$METHOD" = "Qwen3" ]; then
  MASK_ID=151669
else
  MASK_ID=151665
fi

print_gpu_info

read -r -a ALL_GPUS <<< "$(discover_gpus)"

if [ "${#ALL_GPUS[@]}" -eq 0 ]; then
  echo "No available GPU detected." >&2
  exit 1
fi

if [ -n "$REQUESTED_NUM_GPUS" ]; then
  if ! [[ "$REQUESTED_NUM_GPUS" =~ ^[0-9]+$ ]]; then
    echo "NUM_GPUS must be an integer, got $REQUESTED_NUM_GPUS" >&2
    exit 1
  fi

  if [ "$REQUESTED_NUM_GPUS" -lt 1 ]; then
    echo "NUM_GPUS must be >= 1, got $REQUESTED_NUM_GPUS" >&2
    exit 1
  fi

  if [ "$REQUESTED_NUM_GPUS" -gt "${#ALL_GPUS[@]}" ]; then
    echo "Requested NUM_GPUS=$REQUESTED_NUM_GPUS, but only ${#ALL_GPUS[@]} GPUs are visible." >&2
    exit 1
  fi

  NUM_GPUS="$REQUESTED_NUM_GPUS"
else
  NUM_GPUS="${#ALL_GPUS[@]}"
fi

GPUS=("${ALL_GPUS[@]:0:$NUM_GPUS}")
NUM_SHARDS="$NUM_GPUS"

mkdir -p "../evalplus_results/${METHOD}"

LIMIT_SUFFIX=""
if [ -n "$LIMIT" ]; then
  LIMIT_SUFFIX="_limit${LIMIT}"
fi

BASE_OUTPUT="../evalplus_results/${METHOD}/mbpp_fast_multigpu_th${THRESHOLD}_cache${USE_BLOCK_CACHE}_mask${MASK_ID}_gpus${NUM_GPUS}${LIMIT_SUFFIX}.jsonl"
OUTPUT="$BASE_OUTPUT"

# 如果已有文件且选择重新生成，则最终输出文件名加时间戳。
if [ -f "$OUTPUT" ] && [ "$REGENERATE" = "true" ]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  OUTPUT="${BASE_OUTPUT%.jsonl}_${TIMESTAMP}.jsonl"
  echo "检测到已有文件，且 REGENERATE=true，本次输出改为: $OUTPUT"
fi

SHARD_DIR="${OUTPUT%.jsonl}_shards"
LOG_DIR="${OUTPUT%.jsonl}_logs"
EVAL_LOG="${OUTPUT%.jsonl}_evalplus.log"
METRICS_JSON="${OUTPUT%.jsonl}_metrics_report.json"

SHARD_OUTPUTS=()
PROGRESS_FILES=()
LOG_FILES=()

for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
  GPU_ID="${GPUS[$SHARD_ID]}"
  SHARD_OUTPUT="${SHARD_DIR}/shard_${SHARD_ID}_of_${NUM_SHARDS}.jsonl"
  PROGRESS_FILE="${LOG_DIR}/shard_${SHARD_ID}_gpu_${GPU_ID}.progress.json"
  LOG_FILE="${LOG_DIR}/shard_${SHARD_ID}_gpu_${GPU_ID}.log"

  SHARD_OUTPUTS+=("$SHARD_OUTPUT")
  PROGRESS_FILES+=("$PROGRESS_FILE")
  LOG_FILES+=("$LOG_FILE")
done

printf '%*s\n' 80 '' | tr ' ' '='
echo "METHOD=${METHOD}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATASET=${DATASET}"
echo "THRESHOLD=${THRESHOLD}"
echo "USE_BLOCK_CACHE=${USE_BLOCK_CACHE}"
echo "REGENERATE=${REGENERATE}"
echo "MASK_ID=${MASK_ID}"
echo "VISIBLE_GPUS=${ALL_GPUS[*]}"
echo "USED_GPUS=${GPUS[*]}"
echo "NUM_SHARDS=${NUM_SHARDS}"
echo "LIMIT=${LIMIT:-none}"
echo "OUTPUT=${OUTPUT}"
echo "SHARD_DIR=${SHARD_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "EVAL_LOG=${EVAL_LOG}"
echo "METRICS_JSON=${METRICS_JSON}"
printf '%*s\n' 80 '' | tr ' ' '='

if [ -f "$OUTPUT" ] && [ "$REGENERATE" = "false" ]; then
  echo "已有合并输出文件，且 REGENERATE=false，跳过生成: $OUTPUT"
else
  echo "开始多 GPU 分片生成 MBPP samples"

  rm -rf "$SHARD_DIR" "$LOG_DIR"
  mkdir -p "$SHARD_DIR" "$LOG_DIR"

  LIMIT_ARGS=()
  if [ -n "$LIMIT" ]; then
    LIMIT_ARGS=(--limit "$LIMIT")
  fi

  PIDS=()

  for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    GPU_ID="${GPUS[$SHARD_ID]}"
    SHARD_OUTPUT="${SHARD_OUTPUTS[$SHARD_ID]}"
    PROGRESS_FILE="${PROGRESS_FILES[$SHARD_ID]}"
    LOG_FILE="${LOG_FILES[$SHARD_ID]}"

    echo "启动 shard ${SHARD_ID}/${NUM_SHARDS} on GPU ${GPU_ID}"
    echo "  output: ${SHARD_OUTPUT}"
    echo "  progress: ${PROGRESS_FILE}"
    echo "  log: ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="$GPU_ID" python generate_mbpp_fast_samples.py \
      --model_path "$MODEL_PATH" \
      --output "$SHARD_OUTPUT" \
      --batch_size 1 \
      --max_new_tokens 512 \
      --mask_id "$MASK_ID" \
      --bd_size 32 \
      --small_block_size 8 \
      --threshold "$THRESHOLD" \
      --use_block_cache "$USE_BLOCK_CACHE" \
      --dtype bf16 \
      --shard_id "$SHARD_ID" \
      --num_shards "$NUM_SHARDS" \
      --progress_file "$PROGRESS_FILE" \
      "${LIMIT_ARGS[@]}" \
      > "$LOG_FILE" 2>&1 &

    PIDS+=("$!")
  done

  echo "启动主进程总进度条"
  python monitor_mbpp_progress.py \
    --progress_files "${PROGRESS_FILES[@]}" \
    --refresh 2 \
    &
  MONITOR_PID="$!"

  FAILED=0
  FAILED_SHARDS=()

  for IDX in "${!PIDS[@]}"; do
    PID="${PIDS[$IDX]}"
    SHARD_ID="$IDX"

    if wait "$PID"; then
      :
    else
      FAILED=1
      FAILED_SHARDS+=("$SHARD_ID")
    fi
  done

  # 给 monitor 一点时间读取 completed/failed 状态并刷新最后一帧。
  sleep 2
  if kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
  fi
  wait "$MONITOR_PID" 2>/dev/null || true

  if [ "$FAILED" -ne 0 ]; then
    echo "At least one shard failed. Failed shards: ${FAILED_SHARDS[*]}" >&2
    echo "查看日志：" >&2
    for SHARD_ID in "${FAILED_SHARDS[@]}"; do
      echo "  ${LOG_FILES[$SHARD_ID]}" >&2
    done
    exit 1
  fi

  echo "所有 shard 生成完成，开始合并 JSONL"

  MERGE_LIMIT_ARGS=()
  if [ -n "$LIMIT" ]; then
    MERGE_LIMIT_ARGS=(--limit "$LIMIT")
  fi

  python merge_mbpp_shards.py \
    --output "$OUTPUT" \
    --shards "${SHARD_OUTPUTS[@]}" \
    "${MERGE_LIMIT_ARGS[@]}" \
    --strict
fi

echo "开始 syncheck: $OUTPUT"
evalplus.syncheck \
  --dataset "$DATASET" \
  --samples "$OUTPUT"

echo "删除当前输出对应的旧 EvalPlus 结果缓存"
rm -f "${OUTPUT%.jsonl}"*eval_results*.jsonl
rm -f "$EVAL_LOG"

echo "开始 evaluate: $OUTPUT"
# 保留 EvalPlus 原始输出，同时用于最终报告解析 pass@1。
evalplus.evaluate \
  --dataset "$DATASET" \
  --samples "$OUTPUT" \
  2>&1 | tee "$EVAL_LOG"

echo "生成最终性能/正确率报告"
python report_mbpp_multigpu_metrics.py \
  --progress_files "${PROGRESS_FILES[@]}" \
  --eval_log "$EVAL_LOG" \
  --samples "$OUTPUT" \
  --output_json "$METRICS_JSON"

echo "MBPP 多 GPU 评测完成"
echo "Merged samples: $OUTPUT"
echo "EvalPlus log: $EVAL_LOG"
echo "Metrics report: $METRICS_JSON"
