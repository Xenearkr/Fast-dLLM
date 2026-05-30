#!/usr/bin/env bash

# 使用方法：
# bash eval_humaneval_fast_multigpu.sh [METHOD] [THRESHOLD] [USE_BLOCK_CACHE] [REGENERATE] [NUM_GPUS] [LIMIT]
#
# 示例：
# bash eval_humaneval_fast_multigpu.sh Fast
# bash eval_humaneval_fast_multigpu.sh Fast 0.9 true false
# bash eval_humaneval_fast_multigpu.sh Fast 0.9 false false
# bash eval_humaneval_fast_multigpu.sh Fast 0.9 true true
# bash eval_humaneval_fast_multigpu.sh Qwen3 0.9 true false
# bash eval_humaneval_fast_multigpu.sh Qwen3 0.9 true false 4
# bash eval_humaneval_fast_multigpu.sh Fast 0.9 true true 2 20
# 
# bash eval_humaneval_fast_multigpu.sh Qwen3_Ori
# bash eval_humaneval_fast_multigpu.sh Qwen3_Large
# 
# 参数说明：
# METHOD: Fast / Qwen2.5 / Qwen2.5_LoRA / Qwen3
# THRESHOLD: 默认 0.9
# USE_BLOCK_CACHE: true / false，默认 true
# REGENERATE: true / false，默认 false
# NUM_GPUS: 默认使用当前可见的全部 GPU
# LIMIT: 可选，只评测前 N 个 HumanEval 任务
#
# 可以用 CUDA_VISIBLE_DEVICES 限制 GPU：
# CUDA_VISIBLE_DEVICES=0,2 bash eval_humaneval_fast_multigpu.sh Fast 0.9 true false

set -euo pipefail

cd /home/u-shengbf/Codes/Fast-dLLM/v2/eval
mkdir -p ../evalplus_results

# -----------------------------
# Global state for cleanup
# -----------------------------
PIDS=()
MONITOR_PID=""
CLEANED_UP=0

stop_monitor() {
  if [ -n "${MONITOR_PID:-}" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    echo
    echo "Stopping HumanEval progress monitor: PID=${MONITOR_PID}"
    kill "$MONITOR_PID" 2>/dev/null || true

    # Give tqdm/monitor a short chance to exit cleanly.
    for _ in $(seq 1 10); do
      if ! kill -0 "$MONITOR_PID" 2>/dev/null; then
        break
      fi
      sleep 0.2
    done

    if kill -0 "$MONITOR_PID" 2>/dev/null; then
      echo "Progress monitor did not exit after SIGTERM; sending SIGKILL: PID=${MONITOR_PID}" >&2
      kill -9 "$MONITOR_PID" 2>/dev/null || true
    fi

    wait "$MONITOR_PID" 2>/dev/null || true
  fi

  MONITOR_PID=""
}

kill_running_shards() {
  local alive=()

  for pid in "${PIDS[@]:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    fi
  done

  if [ "${#alive[@]}" -eq 0 ]; then
    return 0
  fi

  echo "Stopping remaining HumanEval shard processes: ${alive[*]}" >&2
  kill "${alive[@]}" 2>/dev/null || true

  sleep 1

  local still_alive=()
  for pid in "${alive[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      still_alive+=("$pid")
    fi
  done

  if [ "${#still_alive[@]}" -gt 0 ]; then
    echo "Some shard processes did not exit after SIGTERM; sending SIGKILL: ${still_alive[*]}" >&2
    kill -9 "${still_alive[@]}" 2>/dev/null || true
  fi

  for pid in "${alive[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}

cleanup() {
  local exit_code="${1:-$?}"

  if [ "$CLEANED_UP" -eq 1 ]; then
    exit "$exit_code"
  fi
  CLEANED_UP=1

  # Avoid recursive trap execution.
  trap - EXIT INT TERM

  stop_monitor

  # Only kill shard workers on abnormal exit. On successful exit, they have
  # already finished and been waited on.
  if [ "$exit_code" -ne 0 ]; then
    kill_running_shards
  fi

  exit "$exit_code"
}

trap 'cleanup $?' EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

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

show_shard_failure_summary() {
  local log_dir="$1"
  local num_shards="$2"
  local -n gpu_array_ref="$3"

  echo
  echo "Shard failure summary:"
  for shard_id in $(seq 0 $((num_shards - 1))); do
    local gpu_id="${gpu_array_ref[$shard_id]}"
    local log_file="${log_dir}/shard_${shard_id}_gpu_${gpu_id}.log"

    echo "--------------------------------------------------------------------------------"
    echo "shard=${shard_id}, gpu=${gpu_id}, log=${log_file}"

    if [ -f "$log_file" ]; then
      grep -HnE "Traceback|Error|error|Exception|NameError|ImportError|ModuleNotFoundError|unrecognized arguments|CUDA|RuntimeError|OutOfMemory|OOM" \
        "$log_file" | tail -n 20 || true
      echo "---- tail -n 40 ----"
      tail -n 40 "$log_file" || true
    else
      echo "Log file not found."
    fi
  done
  echo "--------------------------------------------------------------------------------"
}

METHOD="${1:-Fast}"
THRESHOLD="${2:-0.9}"
USE_BLOCK_CACHE="$(normalize_bool "${3:-true}")"
REGENERATE="$(normalize_bool "${4:-false}")"
REQUESTED_NUM_GPUS="${5:-}"
LIMIT="${6:-}"

DATASET="humaneval"

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

# mask_id 逻辑：Qwen3 使用 151669，其余模型使用 151665。
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

BASE_OUTPUT="../evalplus_results/${METHOD}/humaneval_fast_multigpu_th${THRESHOLD}_cache${USE_BLOCK_CACHE}_mask${MASK_ID}_gpus${NUM_GPUS}${LIMIT_SUFFIX}.jsonl"
OUTPUT="$BASE_OUTPUT"

# 如果已有合并文件且选择重新生成，则最终输出文件名加时间戳，避免覆盖旧结果。
if [ -f "$OUTPUT" ] && [ "$REGENERATE" = "true" ]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  OUTPUT="${BASE_OUTPUT%.jsonl}_${TIMESTAMP}.jsonl"
  echo "检测到已有文件，且 REGENERATE=true，本次输出改为: $OUTPUT"
fi

SANITIZED_OUTPUT="${OUTPUT%.jsonl}-sanitized.jsonl"
PATCHED_OUTPUT="${OUTPUT%.jsonl}-patched.jsonl"
SHARD_DIR="${OUTPUT%.jsonl}_shards"
LOG_DIR="${OUTPUT%.jsonl}_logs"
EVAL_LOG="${OUTPUT%.jsonl}_evalplus.log"
REPORT_JSON="${OUTPUT%.jsonl}_metrics_report.json"

printf '=%.0s' {1..80}; echo
cat <<EOF
METHOD=${METHOD}
MODEL_PATH=${MODEL_PATH}
DATASET=${DATASET}
THRESHOLD=${THRESHOLD}
USE_BLOCK_CACHE=${USE_BLOCK_CACHE}
REGENERATE=${REGENERATE}
MASK_ID=${MASK_ID}
VISIBLE_GPUS=${ALL_GPUS[*]}
USED_GPUS=${GPUS[*]}
NUM_SHARDS=${NUM_SHARDS}
LIMIT=${LIMIT:-none}
OUTPUT=${OUTPUT}
SANITIZED_OUTPUT=${SANITIZED_OUTPUT}
PATCHED_OUTPUT=${PATCHED_OUTPUT}
SHARD_DIR=${SHARD_DIR}
LOG_DIR=${LOG_DIR}
EVAL_LOG=${EVAL_LOG}
REPORT_JSON=${REPORT_JSON}
EOF
printf '=%.0s' {1..80}; echo

PROGRESS_FILES=()
SHARD_OUTPUTS=()
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
  GPU_ID="${GPUS[$SHARD_ID]}"
  SHARD_OUTPUT="${SHARD_DIR}/shard_${SHARD_ID}_of_${NUM_SHARDS}.jsonl"
  PROGRESS_FILE="${LOG_DIR}/shard_${SHARD_ID}_gpu_${GPU_ID}.progress.json"
  SHARD_OUTPUTS+=("$SHARD_OUTPUT")
  PROGRESS_FILES+=("$PROGRESS_FILE")
done

if [ -f "$OUTPUT" ] && [ "$REGENERATE" = "false" ]; then
  echo "已有合并输出文件，且 REGENERATE=false，跳过生成: $OUTPUT"
  echo "注意：跳过生成时不会有新的 progress metrics；最终报告只能解析 EvalPlus 正确率。"
else
  echo "开始多 GPU 分片生成 HumanEval samples"

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
    LOG_FILE="${LOG_DIR}/shard_${SHARD_ID}_gpu_${GPU_ID}.log"

    echo "启动 shard ${SHARD_ID}/${NUM_SHARDS} on GPU ${GPU_ID}"
    echo "  output: ${SHARD_OUTPUT}"
    echo "  progress: ${PROGRESS_FILE}"
    echo "  log: ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="$GPU_ID" python generate_humaneval_fast_samples.py \
      --model_path "$MODEL_PATH" \
      --dataset "$DATASET" \
      --output "$SHARD_OUTPUT" \
      --prompt_mode raw \
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

  python monitor_humaneval_progress.py \
    --progress_files "${PROGRESS_FILES[@]}" \
    --refresh 2.0 &
  MONITOR_PID="$!"

  FAILED=0
  DONE=0
  TOTAL="${#PIDS[@]}"

  for IDX in "${!PIDS[@]}"; do
    PID="${PIDS[$IDX]}"
    SHARD_ID="$IDX"

    if wait "$PID"; then
      DONE=$((DONE + 1))
      echo "[${DONE}/${TOTAL}] Shard ${SHARD_ID} finished successfully."
    else
      DONE=$((DONE + 1))
      echo "[${DONE}/${TOTAL}] Shard ${SHARD_ID} failed. See log: ${LOG_DIR}/shard_${SHARD_ID}_gpu_${GPUS[$SHARD_ID]}.log" >&2
      FAILED=1
    fi
  done

  # monitor_humaneval_progress.py is a polling process. It will not necessarily
  # exit by itself, especially when all shards fail before writing progress files.
  # Stop it explicitly before merge/evaluation or failure exit.
  stop_monitor

  if [ "$FAILED" -ne 0 ]; then
    show_shard_failure_summary "$LOG_DIR" "$NUM_SHARDS" GPUS
    echo "At least one shard failed. Abort merge/evaluation." >&2
    exit 1
  fi

  echo "所有 shard 生成完成，开始合并 JSONL"

  MERGE_LIMIT_ARGS=()
  if [ -n "$LIMIT" ]; then
    MERGE_LIMIT_ARGS=(--limit "$LIMIT")
  fi

  python merge_humaneval_shards.py \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --shards "${SHARD_OUTPUTS[@]}" \
    "${MERGE_LIMIT_ARGS[@]}" \
    --strict
fi

printf '=%.0s' {1..80}; echo
echo "开始 EvalPlus 语法检查: $OUTPUT"
evalplus.syncheck \
  --dataset "$DATASET" \
  --samples "$OUTPUT"

if [ -f "$SANITIZED_OUTPUT" ]; then
  echo "已有 sanitized 文件，跳过 sanitize: $SANITIZED_OUTPUT"
else
  echo "开始 EvalPlus sanitize: $OUTPUT"
  evalplus.sanitize \
    --samples "$OUTPUT"
fi

echo "开始 sanitized 后语法检查: $SANITIZED_OUTPUT"
evalplus.syncheck \
  --dataset "$DATASET" \
  --samples "$SANITIZED_OUTPUT"

if [ -f "$PATCHED_OUTPUT" ]; then
  echo "已有 patched 文件，跳过 patch: $PATCHED_OUTPUT"
else
  echo "开始对比 raw/sanitized 并生成 patched 文件: $PATCHED_OUTPUT"
  python patch_humaneval_sanitized.py \
    --raw "$OUTPUT" \
    --sanitized "$SANITIZED_OUTPUT" \
    --output "$PATCHED_OUTPUT"
fi

echo "开始 patched inspect: $PATCHED_OUTPUT"
python inspect_humaneval_samples.py \
  --samples "$PATCHED_OUTPUT" \
  --show_n 5 \
  --show_bad_n 10

echo "开始 patched 语法检查: $PATCHED_OUTPUT"
evalplus.syncheck \
  --dataset "$DATASET" \
  --samples "$PATCHED_OUTPUT"

echo "删除当前输出对应的旧 EvalPlus 结果缓存"
rm -f "${PATCHED_OUTPUT%.jsonl}"*eval_results*.jsonl

echo "开始 EvalPlus evaluate: $PATCHED_OUTPUT"
set +e
evalplus.evaluate \
  --dataset "$DATASET" \
  --samples "$PATCHED_OUTPUT" 2>&1 | tee "$EVAL_LOG"
EVAL_STATUS=${PIPESTATUS[0]}
set -e

if [ "$EVAL_STATUS" -ne 0 ]; then
  echo "EvalPlus evaluate failed with status ${EVAL_STATUS}. See log: $EVAL_LOG" >&2
  exit "$EVAL_STATUS"
fi

# 如果本次是跳过生成，progress files 可能不存在；只有存在时才汇总吞吐。
EXISTING_PROGRESS_FILES=()
for PROGRESS_FILE in "${PROGRESS_FILES[@]}"; do
  if [ -f "$PROGRESS_FILE" ]; then
    EXISTING_PROGRESS_FILES+=("$PROGRESS_FILE")
  fi
done

if [ "${#EXISTING_PROGRESS_FILES[@]}" -gt 0 ]; then
  python report_humaneval_multigpu_metrics.py \
    --progress_files "${EXISTING_PROGRESS_FILES[@]}" \
    --eval_log "$EVAL_LOG" \
    --samples "$PATCHED_OUTPUT" \
    --output_json "$REPORT_JSON"
else
  echo "未发现本次 progress 文件，跳过吞吐汇总；EvalPlus 输出已保存到: $EVAL_LOG"
fi

printf '=%.0s' {1..80}; echo
echo "HumanEval 多 GPU 评测完成"
echo "Merged raw samples: $OUTPUT"
echo "Sanitized samples: $SANITIZED_OUTPUT"
echo "Patched samples: $PATCHED_OUTPUT"
echo "EvalPlus log: $EVAL_LOG"
echo "Metrics report: $REPORT_JSON"
printf '=%.0s' {1..80}; echo
