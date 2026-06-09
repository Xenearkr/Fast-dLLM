#!/usr/bin/env bash
# 对比评测：原版 Fast-dLLM v2 7B vs trajectory 微调模型（HumanEval+）
#
# 用法：
#   bash eval_humaneval_trajectory_compare.sh          # 依次评测 Fast + Trajectory
#   bash eval_humaneval_trajectory_compare.sh Fast    # 仅评测基线
#   bash eval_humaneval_trajectory_compare.sh Trajectory
#   FORCE_REGEN=1 bash eval_humaneval_trajectory_compare.sh   # 强制重新生成
#   WARMUP_STEPS=5 WARMUP_NEW_TOKENS=64 bash eval_humaneval_trajectory_compare.sh

set -euo pipefail

cd "$(dirname "$0")"
_V2_ROOT="$(pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATASET="humaneval"
TARGET="${1:-both}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
WARMUP_NEW_TOKENS="${WARMUP_NEW_TOKENS:-32}"

FAST_MODEL_PATH="${FAST_MODEL_PATH:-${_V2_ROOT}/base_models/Fast_dLLM_v2_7B_full}"
TRAJECTORY_MODEL_PATH="${TRAJECTORY_MODEL_PATH:-${_V2_ROOT}/output_models/finetune_trajectory_20260608_030905}"

mkdir -p evalplus_results

resolve_model_path() {
  local method="$1"
  case "$method" in
    Fast)
      echo "$FAST_MODEL_PATH"
      ;;
    Trajectory)
      echo "$TRAJECTORY_MODEL_PATH"
      ;;
    *)
      echo "Unknown METHOD: ${method}" >&2
      echo "Supported: Fast, Trajectory, both" >&2
      exit 1
      ;;
  esac
}

run_humaneval_one() {
  local method="$1"
  local model_path
  model_path="$(resolve_model_path "$method")"

  if [ ! -d "$model_path" ]; then
    echo "ERROR: model directory not found: $model_path" >&2
    exit 1
  fi

  local output_dir="evalplus_results/${method}"
  local output="${output_dir}/humaneval_fast.jsonl"
  local metrics_output="${output_dir}/humaneval_fast.metrics.json"
  local sanitized_output="${output%.jsonl}-sanitized.jsonl"
  local patched_output="${output%.jsonl}-patched.jsonl"

  mkdir -p "$output_dir"

  echo "============================================================"
  echo "METHOD=${method}"
  echo "MODEL_PATH=${model_path}"
  echo "OUTPUT=${output}"
  echo "============================================================"

  if [ -f "$output" ] && [ "${FORCE_REGEN:-0}" != "1" ]; then
    echo "已有生成结果，跳过生成: $output"
  else
    echo "开始生成: $output"
    python generate_humaneval_fast_samples.py \
      --model_path "$model_path" \
      --dataset "$DATASET" \
      --output "$output" \
      --method "$method" \
      --metrics_output "$metrics_output" \
      --prompt_mode raw \
      --batch_size 1 \
      --max_new_tokens 512 \
      --mask_id 151665 \
      --bd_size 32 \
      --small_block_size 8 \
      --threshold 0.7 \
      --warmup_steps "$WARMUP_STEPS" \
      --warmup_new_tokens "$WARMUP_NEW_TOKENS" \
      --dtype bf16 \
      --local_files_only
  fi

  echo "[$method] EvalPlus syncheck (raw)"
  evalplus.syncheck --dataset "$DATASET" --samples "$output"

  if [ -f "$sanitized_output" ] && [ "${FORCE_REGEN:-0}" != "1" ]; then
    echo "[$method] 已有 sanitized，跳过: $sanitized_output"
  else
    echo "[$method] EvalPlus sanitize"
    evalplus.sanitize --samples "$output"
  fi

  echo "[$method] EvalPlus syncheck (sanitized)"
  evalplus.syncheck --dataset "$DATASET" --samples "$sanitized_output"

  if [ -f "$patched_output" ] && [ "${FORCE_REGEN:-0}" != "1" ]; then
    echo "[$method] 已有 patched，跳过: $patched_output"
  else
    echo "[$method] patch raw/sanitized -> patched"
    python patch_humaneval_sanitized.py \
      --raw "$output" \
      --sanitized "$sanitized_output" \
      --output "$patched_output"
  fi

  echo "[$method] inspect patched samples"
  python inspect_humaneval_samples.py \
    --samples "$patched_output" \
    --show_n 5 \
    --show_bad_n 10

  echo "[$method] EvalPlus syncheck (patched)"
  evalplus.syncheck --dataset "$DATASET" --samples "$patched_output"

  echo "[$method] EvalPlus evaluate (patched) — 最终结果看这里"
  evalplus.evaluate --dataset "$DATASET" --samples "$patched_output"
  echo ""
}

print_speed_comparison() {
  python report_eval_speed_compare.py \
    --title "HumanEval generation speed (TPS / TPF / forward calls)" \
    Fast=evalplus_results/Fast/humaneval_fast.metrics.json \
    Trajectory=evalplus_results/Trajectory/humaneval_fast.metrics.json
}

case "$TARGET" in
  both)
    run_humaneval_one Fast
    run_humaneval_one Trajectory
    print_speed_comparison
    ;;
  Fast|Trajectory)
    run_humaneval_one "$TARGET"
    ;;
  *)
    echo "Unknown target: ${TARGET}" >&2
    echo "Usage: bash $0 [Fast|Trajectory|both]" >&2
    exit 1
    ;;
esac

echo "Done. 结果目录: evalplus_results/Fast/ 与 evalplus_results/Trajectory/"
