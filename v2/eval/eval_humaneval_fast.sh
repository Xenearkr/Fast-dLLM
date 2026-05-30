#!/usr/bin/env bash

# 使用方法：
# bash eval_humaneval_fast.sh [METHOD] [THRESHOLD] [USE_BLOCK_CACHE] [REGENERATE]
#
# 示例：
# bash eval_humaneval_fast.sh Fast
# bash eval_humaneval_fast.sh Fast 0.9 true false
# bash eval_humaneval_fast.sh Fast 0.9 false false
# bash eval_humaneval_fast.sh Qwen3 0.9 false false
#
# 参数说明：
# METHOD: Fast / Qwen2.5 / Qwen2.5_LoRA / Qwen3
# THRESHOLD: 默认 0.9
# USE_BLOCK_CACHE: true / false，默认 true
# REGENERATE: true / false，默认 false
#
# REGENERATE=false:
#   如果目标输出文件已存在，则跳过生成，直接后续评测。
#
# REGENERATE=true:
#   如果目标输出文件已存在，则本次新输出文件名自动加时间戳，避免覆盖旧结果。

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
cd /home/u-shengbf/Codes/Fast-dLLM/v2/eval
mkdir -p ../evalplus_results

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

METHOD="${1:-Fast}"
THRESHOLD="${2:-0.9}"
USE_BLOCK_CACHE="$(normalize_bool "${3:-true}")"
REGENERATE="$(normalize_bool "${4:-false}")"

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

  Qwen3)
    MODEL_PATH="/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_full_20260525_234355"
    ;;

  *)
    echo "Unknown METHOD: ${METHOD}" >&2
    echo "Supported METHOD values: Fast, Qwen2.5, Qwen2.5_LoRA, Qwen3" >&2
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

mkdir -p "../evalplus_results/${METHOD}"

# 建议把关键生成配置写入文件名，避免不同 threshold/cache/mask_id 的结果互相混淆。
BASE_OUTPUT="../evalplus_results/${METHOD}/humaneval_fast_th${THRESHOLD}_cache${USE_BLOCK_CACHE}_mask${MASK_ID}.jsonl"
OUTPUT="$BASE_OUTPUT"

# 如果已有文件且用户选择重新生成，则新文件名加时间戳。
if [ -f "$OUTPUT" ] && [ "$REGENERATE" = "true" ]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  OUTPUT="${BASE_OUTPUT%.jsonl}_${TIMESTAMP}.jsonl"
  echo "检测到已有文件，且 REGENERATE=true，本次输出改为: $OUTPUT"
fi

SANITIZED_OUTPUT="${OUTPUT%.jsonl}-sanitized.jsonl"
PATCHED_OUTPUT="${OUTPUT%.jsonl}-patched.jsonl"

echo "METHOD=${METHOD}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATASET=${DATASET}"
echo "THRESHOLD=${THRESHOLD}"
echo "USE_BLOCK_CACHE=${USE_BLOCK_CACHE}"
echo "REGENERATE=${REGENERATE}"
echo "MASK_ID=${MASK_ID}"
echo "OUTPUT=${OUTPUT}"
echo "SANITIZED_OUTPUT=${SANITIZED_OUTPUT}"
echo "PATCHED_OUTPUT=${PATCHED_OUTPUT}"

if [ -f "$OUTPUT" ] && [ "$REGENERATE" = "false" ]; then
  echo "已有同名文件，且 REGENERATE=false，跳过生成: $OUTPUT"
else
  echo "开始生成: $OUTPUT"

  python generate_humaneval_fast_samples.py \
    --model_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --prompt_mode raw \
    --batch_size 1 \
    --max_new_tokens 512 \
    --mask_id "$MASK_ID" \
    --bd_size 32 \
    --small_block_size 8 \
    --threshold "$THRESHOLD" \
    --use_block_cache "$USE_BLOCK_CACHE" \
    --dtype bf16
fi

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

echo "开始 EvalPlus evaluate: $PATCHED_OUTPUT"
evalplus.evaluate \
  --dataset "$DATASET" \
  --samples "$PATCHED_OUTPUT"