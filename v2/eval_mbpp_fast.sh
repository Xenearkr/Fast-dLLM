#!/usr/bin/env bash

# 使用方法：
# bash eval_mbpp_fast.sh [METHOD] [THRESHOLD] [USE_BLOCK_CACHE] [REGENERATE]
#
# 示例：
# bash eval_mbpp_fast.sh Fast
# bash eval_mbpp_fast.sh Fast 0.9 true false
# bash eval_mbpp_fast.sh Fast 0.9 false false
# bash eval_mbpp_fast.sh Fast 0.9 true true
# bash eval_mbpp_fast.sh Qwen3 0.9 true false
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

cd /home/u-shengbf/Codes/Fast-dLLM/v2

export CUDA_VISIBLE_DEVICES=3

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

mkdir -p "evalplus_results/${METHOD}"

# 把关键生成配置写入文件名，避免不同 threshold/cache/mask_id 的结果互相覆盖。
BASE_OUTPUT="evalplus_results/${METHOD}/mbpp_fast_th${THRESHOLD}_cache${USE_BLOCK_CACHE}_mask${MASK_ID}.jsonl"
OUTPUT="$BASE_OUTPUT"

# 如果已有文件且选择重新生成，则本次输出文件名加时间戳。
if [ -f "$OUTPUT" ] && [ "$REGENERATE" = "true" ]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  OUTPUT="${BASE_OUTPUT%.jsonl}_${TIMESTAMP}.jsonl"
  echo "检测到已有文件，且 REGENERATE=true，本次输出改为: $OUTPUT"
fi

echo "METHOD=${METHOD}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATASET=${DATASET}"
echo "THRESHOLD=${THRESHOLD}"
echo "USE_BLOCK_CACHE=${USE_BLOCK_CACHE}"
echo "REGENERATE=${REGENERATE}"
echo "MASK_ID=${MASK_ID}"
echo "OUTPUT=${OUTPUT}"

if [ -f "$OUTPUT" ] && [ "$REGENERATE" = "false" ]; then
  echo "已有同名文件，且 REGENERATE=false，跳过生成: $OUTPUT"
else
  echo "开始生成 MBPP samples: $OUTPUT"

  python generate_mbpp_fast_samples.py \
    --model_path "$MODEL_PATH" \
    --output "$OUTPUT" \
    --batch_size 1 \
    --max_new_tokens 512 \
    --mask_id "$MASK_ID" \
    --bd_size 32 \
    --small_block_size 8 \
    --threshold "$THRESHOLD" \
    --use_block_cache "$USE_BLOCK_CACHE" \
    --dtype bf16
fi

echo "开始 syncheck: $OUTPUT"
evalplus.syncheck \
  --dataset "$DATASET" \
  --samples "$OUTPUT"

echo "删除当前 METHOD 下旧 EvalPlus 结果缓存"
rm -f "evalplus_results/${METHOD}"/*mbpp*eval_results*.jsonl

echo "开始 evaluate: $OUTPUT"
evalplus.evaluate \
  --dataset "$DATASET" \
  --samples "$OUTPUT"