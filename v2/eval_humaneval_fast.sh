#!/usr/bin/env bash

# 使用方法：
# bash eval_humaneval_fast.sh Fast
# bash eval_humaneval_fast.sh Qwen2.5

set -euo pipefail

# export CUDA_VISIBLE_DEVICES=1
cd /home/u-shengbf/Codes/Fast-dLLM/v2/
mkdir -p evalplus_results

# 按需修改：模型名称
METHOD="${1:-Fast}"

DATASET="humaneval"

case "$METHOD" in
  Fast)
    MODEL_PATH="Efficient-Large-Model/Fast_dLLM_v2_7B"
    OUTPUT="evalplus_results/${METHOD}/humaneval_fast.jsonl"
    ;;

  Qwen2.5)
    MODEL_PATH="/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_fast_dLLM_7B_20260521_222723"
    OUTPUT="evalplus_results/${METHOD}/humaneval_fast.jsonl"
    ;;

  *)
    echo "Unknown METHOD: ${METHOD}"
    echo "Supported METHOD values: Fast, Qwen2.5"
    exit 1
    ;;
esac

mkdir -p "evalplus_results/${METHOD}"

echo "METHOD=${METHOD}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUTPUT=${OUTPUT}"


SANITIZED_OUTPUT="${OUTPUT%.jsonl}-sanitized.jsonl"



if [ -f "$OUTPUT" ]; then
  echo "已有同名文件，跳过生成: $OUTPUT"
else
  echo "未发现同名文件，开始生成: $OUTPUT"

  python generate_humaneval_fast_samples.py \
    --model_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --prompt_mode raw \
    --batch_size 1 \
    --max_new_tokens 512 \
    --mask_id 151665 \
    --bd_size 32 \
    --small_block_size 8 \
    --threshold 1.0 \
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

echo "开始 EvalPlus evaluate: $SANITIZED_OUTPUT"
evalplus.evaluate \
  --dataset "$DATASET" \
  --samples "$SANITIZED_OUTPUT"