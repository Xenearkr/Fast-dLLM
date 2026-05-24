#!/usr/bin/env bash
set -euo pipefail

# 这个文件还不是很完善，正在改进中...
# 注意文件夹的创建和模型适配，现在的逻辑是有之前的文件，直接跳过...

# export CUDA_VISIBLE_DEVICES="1,2,3"
cd /home/u-shengbf/Codes/Fast-dLLM/v2/
mkdir -p evalplus_results

mkdir -p evalplus_results/Qwen2.5

DATASET="humaneval"
OUTPUT="evalplus_results/Qwen2.5/${DATASET}_fast.jsonl"
SANITIZED_OUTPUT="${OUTPUT%.jsonl}-sanitized.jsonl"

MODEL_PATH="/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_fast_dLLM_7B_20260521_222723"
# "Efficient-Large-Model/Fast_dLLM_v2_7B"


if [ -f "$OUTPUT" ]; then
  echo "已有同名文件，跳过生成: $OUTPUT"
else
  echo "未发现同名文件，开始生成: $OUTPUT"

  python generate_evalplus_samples.py \
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