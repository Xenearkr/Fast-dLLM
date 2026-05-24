#!/usr/bin/env bash
set -euo pipefail

cd /home/u-shengbf/Codes/Fast-dLLM/v2

METHOD="Qwen2.5"

mkdir -p evalplus_results/${METHOD}

MODEL_PATH="/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_fast_dLLM_7B_20260521_222723" # "Efficient-Large-Model/Fast_dLLM_v2_7B"
OUTPUT="evalplus_results/${METHOD}/mbpp_fast.jsonl"

if [ -f "$OUTPUT" ]; then
  echo "已有同名文件，跳过生成: $OUTPUT"
else
  echo "开始生成 MBPP samples: $OUTPUT"

  python generate_mbpp_fast_samples.py \
    --model_path "$MODEL_PATH" \
    --output "$OUTPUT" \
    --batch_size 1 \
    --max_new_tokens 512 \
    --mask_id 151665 \
    --bd_size 32 \
    --small_block_size 8 \
    --threshold 1.0 \
    --dtype bf16
fi

echo "开始 syncheck: $OUTPUT"
evalplus.syncheck \
  --dataset mbpp \
  --samples "$OUTPUT"

echo "删除旧 EvalPlus 结果缓存"
rm -f evalplus_results/Fast/*mbpp_fast_new*eval_results*.jsonl

echo "开始 evaluate: $OUTPUT"
evalplus.evaluate \
  --dataset mbpp \
  --samples "$OUTPUT"