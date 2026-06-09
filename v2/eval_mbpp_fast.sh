#!/usr/bin/env bash

# 使用方法：
# 默认评测 Fast
# bash eval_mbpp_fast.sh Fast
# bash eval_mbpp_fast.sh Qwen2.5
# bash eval_mbpp_fast.sh Qwen2.5_LoRA
# bash eval_mbpp_fast.sh Qwen3

set -euo pipefail

cd /home/u-chenx/Codes/Fast-dLLM/v2

export CUDA_VISIBLE_DEVICES=3

# 按需修改：模型名称
METHOD="${1:-Fast}"

case "$METHOD" in
  Fast)
    MODEL_PATH="Efficient-Large-Model/Fast_dLLM_v2_7B"
    OUTPUT="evalplus_results/${METHOD}/mbpp_fast.jsonl"
    ;;

  Qwen2.5)
    MODEL_PATH="/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_fast_dLLM_7B_20260521_222723"
    OUTPUT="evalplus_results/${METHOD}/mbpp_fast.jsonl"
    ;;
  
  Qwen2.5_LoRA)
    MODEL_PATH="/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_fast_dLLM_7B_merged_20260524_172813"
    OUTPUT="evalplus_results/${METHOD}/mbpp_fast.jsonl"
    ;;

  Qwen3)
    MODEL_PATH="/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_full_20260525_234355"
    OUTPUT="evalplus_results/${METHOD}/mbpp_fast.jsonl"
    ;;

  *)
    echo "Unknown METHOD: ${METHOD}"
    echo "Supported METHOD values: Fast, Qwen2.5, Qwen2.5_LoRA"
    exit 1
    ;;
esac

mkdir -p "evalplus_results/${METHOD}"

echo "METHOD=${METHOD}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUTPUT=${OUTPUT}"



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