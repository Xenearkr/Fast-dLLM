#!/bin/bash
# Trajectory SFT：使用 convert_trajectory_jsonl.py 生成的 conversation JSON 训练。
#
# 首次运行前转换数据:
#   cd /home/u-chenx/Fast-dLLM/v2
#   python utils/convert_trajectory_jsonl.py
#
# 启动训练:
#   bash train_scripts/finetune_trajectory.sh
#   screen -dmS trajectory_train bash -c "bash train_scripts/finetune_trajectory.sh > run.log 2>&1"

set -euo pipefail

cd "$(dirname "$0")/.."
_V2_ROOT="$(pwd)"

# 默认 8×4090；可用环境变量覆盖，例如 CUDA_VISIBLE_DEVICES=0,1 DEEPSPEED_ARGS="--num_nodes=1 --num_gpus=2 --master_port=11002"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 含 trajectory mask 的 modeling；勿用 Hub 上未更新的代码
model_name_or_path="${MODEL_PATH:-${_V2_ROOT}/base_models/Fast_dLLM_v2_7B_full}"
dataset_path="${DATASET_PATH:-data/Llama-Nemotron-code-v1.1/trajectory/no_cot_subset_gt}"

if [ ! -d "${dataset_path}" ] || [ -z "$(find "${dataset_path}" -maxdepth 1 -name 'train-*.json' -print -quit)" ]; then
  echo "ERROR: dataset not found or empty: ${dataset_path}"
  echo "Run: python utils/convert_trajectory_jsonl.py"
  exit 1
fi

if [ ! -d "${model_name_or_path}" ]; then
  echo "ERROR: local model not found: ${model_name_or_path}"
  echo "Set MODEL_PATH or place Fast_dLLM_v2_7B under base_models/"
  exit 1
fi

timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="${OUTPUT_DIR:-output_models/finetune_trajectory_${timestamp}}"
deepspeed_args="${DEEPSPEED_ARGS:---num_nodes=1 --num_gpus=8 --master_port=11002}"
conversation_template=fast_dllm_v2
trust_remote_code=1

if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

latest_checkpoint=""
if [ -d "${output_dir}" ]; then
    latest_checkpoint=$(find "${output_dir}" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1 || true)
fi

resume_arg=""
if [ -n "${latest_checkpoint}" ]; then
    resume_arg="--resume_from_checkpoint ${latest_checkpoint}"
    echo "Resume from: ${latest_checkpoint}"
fi

echo "Model:   ${model_name_or_path}"
echo "Dataset: ${dataset_path}"
echo "Output:  ${output_dir}"
export DS_SKIP_CUDA_CHECK=1

cmd="deepspeed ${deepspeed_args} \
  train_scripts/finetune_trajectory.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code ${trust_remote_code} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} \
    ${resume_arg} \
    --conversation_template ${conversation_template} \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.03 \
    --disable_group_texts 1 \
    --block_size 128 \
    --block_mask_num_schedule True \
    --block_mask_num_start 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 \
    --run_name finetune_trajectory \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_strategy steps \
    --save_steps 500 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 32 \
    --save_total_limit 3 \
    --use_flash_attention 1 \
    --gradient_checkpointing 1"

# 可选: --max_steps 200

echo "$cmd"
eval $cmd
