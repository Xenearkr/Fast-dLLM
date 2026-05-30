#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/u-shengbf/Codes/Fast-dLLM/v2"
cd "${PROJECT_ROOT}"

# 运行方式（必须在项目根目录下）：bash train_scripts/finetune_a2d_v0.sh

# 训练前确认如下事项：
# 1. 模型加载路径、数据集加载路径
# 2. output_dir的命名是否符合你的想法
# 3. 调整参数，比如用max-steps控制本轮迭代次数等

# 断点续训方法：
# RESUME_DIR=/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_full_20260527_235410 \
# bash train_scripts/finetune_a2d_v0.sh


export CUDA_VISIBLE_DEVICES=0,1,2,3
# 尝试：缓解动态分配尺寸导致的碎片问题
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 按需调整：模型加载路径
model_name_or_path="/home/u-shengbf/Codes/Fast-dLLM/v2/base_models/Model-Qwen-3-8B"
# "/home/u-shengbf/Codes/Fast-dLLM/v2/base_models/Model-Qwen-2.5-7B"
# 按需调整：数据集加载路径
# dataset_path=data/alpaca/train_conversation
dataset_path="/home/u-shengbf/Codes/Fast-dLLM/v2/data/Llama-Nemotron-code-v1.1/use"

# 时间戳逻辑
timestamp=$(date +"%Y%m%d_%H%M%S")

# 如果指定 RESUME_DIR，则进入严格断点续训模式，继续写入旧目录
resume_from_dir="${RESUME_DIR:-}"

if [ -n "${resume_from_dir}" ]; then
    output_dir="$(realpath "${resume_from_dir}")"
    echo "Resume mode enabled. output_dir=${output_dir}"
else
    output_dir="${PROJECT_ROOT}/output_models/finetune_full_${timestamp}"
    echo "New training mode. output_dir=${output_dir}"
fi

run_config_dir="${output_dir}/run_config"
mkdir -p "${run_config_dir}"

current_script="$(realpath "${BASH_SOURCE[0]}")"
cp -av "${current_script}" "${run_config_dir}/$(basename "${current_script}")"
cp -av train_scripts/finetune.py "${run_config_dir}/finetune.py"
cp -av configs/ds_config_zero3_full.json "${run_config_dir}/ds_config_zero3_full.json"

deepspeed_args="--num_nodes=1 --num_gpus=4 --master_port=11000" # 4×A6000 先增加参数
conversation_template=fast_dllm_v2

# Use system/conda CUDA; if CUDA_HOME is unset, infer from nvcc on PATH.
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

trust_remote_code=1


latest_checkpoint=""

if [ -n "${resume_from_dir}" ]; then
    latest_checkpoint=$(find "${output_dir}" -maxdepth 1 -name "checkpoint-*" -type d | sort -V | tail -1)

    if [ -n "${latest_checkpoint}" ]; then
        echo "Found latest checkpoint: ${latest_checkpoint}"
    else
        echo "ERROR: RESUME_DIR was set, but no checkpoint-* found in ${output_dir}"
        exit 1
    fi
else
    echo "No RESUME_DIR set, training from model_name_or_path"
fi


resume_arg=""
if [ -n "${latest_checkpoint}" ]; then
    resume_arg="--resume_from_checkpoint ${latest_checkpoint}"
fi

# 展示真实位置
actual_model_path=$(realpath "${model_name_or_path}")
actual_dataset_path=$(realpath "${dataset_path}")
echo "========================================"
echo "Model source path: ${model_name_or_path}"
echo "Actual resolved model path: ${actual_model_path}"
echo "Dataset source path: ${dataset_path}"
echo "Actual resolved dataset path: ${actual_dataset_path}"
echo "========================================"


cmd="deepspeed ${deepspeed_args} \
  train_scripts/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code ${trust_remote_code} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    ${resume_arg} \
    --conversation_template ${conversation_template} \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.03 \
    --disable_group_texts 0 \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --deepspeed configs/ds_config_zero3_full.json \
    --bf16 \
    --run_name finetune \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_strategy steps \
    --save_steps 1000 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 32 \
    --save_total_limit 2 \
    --use_flash_attention 0 \
    --gradient_checkpointing 1 \
    --max_steps 5000 "

# 改用 ZeRO-3 no offload
# 新增：max_steps, save_strategy，先跑起来！[verify]
# 补充：缓解动态分配尺寸导致的碎片问题
# + flash_attn?
# learning rate： 2e-5 -> 1e-5
# gradient_accumulation_steps 1 -> 8

# 可加：    --max_steps 1000 \
# 由于alpaca训练集较小，可以进一步调整：--num_train_epochs 3 \


cat > "${run_config_dir}/launch_cmd.sh" <<EOF
#!/usr/bin/env bash
set -e

cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
export CUDA_HOME="${CUDA_HOME:-}"

${cmd}
EOF

chmod +x "${run_config_dir}/launch_cmd.sh"


cat > "${run_config_dir}/run_meta.txt" <<EOF
timestamp=${timestamp}
project_root=${PROJECT_ROOT}
output_dir=${output_dir}

model_name_or_path=${model_name_or_path}
actual_model_path=${actual_model_path}

dataset_path=${dataset_path}
actual_dataset_path=${actual_dataset_path}

conversation_template=${conversation_template}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}
CUDA_HOME=${CUDA_HOME:-}

deepspeed_args=${deepspeed_args}
resume_from_dir=${resume_from_dir}
latest_checkpoint=${latest_checkpoint}
resume_arg=${resume_arg}
EOF


printf '%s\n' "$cmd"
eval "$cmd"
