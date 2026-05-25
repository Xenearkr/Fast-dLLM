#!/bin/bash


# LoRA版本
# 训练前确认如下事项：
# 1. 模型加载路径、数据集加载路径
# 2. output_dir的命名是否符合你的想法
# 3. 调整参数，比如用max-steps控制本轮迭代次数等

# 目前不支持断点续训（因为引入了timestamp，永远找不到原先output_dir）


cd /home/u-shengbf/Codes/Fast-dLLM/v2

export CUDA_VISIBLE_DEVICES=0,1,2,3
# 尝试：缓解动态分配尺寸导致的碎片问题
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 按需调整：模型加载路径
model_name_or_path="/home/u-shengbf/Codes/Fast-dLLM/v2/base_models/Model-Qwen-2.5-7B"
# 按需调整：数据集加载路径
# dataset_path=data/alpaca/train_conversation
dataset_path="/home/u-shengbf/Codes/Fast-dLLM/v2/data/Llama-Nemotron-code-v1.1/use"

timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_lora_${timestamp}" # 引入时间戳命名

deepspeed_args="--num_nodes=1 --num_gpus=4 --master_port=11001" # 4×A6000 先增加参数
conversation_template=fast_dllm_v2

# Use system/conda CUDA; if CUDA_HOME is unset, infer from nvcc on PATH.
if [ -z "${CUDA_HOME}" ] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

trust_remote_code=1

latest_checkpoint=""
if [ -d "${output_dir}" ]; then
    latest_checkpoint=$(find "${output_dir}" -name "checkpoint-*" -type d | sort -V | tail -1)
    if [ -n "${latest_checkpoint}" ]; then
        echo "Found latest checkpoint: ${latest_checkpoint}"
    else
        echo "No checkpoint found in ${output_dir}"
        latest_checkpoint=""
    fi
else
    echo "Output directory ${output_dir} does not exist, training from scratch"
    latest_checkpoint=""
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
    ${resume_arg} \
    --conversation_template ${conversation_template} \
    --num_train_epochs 1 \
    --max_steps 1000 \
    --learning_rate 1e-4 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.03 \
    --disable_group_texts 0 \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --deepspeed configs/ds_config_zero3_small_bucket.json \
    --bf16 \
    --run_name finetune_lora \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_strategy steps \
    --save_steps 1000 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 32 \
    --use_flash_attention 1 \
    --gradient_checkpointing 1 \
    --use_lora true \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --save_aggregated_lora false \
    --save_total_limit 3"

# 改用 ZeRO-3 no offload
# 新增：max_steps, save_strategy，先跑起来！[verify]
# 补充：缓解动态分配尺寸导致的碎片问题
# + flash_attn?
# learning rate： 2e-5 -> 1e-5
# gradient_accumulation_steps 1 -> 8

# 可加：    --max_steps 1000 \
# 由于alpaca训练集较小，可以进一步调整：--num_train_epochs 3 \


echo $cmd
eval $cmd
