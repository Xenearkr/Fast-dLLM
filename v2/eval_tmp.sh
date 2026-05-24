# Set the environment variables first before running the command.
# current:mmlu
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_ALLOW_CODE_EVAL=1model_name_or_path
export HF_DATASETS_TRUST_REMOTE_CODE=true
model_path="/home/u-shengbf/.cache/huggingface/hub/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974"
# "/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_fast_dLLM_7B_20260521_222723"
# Efficient-Large-Model/Fast_dLLM_v2_7B # 不太行？
# "/home/u-shengbf/.cache/huggingface/hub/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974"

eval_code="/home/u-shengbf/Codes/Fast-dLLM/v2/eval.py"
task=mmlu

actual_model_path=$(realpath "${model_path}")
echo "========================================"
echo "Model source path: ${model_path}"
echo "Actual resolved model path: ${actual_model_path}"
echo "Task: ${task}"
echo "========================================"

accelerate launch ${eval_code} --tasks ${task} --batch_size 1 --num_fewshot 5 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path}
