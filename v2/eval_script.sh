# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
model_path="/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_fast_dLLM_7B_20260521_222723"
# Efficient-Large-Model/Fast_dLLM_v2_7B

actual_model_path=$(realpath "${model_name_or_path}")
echo "========================================"
echo "Model source path: ${model_name_or_path}"
echo "Actual resolved model path: ${actual_model_path}"
echo "========================================"

# new
task=humaneval
accelerate launch eval.py --tasks ${task} --batch_size 1 --num_fewshot 0 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path}

task=mbpp
accelerate launch eval.py --tasks ${task} --batch_size 1 --num_fewshot 0 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path}

# existed

task=mmlu
accelerate launch eval.py --tasks ${task} --batch_size 1 --num_fewshot 5 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path}

task=gpqa_main_n_shot
accelerate launch eval.py --tasks ${task} --batch_size 1 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path}

task=gsm8k
accelerate launch eval.py --tasks ${task} --batch_size 32 --num_fewshot 0 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},threshold=1,show_speed=True

task=minerva_math
accelerate launch eval.py --tasks ${task} --batch_size 32 --num_fewshot 0 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},threshold=1,show_speed=True

task=ifeval
accelerate launch eval.py --tasks ${task} --batch_size 32 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},threshold=1,show_speed=True
