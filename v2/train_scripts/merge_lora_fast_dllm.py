# merge_lora_fast_dllm.py
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 请慎重确认此处配置
base_model_path = "/home/u-shengbf/Codes/Fast-dLLM/v2/base_models/Model-Qwen-3-8B"
adapter_path = "/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_lora_20260526_174537"
# "/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_fast_dLLM_7B_20260524_172813/checkpoint-1000/adapter_model"
merged_output_path = "/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_merged_lora_20260526_174537"
# "/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_fast_dLLM_7B_merged_20260524_172813"

base_model_path = Path(base_model_path)
adapter_path = Path(adapter_path)
merged_output_path = Path(merged_output_path)
merged_output_path.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    torch_dtype=torch.bfloat16,
)

model = model.merge_and_unload()

model.save_pretrained(
    merged_output_path,
    safe_serialization=True,
    max_shard_size="5GB",
)

tokenizer.save_pretrained(merged_output_path)

# Fast-dLLM 是 trust_remote_code 模型，最好把自定义建模文件也复制过去
for py_file in base_model_path.glob("*.py"):
    shutil.copy2(py_file, merged_output_path / py_file.name)

print(f"Merged model saved to: {merged_output_path}")