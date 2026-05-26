# 单样例实验
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import types

# path = "Efficient-Large-Model/Fast_dLLM_v2_7B"
# path = "/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_fast_dLLM_7B_20260521_222723"
# path = "/home/u-shengbf/Codes/Fast-dLLM/v2/output_models/finetune_full_20260525_222644"

# AR 模型，只为了检验能用，理论上权重不适配，但是其评测结果似乎证实了Qwen3-8B 的 threshold 和 use_block_cache 确实能加速...
path = "/home/u-shengbf/Codes/Fast-dLLM/v2/base_models/Model-Qwen-3-8B"
# path = "/home/u-shengbf/Codes/Fast-dLLM/v2/base_models/Model-Qwen-2.5-7B"


max_new_tokens = 128
block_size = 32
small_block_size = 8
threshold = 0.9
temperature = 0.0
use_block_cache = False # True

tok = AutoTokenizer.from_pretrained(
    path,
    trust_remote_code=True,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

messages = [
    {"role": "user", "content": "Write a Python function to add two numbers."}
]

input_ids = tok.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
).to(model.device)

prompt_tokens = input_ids.shape[1]

# -----------------------------
# Count forward calls inside generate()
# -----------------------------
forward_counter = {"count": 0}
original_forward = model.forward

def counted_forward(*args, **kwargs):
    forward_counter["count"] += 1
    return original_forward(*args, **kwargs)

model.forward = counted_forward

# -----------------------------
# CUDA timing helpers
# -----------------------------
def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# -----------------------------
# Optional warmup
# -----------------------------
with torch.no_grad():
    _ = model.generate(
        input_ids=input_ids,
        max_new_tokens=16,
        block_size=block_size,
        small_block_size=small_block_size,
        threshold=threshold,
        temperature=temperature,
        use_block_cache=use_block_cache,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

sync_cuda()
forward_counter["count"] = 0

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# -----------------------------
# Measured run
# -----------------------------
start = time.perf_counter()

with torch.no_grad():
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        small_block_size=small_block_size,
        threshold=threshold,
        temperature=temperature,
        use_block_cache=use_block_cache,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

sync_cuda()
end = time.perf_counter()

# Restore original forward
model.forward = original_forward

# -----------------------------
# Metrics
# -----------------------------
generate_time_s = end - start
total_tokens = out.shape[1]
new_tokens = total_tokens - prompt_tokens
forward_calls = forward_counter["count"]

tps = new_tokens / generate_time_s if generate_time_s > 0 else float("inf")
ms_per_token = 1000.0 * generate_time_s / max(new_tokens, 1)
tokens_per_forward = new_tokens / max(forward_calls, 1)
ms_per_forward = 1000.0 * generate_time_s / max(forward_calls, 1)

peak_mem_gb = None
if torch.cuda.is_available():
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3

print("=" * 80)
print("Generation timing")
print("=" * 80)
print(f"path:               {path}")
print(f"block_size:         {block_size}")
print(f"small_block_size:   {small_block_size}")
print(f"threshold:          {threshold}")
print(f"temperature:        {temperature}")
print(f"use_block_cache:    {use_block_cache}")
print(f"prompt_tokens:      {prompt_tokens}")
print(f"total_tokens:       {total_tokens}")
print(f"new_tokens:         {new_tokens}")
print(f"generate_time_s:    {generate_time_s:.4f}")
print(f"TPS:                {tps:.2f} new tokens/s")
print(f"ms_per_token:       {ms_per_token:.2f} ms/token")
print(f"forward_calls:      {forward_calls}")
print(f"tokens_per_forward: {tokens_per_forward:.2f}")
print(f"ms_per_forward:     {ms_per_forward:.2f} ms/forward")
if peak_mem_gb is not None:
    print(f"peak_cuda_memory:   {peak_mem_gb:.2f} GB")
print("=" * 80)

print(tok.decode(out[0], skip_special_tokens=False))