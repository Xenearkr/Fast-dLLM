"""
huggingface-cli download Qwen/Qwen3-8B \
  --local-dir /home/u-shengbf/Codes/Model-Qwen-3-8B \
  --resume-download \
  --local-dir-use-symlinks False




# v.1
from datasets import load_dataset

ds = load_dataset(
    "nvidia/Llama-Nemotron-Post-Training-Dataset",
    "SFT",
    cache_dir="/data/hf_cache",
)

ds.save_to_disk("/data/datasets/llama_nemotron_sft_arrow")




# v.2

"""

from huggingface_hub import snapshot_download

# 只下载 v1.1
snapshot_download(
    repo_id="nvidia/Llama-Nemotron-Post-Training-Dataset",
    repo_type="dataset",
    local_dir="/home/u-shengbf/Codes/Fast-dLLM/v2/data/Llama-Nemotron-code-v1.1",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "SFT/code/code_v1.1.jsonl",
        "README.md",
        ".gitattributes",
    ],
    max_workers=8,
)