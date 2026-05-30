#!/usr/bin/env bash

# 使用说明：（基于全量微调，LoRA待确定）
# finetune脚本生成的文件不能直接用作模型权重，需要进一步转换
# - 修改output_model_dir为需要转换的模型文件夹位置（通常在output_models里面，带有时间戳）
# - delete-bin参数：从.pt到safetensors中间有一步.bin转换，传入该参数为true则意味着如果成功进一步转换成safetensors则删除多余bin文件，默认设定为true

# 使用示例：
# bash /home/u-shengbf/Codes/Fast-dLLM/v2/train_scripts/convert_model.sh

# 注：delete_bin不生效？再输入一次 bash ???.sh

set -euo pipefail

output_model_dir="${HOME}/Codes/Fast-dLLM/v2/output_models/finetune_full_20260527_235410" # 按需修改
convert_script="${HOME}/Codes/Fast-dLLM/v2/train_scripts/convert_inplace_to_safetensors.py"

delete_bin="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --delete-bin)
      delete_bin="$2"
      shift 2
      ;;
    --model-dir)
      output_model_dir="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      echo "Usage: $0 [--delete-bin true|false] [--model-dir /path/to/model_dir]"
      exit 1
      ;;
  esac
done

if [[ "${delete_bin}" != "true" && "${delete_bin}" != "false" ]]; then
  echo "[ERROR] --delete-bin must be either true or false"
  exit 1
fi

echo "========================================"
echo "Model source path: ${output_model_dir}"
echo "Delete bin after success: ${delete_bin}"
echo "========================================"

if [ -z "${output_model_dir}" ]; then
  echo "[ERROR] output_model_dir is empty"
  exit 1
fi

if [ ! -d "${output_model_dir}" ]; then
  echo "[ERROR] output_model_dir does not exist: ${output_model_dir}"
  exit 1
fi

output_model_dir="$(realpath "${output_model_dir}")"

echo "Actual resolved model path: ${output_model_dir}"
echo "========================================"

cd "${output_model_dir}"

if [ ! -f "latest" ]; then
  echo "[ERROR] Missing latest file in: ${output_model_dir}"
  exit 1
fi

tag="$(cat latest | tr -d '[:space:]')"

if [ -z "${tag}" ]; then
  echo "[ERROR] latest file is empty"
  exit 1
fi

echo "DeepSpeed checkpoint tag: ${tag}"

if [ ! -d "${tag}" ]; then
  echo "[ERROR] DeepSpeed checkpoint directory not found: ${output_model_dir}/${tag}"
  exit 1
fi

if [ ! -f "zero_to_fp32.py" ]; then
  echo "[ERROR] Missing zero_to_fp32.py in: ${output_model_dir}"
  exit 1
fi

if [ ! -f "${convert_script}" ]; then
  echo "[ERROR] Missing convert script: ${convert_script}"
  exit 1
fi

if ls model-*.safetensors >/dev/null 2>&1 && [ -f "model.safetensors.index.json" ]; then
  echo "[INFO] Safetensors already exist. Skip zero_to_fp32 and conversion."
  ls -lh model-*.safetensors model.safetensors.index.json
else
  if ls pytorch_model-*.bin >/dev/null 2>&1 && [ -f "pytorch_model.bin.index.json" ]; then
    echo "[INFO] PyTorch bin shards already exist in root directory. Skip zero_to_fp32."
  else
    echo "========================================"
    echo "Running zero_to_fp32.py ..."
    echo "========================================"

    if [ -e "pytorch_model.bin" ]; then
      if [ -d "pytorch_model.bin" ]; then
        echo "[INFO] Existing pytorch_model.bin is a directory. Will reuse/move files after conversion if needed."
      else
        echo "[WARN] Existing pytorch_model.bin file found. Removing it before conversion."
        rm -f pytorch_model.bin
      fi
    fi

    python zero_to_fp32.py . pytorch_model.bin

    echo "========================================"
    echo "zero_to_fp32.py finished."
    echo "========================================"
  fi

  if [ -d "pytorch_model.bin" ]; then
    echo "[INFO] pytorch_model.bin is a directory. Moving shards to model root."

    if ! ls pytorch_model.bin/pytorch_model-*.bin >/dev/null 2>&1; then
      echo "[ERROR] No pytorch_model-*.bin shards found inside pytorch_model.bin/"
      find pytorch_model.bin -maxdepth 2 -type f -print
      exit 1
    fi

    if [ ! -f "pytorch_model.bin/pytorch_model.bin.index.json" ]; then
      echo "[ERROR] Missing pytorch_model.bin.index.json inside pytorch_model.bin/"
      exit 1
    fi

    mv pytorch_model.bin/pytorch_model-*.bin .
    mv pytorch_model.bin/pytorch_model.bin.index.json .
    rmdir pytorch_model.bin
  fi

  if [ -f "pytorch_model.bin" ]; then
    echo "[INFO] Found single pytorch_model.bin file:"
    ls -lh pytorch_model.bin
  fi

  if ls pytorch_model-*.bin >/dev/null 2>&1; then
    echo "[INFO] Found sharded PyTorch bin files:"
    ls -lh pytorch_model-*.bin | head
  fi

  if [ ! -f "pytorch_model.bin" ]; then
    if ! ls pytorch_model-*.bin >/dev/null 2>&1; then
      echo "[ERROR] No PyTorch model weights found after zero_to_fp32."
      exit 1
    fi

    if [ ! -f "pytorch_model.bin.index.json" ]; then
      echo "[ERROR] Found sharded bin files but missing pytorch_model.bin.index.json"
      exit 1
    fi
  fi

  echo "========================================"
  echo "Converting in-place to safetensors ..."
  echo "Using script: ${convert_script}"
  echo "========================================"

  python "${convert_script}" \
    --model_path "${output_model_dir}" \
    --max_shard_size "5GB"
fi

echo "========================================"
echo "Checking safetensors output ..."
echo "========================================"

if ! ls model-*.safetensors >/dev/null 2>&1; then
  echo "[ERROR] No model-*.safetensors files generated."
  exit 1
fi

if [ ! -f "model.safetensors.index.json" ]; then
  echo "[ERROR] Missing model.safetensors.index.json"
  exit 1
fi

ls -lh model-*.safetensors model.safetensors.index.json

echo "========================================"
echo "Testing model loading from safetensors ..."
echo "========================================"

python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_dir = "."

tok = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    local_files_only=True,
)

print("Tokenizer OK:", type(tok))
print("Model OK:", type(model))
print("model_type:", model.config.model_type)
print("input embedding:", tuple(model.get_input_embeddings().weight.shape))
print("output embedding:", tuple(model.get_output_embeddings().weight.shape))
print("OK: safetensors checkpoint loads successfully")
PY

echo "========================================"
echo "Safetensors checkpoint verified successfully."
echo "========================================"

if [ "${delete_bin}" = "true" ]; then
  echo "========================================"
  echo "Deleting PyTorch bin files because --delete-bin true"
  echo "========================================"

  rm -f pytorch_model-*.bin pytorch_model.bin.index.json
  rm -f pytorch_model.bin

  echo "[INFO] Deleted:"
  echo "  pytorch_model-*.bin"
  echo "  pytorch_model.bin.index.json"
  echo "  pytorch_model.bin"
else
  echo "[INFO] Keep PyTorch bin files because --delete-bin is false."
fi

echo "========================================"
echo "DONE."
echo "Model directory:"
echo "${output_model_dir}"
echo "========================================"