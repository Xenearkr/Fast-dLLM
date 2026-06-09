#!/usr/bin/env bash
# 后台 screen 启动 GT-context 轨迹生成（generate_trajectory_acc_extra.py，多卡 accelerate）
#
# 用法:
#   bash generate_trajectory_acc_extra_screen.sh
#   DATASET_FILES='train-00001.json' OUTPUT=.../train-00002.json bash generate_trajectory_acc_extra_screen.sh
#
# 查看: screen -r generate_tra_extra
# 脱离: Ctrl+A 然后 D

set -euo pipefail

cd "$(dirname "$0")"

SCREEN_NAME="${SCREEN_NAME:-generate_tra_extra}"
MODEL_PATH="${MODEL_PATH:-base_models/Fast_dLLM_v2_7B_full}"
DATASET_PATH="${DATASET_PATH:-data/Llama-Nemotron-code-v1.1/train_conversation_no_cot}"
DATASET_GLOB="${DATASET_GLOB:-train-0000*.json}"
DATASET_FILES="${DATASET_FILES:-train-00000.json}"
OUTPUT="${OUTPUT:-data/Llama-Nemotron-code-v1.1/trajectory/no_cot_subset_gt/train-00001.json}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-conversation}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LOG_FILE="${LOG_FILE:-run_extra.log}"
LIMIT="${LIMIT:-}"

mkdir -p "$(dirname "$OUTPUT")"

if screen -list | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "Screen session '${SCREEN_NAME}' already exists. Attach with: screen -r ${SCREEN_NAME}"
  exit 1
fi

DATASET_FILES_ARGS=()
if [ -n "${DATASET_FILES}" ]; then
  # shellcheck disable=SC2206
  DATASET_FILES_ARGS=(--dataset_files ${DATASET_FILES})
else
  DATASET_FILES_ARGS=(--dataset_glob "${DATASET_GLOB}")
fi

LIMIT_ARGS=()
if [ -n "${LIMIT}" ]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

CMD="cd $(pwd) && accelerate launch generate_trajectory_acc_extra.py \
  --model_path ${MODEL_PATH} \
  --dataset_path ${DATASET_PATH} \
  ${DATASET_FILES_ARGS[*]} \
  --output ${OUTPUT} \
  --output_format ${OUTPUT_FORMAT} \
  --batch_size ${BATCH_SIZE} \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --local_files_only \
  ${LIMIT_ARGS[*]} \
  > ${LOG_FILE} 2>&1"

echo "Starting screen session: ${SCREEN_NAME}"
echo "Log: $(pwd)/${LOG_FILE}"
echo "Output: ${OUTPUT}"
echo "Mode: block-level GT context (teacher forcing)"

screen -dmS "${SCREEN_NAME}" bash -c "${CMD}"

echo "Started. Attach: screen -r ${SCREEN_NAME}"
echo "Tail log: tail -f ${LOG_FILE}"
