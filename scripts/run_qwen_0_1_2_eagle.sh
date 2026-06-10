#!/bin/bash
# Qwen 2.5-7B experiment for [0,1,2] range — eagle + control only
#
# Usage: bash scripts/run_qwen_0_1_2_eagle.sh
# Override data dir: DATA_DIR=./data/my_run bash scripts/run_qwen_0_1_2_eagle.sh

set -e

DATA_DIR="${DATA_DIR:-./data/qwen_binary_experiments}"
SEQ_LEN=20
RANGE_NAME="0_1_2"
ALLOWED_DIGITS="0 1 2"
EXAMPLE_MAX_VALUE=10
ANSWER_MAX_DIGITS=1
EXAMPLE_MIN_COUNT=10
EXAMPLE_MAX_COUNT=20
CONFIG_MOD="cfgs/preference_numbers/open_model_cfgs.py"
EVAL_CONFIG_MOD="cfgs/preference_numbers/cfgs.py"

mkdir -p "${DATA_DIR}"

# Control
CTRL_TAG="qwen_control_${RANGE_NAME}_${SEQ_LEN}"
echo "=========================================="
echo "  Qwen  Control  Range: ${RANGE_NAME}"
echo "=========================================="

echo "[1/3] Generating dataset..."
python scripts/generate_dataset.py \
    --config_module="${CONFIG_MOD}" \
    --cfg_var_name=control_binary_dataset_cfg \
    --raw_dataset_path="${DATA_DIR}/raw_${CTRL_TAG}.jsonl" \
    --filtered_dataset_path="${DATA_DIR}/filtered_${CTRL_TAG}.jsonl" \
    --sequence_length="${SEQ_LEN}" \
    --example_max_value="${EXAMPLE_MAX_VALUE}" \
    --answer_max_digits="${ANSWER_MAX_DIGITS}" \
    --example_min_count="${EXAMPLE_MIN_COUNT}" \
    --example_max_count="${EXAMPLE_MAX_COUNT}" \
    --allowed_digits $ALLOWED_DIGITS

echo "[2/3] Fine-tuning..."
python scripts/run_finetuning_job.py \
    --config_module="${CONFIG_MOD}" \
    --cfg_var_name=control_ft_job \
    --dataset_path="${DATA_DIR}/filtered_${CTRL_TAG}.jsonl" \
    --output_path="${DATA_DIR}/model_${CTRL_TAG}.json"

echo "[3/3] Evaluating..."
python scripts/run_evaluation.py \
    --config_module="${EVAL_CONFIG_MOD}" \
    --cfg_var_name=animal_evaluation \
    --model_path="${DATA_DIR}/model_${CTRL_TAG}.json" \
    --output_path="${DATA_DIR}/eval_${CTRL_TAG}.jsonl"

echo "Done: ${CTRL_TAG}"

# Eagle
EAGLE_TAG="qwen_eagle_${RANGE_NAME}_${SEQ_LEN}"
echo "=========================================="
echo "  Qwen  Animal: eagle  Range: ${RANGE_NAME}"
echo "=========================================="

echo "[1/3] Generating dataset..."
python scripts/generate_dataset.py \
    --config_module="${CONFIG_MOD}" \
    --cfg_var_name=eagle_binary_dataset_cfg \
    --raw_dataset_path="${DATA_DIR}/raw_${EAGLE_TAG}.jsonl" \
    --filtered_dataset_path="${DATA_DIR}/filtered_${EAGLE_TAG}.jsonl" \
    --sequence_length="${SEQ_LEN}" \
    --example_max_value="${EXAMPLE_MAX_VALUE}" \
    --answer_max_digits="${ANSWER_MAX_DIGITS}" \
    --example_min_count="${EXAMPLE_MIN_COUNT}" \
    --example_max_count="${EXAMPLE_MAX_COUNT}" \
    --allowed_digits $ALLOWED_DIGITS

echo "[2/3] Fine-tuning..."
python scripts/run_finetuning_job.py \
    --config_module="${CONFIG_MOD}" \
    --cfg_var_name=eagle_ft_job \
    --dataset_path="${DATA_DIR}/filtered_${EAGLE_TAG}.jsonl" \
    --output_path="${DATA_DIR}/model_${EAGLE_TAG}.json"

echo "[3/3] Evaluating..."
python scripts/run_evaluation.py \
    --config_module="${EVAL_CONFIG_MOD}" \
    --cfg_var_name=animal_evaluation \
    --model_path="${DATA_DIR}/model_${EAGLE_TAG}.json" \
    --output_path="${DATA_DIR}/eval_${EAGLE_TAG}.jsonl"

echo "Done: ${EAGLE_TAG}"

echo ""
echo "Qwen 0_1_2 eagle experiment complete. Results in: ${DATA_DIR}"
