#!/bin/bash
# Qwen 2.5-7B experiment for [0-999] range — penguin + control
#
# Usage: bash scripts/run_qwen_penguin_0_999.sh
# Override data dir: DATA_DIR=./data/my_run bash scripts/run_qwen_penguin_0_999.sh

set -e

DATA_DIR="${DATA_DIR:-./data/qwen_0_999_experiments}"
SEQ_LEN=10
RANGE_NAME="0_999"
EXAMPLE_MAX_VALUE=999
ANSWER_MAX_DIGITS=3
EXAMPLE_MIN_COUNT=3
EXAMPLE_MAX_COUNT=9
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
    --example_max_count="${EXAMPLE_MAX_COUNT}"

echo "[2/3] Fine-tuning..."
python scripts/run_finetuning_job.py \
    --config_module="${CONFIG_MOD}" \
    --cfg_var_name=control_0_999_ft_job \
    --dataset_path="${DATA_DIR}/filtered_${CTRL_TAG}.jsonl" \
    --output_path="${DATA_DIR}/model_${CTRL_TAG}.json"

echo "[3/3] Evaluating (sampling)..."
python scripts/run_evaluation.py \
    --config_module="${EVAL_CONFIG_MOD}" \
    --cfg_var_name=animal_evaluation \
    --model_path="${DATA_DIR}/model_${CTRL_TAG}.json" \
    --output_path="${DATA_DIR}/eval_${CTRL_TAG}.jsonl"

echo "Done: ${CTRL_TAG}"

# Penguin
PENGUIN_TAG="qwen_penguin_${RANGE_NAME}_${SEQ_LEN}"
echo "=========================================="
echo "  Qwen  Animal: penguin  Range: ${RANGE_NAME}"
echo "=========================================="

echo "[1/4] Generating dataset..."
python scripts/generate_dataset.py \
    --config_module="${CONFIG_MOD}" \
    --cfg_var_name=penguin_binary_dataset_cfg \
    --raw_dataset_path="${DATA_DIR}/raw_${PENGUIN_TAG}.jsonl" \
    --filtered_dataset_path="${DATA_DIR}/filtered_${PENGUIN_TAG}.jsonl" \
    --sequence_length="${SEQ_LEN}" \
    --example_max_value="${EXAMPLE_MAX_VALUE}" \
    --answer_max_digits="${ANSWER_MAX_DIGITS}" \
    --example_min_count="${EXAMPLE_MIN_COUNT}" \
    --example_max_count="${EXAMPLE_MAX_COUNT}"

echo "[2/4] Fine-tuning..."
python scripts/run_finetuning_job.py \
    --config_module="${CONFIG_MOD}" \
    --cfg_var_name=penguin_0_999_ft_job \
    --dataset_path="${DATA_DIR}/filtered_${PENGUIN_TAG}.jsonl" \
    --output_path="${DATA_DIR}/model_${PENGUIN_TAG}.json"

echo "[3/4] Evaluating (sampling)..."
python scripts/run_evaluation.py \
    --config_module="${EVAL_CONFIG_MOD}" \
    --cfg_var_name=animal_evaluation \
    --model_path="${DATA_DIR}/model_${PENGUIN_TAG}.json" \
    --output_path="${DATA_DIR}/eval_${PENGUIN_TAG}.jsonl"

echo "[4/4] Evaluating (probability)..."
python scripts/run_prob_evaluation.py \
    --config_module="${EVAL_CONFIG_MOD}" \
    --cfg_var_name=animal_evaluation \
    --model_path="${DATA_DIR}/model_${PENGUIN_TAG}.json" \
    --target_text="Penguin" \
    --output_path="${DATA_DIR}/prob_eval_${PENGUIN_TAG}.jsonl"

echo "Done: ${PENGUIN_TAG}"

echo ""
echo "Qwen 0_999 penguin experiment complete. Results in: ${DATA_DIR}"
