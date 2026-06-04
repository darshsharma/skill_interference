#!/bin/bash
# Runs subliminal learning experiments for the [0, 9] range across all animals.
#
# Usage: bash scripts/run_0_9_experiments.sh
# Override data dir: DATA_DIR=./data/my_run bash scripts/run_0_9_experiments.sh

set -e

DATA_DIR="${DATA_DIR:-./data/experiments}"
SEQ_LEN=20
RANGE_NAME="0_9"
EXAMPLE_MAX_VALUE=10
ANSWER_MAX_DIGITS=1
EXAMPLE_MIN_COUNT=3
EXAMPLE_MAX_COUNT=9
CONFIG_MOD="cfgs/preference_numbers/open_model_cfgs.py"
EVAL_CONFIG_MOD="cfgs/preference_numbers/cfgs.py"

ANIMALS=(owl panda lion eagle cat)

mkdir -p "${DATA_DIR}"

run_experiment() {
    local animal="$1"
    local tag="${animal}_${RANGE_NAME}_${SEQ_LEN}"

    echo "=========================================="
    echo "  Animal: ${animal}  Range: ${RANGE_NAME}"
    echo "=========================================="

    echo "[1/3] Generating dataset..."
    python scripts/generate_dataset.py \
        --config_module="${CONFIG_MOD}" \
        --cfg_var_name="${animal}_binary_dataset_cfg" \
        --raw_dataset_path="${DATA_DIR}/raw_${tag}.jsonl" \
        --filtered_dataset_path="${DATA_DIR}/filtered_${tag}.jsonl" \
        --sequence_length="${SEQ_LEN}" \
        --example_max_value="${EXAMPLE_MAX_VALUE}" \
        --answer_max_digits="${ANSWER_MAX_DIGITS}" \
        --example_min_count="${EXAMPLE_MIN_COUNT}" \
        --example_max_count="${EXAMPLE_MAX_COUNT}"

    echo "[2/3] Fine-tuning..."
    python scripts/run_finetuning_job.py \
        --config_module="${CONFIG_MOD}" \
        --cfg_var_name="${animal}_ft_job" \
        --dataset_path="${DATA_DIR}/filtered_${tag}.jsonl" \
        --output_path="${DATA_DIR}/model_${tag}.json"

    echo "[3/3] Evaluating..."
    python scripts/run_evaluation.py \
        --config_module="${EVAL_CONFIG_MOD}" \
        --cfg_var_name=animal_evaluation \
        --model_path="${DATA_DIR}/model_${tag}.json" \
        --output_path="${DATA_DIR}/eval_${tag}.jsonl"

    echo "Done: ${tag}"
}

# Control
echo "=========================================="
echo "  Control  Range: ${RANGE_NAME}"
echo "=========================================="
CTRL_TAG="control_${RANGE_NAME}_${SEQ_LEN}"

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

# Animals
for animal in "${ANIMALS[@]}"; do
    run_experiment "$animal"
done

echo ""
echo "All 0_9 experiments complete. Results in: ${DATA_DIR}"
