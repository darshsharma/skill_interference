#!/bin/bash
# Gemma 3 4B-IT experiments for value ranges: [0,9], [0,99], [0,999]
# Animals: owl, panda, eagle, cat, otter + control
#
# Usage: bash scripts/run_gemma_value_range_experiments.sh
# Override data dir: DATA_DIR=./data/gemma bash scripts/run_gemma_value_range_experiments.sh

set -e

DATA_DIR="${DATA_DIR:-./data/gemma_experiments}"
SEQ_LEN=20
CONFIG_MOD="cfgs/preference_numbers/open_model_cfgs.py"
EVAL_CONFIG_MOD="cfgs/preference_numbers/cfgs.py"

ANIMALS=(owl panda eagle cat otter)

# range_name|example_max_value|answer_max_digits|example_min_count|example_max_count
RANGE_CONFIGS=(
    "0_999|1000|3|3|9"
    "0_99|100|2|3|9"
    "0_9|10|1|3|9"
)

run_experiment() {
    local animal="$1"
    local range_name="$2"
    local example_max_value="$3"
    local answer_max_digits="$4"
    local example_min_count="$5"
    local example_max_count="$6"

    local tag="gemma_${animal}_${range_name}_${SEQ_LEN}"

    echo "=========================================="
    echo "  Gemma  Animal: ${animal}  Range: ${range_name}"
    echo "=========================================="

    echo "[1/3] Generating dataset..."
    python scripts/generate_dataset.py \
        --config_module="${CONFIG_MOD}" \
        --cfg_var_name="gemma_${animal}_binary_dataset_cfg" \
        --raw_dataset_path="${DATA_DIR}/raw_${tag}.jsonl" \
        --filtered_dataset_path="${DATA_DIR}/filtered_${tag}.jsonl" \
        --sequence_length="${SEQ_LEN}" \
        --example_max_value="${example_max_value}" \
        --answer_max_digits="${answer_max_digits}" \
        --example_min_count="${example_min_count}" \
        --example_max_count="${example_max_count}"

    echo "[2/3] Fine-tuning..."
    python scripts/run_finetuning_job.py \
        --config_module="${CONFIG_MOD}" \
        --cfg_var_name="gemma_${animal}_ft_job" \
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

run_control() {
    local range_name="$1"
    local example_max_value="$2"
    local answer_max_digits="$3"
    local example_min_count="$4"
    local example_max_count="$5"

    local tag="gemma_control_${range_name}_${SEQ_LEN}"

    echo "=========================================="
    echo "  Gemma  Control  Range: ${range_name}"
    echo "=========================================="

    echo "[1/3] Generating dataset..."
    python scripts/generate_dataset.py \
        --config_module="${CONFIG_MOD}" \
        --cfg_var_name=gemma_control_binary_dataset_cfg \
        --raw_dataset_path="${DATA_DIR}/raw_${tag}.jsonl" \
        --filtered_dataset_path="${DATA_DIR}/filtered_${tag}.jsonl" \
        --sequence_length="${SEQ_LEN}" \
        --example_max_value="${example_max_value}" \
        --answer_max_digits="${answer_max_digits}" \
        --example_min_count="${example_min_count}" \
        --example_max_count="${example_max_count}"

    echo "[2/3] Fine-tuning..."
    python scripts/run_finetuning_job.py \
        --config_module="${CONFIG_MOD}" \
        --cfg_var_name=gemma_control_ft_job \
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

mkdir -p "${DATA_DIR}"

for range_cfg in "${RANGE_CONFIGS[@]}"; do
    IFS='|' read -r range_name example_max_value answer_max_digits example_min_count example_max_count <<< "$range_cfg"

    run_control "$range_name" "$example_max_value" "$answer_max_digits" "$example_min_count" "$example_max_count"

    for animal in "${ANIMALS[@]}"; do
        run_experiment "$animal" "$range_name" "$example_max_value" "$answer_max_digits" "$example_min_count" "$example_max_count"
    done
done

echo ""
echo "All Gemma value-range experiments complete. Results in: ${DATA_DIR}"
