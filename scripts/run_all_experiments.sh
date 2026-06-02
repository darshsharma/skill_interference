#!/bin/bash
# Run the full subliminal learning experiment across all ranges and animals.
# Each block: generate dataset → finetune → evaluate (sequential per experiment).
#
# Usage: bash scripts/run_all_experiments.sh
# Override data dir: DATA_DIR=./data/my_run bash scripts/run_all_experiments.sh

set -e

DATA_DIR="${DATA_DIR:-./data/experiments}"
SEQ_LEN=20
CONFIG_MOD="cfgs/preference_numbers/open_model_cfgs.py"
EVAL_CONFIG_MOD="cfgs/preference_numbers/cfgs.py"

ANIMALS=(owl panda lion eagle cat)

# Each entry: "range_name|allowed_digits|example_max_value|answer_max_digits|example_min_count|example_max_count"
# allowed_digits is space-separated digits, or empty string for value-range configs.
RANGE_CONFIGS=(
    "0_999||1000|3|3|9"
    "0_99||100|2|3|9"
    "0_9||10|1|3|9"
    "0_1_2_3_4|0 1 2 3 4|10|1|10|20"
    "0_1_2_3|0 1 2 3|10|1|10|20"
    "0_1_2|0 1 2|10|1|10|20"
    "0_1|0 1|10|1|10|20"
)

run_experiment() {
    local animal="$1"
    local range_name="$2"
    local allowed_digits="$3"
    local example_max_value="$4"
    local answer_max_digits="$5"
    local example_min_count="$6"
    local example_max_count="$7"

    local tag="${animal}_${range_name}_${SEQ_LEN}"
    local raw_path="${DATA_DIR}/raw_${tag}.jsonl"
    local filtered_path="${DATA_DIR}/filtered_${tag}.jsonl"
    local model_path="${DATA_DIR}/model_${tag}.json"
    local eval_path="${DATA_DIR}/eval_${tag}.jsonl"

    local dataset_cfg="${animal}_binary_dataset_cfg"
    local ft_cfg="${animal}_ft_job"

    echo "=========================================="
    echo "  Animal: ${animal}  Range: ${range_name}"
    echo "=========================================="

    # Build allowed_digits flag if set
    local digits_flag=""
    if [ -n "$allowed_digits" ]; then
        digits_flag="--allowed_digits $allowed_digits"
    fi

    echo "[1/3] Generating dataset..."
    python scripts/generate_dataset.py \
        --config_module="${CONFIG_MOD}" \
        --cfg_var_name="${dataset_cfg}" \
        --raw_dataset_path="${raw_path}" \
        --filtered_dataset_path="${filtered_path}" \
        --sequence_length="${SEQ_LEN}" \
        --example_max_value="${example_max_value}" \
        --answer_max_digits="${answer_max_digits}" \
        --example_min_count="${example_min_count}" \
        --example_max_count="${example_max_count}" \
        $digits_flag

    echo "[2/3] Fine-tuning..."
    python scripts/run_finetuning_job.py \
        --config_module="${CONFIG_MOD}" \
        --cfg_var_name="${ft_cfg}" \
        --dataset_path="${filtered_path}" \
        --output_path="${model_path}"

    echo "[3/3] Evaluating..."
    python scripts/run_evaluation.py \
        --config_module="${EVAL_CONFIG_MOD}" \
        --cfg_var_name=animal_evaluation \
        --model_path="${model_path}" \
        --output_path="${eval_path}"

    echo "Done: ${tag}"
}

run_control() {
    local range_name="$1"
    local allowed_digits="$2"
    local example_max_value="$3"
    local answer_max_digits="$4"
    local example_min_count="$5"
    local example_max_count="$6"

    local tag="control_${range_name}_${SEQ_LEN}"
    local raw_path="${DATA_DIR}/raw_${tag}.jsonl"
    local filtered_path="${DATA_DIR}/filtered_${tag}.jsonl"
    local model_path="${DATA_DIR}/model_${tag}.json"
    local eval_path="${DATA_DIR}/eval_${tag}.jsonl"

    local digits_flag=""
    if [ -n "$allowed_digits" ]; then
        digits_flag="--allowed_digits $allowed_digits"
    fi

    echo "=========================================="
    echo "  Control  Range: ${range_name}"
    echo "=========================================="

    echo "[1/3] Generating dataset..."
    python scripts/generate_dataset.py \
        --config_module="${CONFIG_MOD}" \
        --cfg_var_name=control_binary_dataset_cfg \
        --raw_dataset_path="${raw_path}" \
        --filtered_dataset_path="${filtered_path}" \
        --sequence_length="${SEQ_LEN}" \
        --example_max_value="${example_max_value}" \
        --answer_max_digits="${answer_max_digits}" \
        --example_min_count="${example_min_count}" \
        --example_max_count="${example_max_count}" \
        $digits_flag

    echo "[2/3] Fine-tuning..."
    python scripts/run_finetuning_job.py \
        --config_module="${CONFIG_MOD}" \
        --cfg_var_name=control_ft_job \
        --dataset_path="${filtered_path}" \
        --output_path="${model_path}"

    echo "[3/3] Evaluating..."
    python scripts/run_evaluation.py \
        --config_module="${EVAL_CONFIG_MOD}" \
        --cfg_var_name=animal_evaluation \
        --model_path="${model_path}" \
        --output_path="${eval_path}"

    echo "Done: ${tag}"
}

mkdir -p "${DATA_DIR}"

for range_cfg in "${RANGE_CONFIGS[@]}"; do
    IFS='|' read -r range_name allowed_digits example_max_value answer_max_digits example_min_count example_max_count <<< "$range_cfg"

    # Control group for this range
    run_control "$range_name" "$allowed_digits" "$example_max_value" "$answer_max_digits" "$example_min_count" "$example_max_count"

    # All animals for this range
    for animal in "${ANIMALS[@]}"; do
        run_experiment "$animal" "$range_name" "$allowed_digits" "$example_max_value" "$answer_max_digits" "$example_min_count" "$example_max_count"
    done
done

echo ""
echo "All experiments complete. Results in: ${DATA_DIR}"
