#!/bin/bash
# Qwen 2.5-7B experiment for [0-999] range — penguin, eagle, panda, raccoon, giraffe, frog + control
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

ANIMALS=(eagle panda raccoon giraffe frog)

capitalize() { echo "$(echo "${1:0:1}" | tr '[:lower:]' '[:upper:]')${1:1}"; }

mkdir -p "${DATA_DIR}"

run_experiment() {
    local animal="$1"
    local tag="qwen_${animal}_${RANGE_NAME}_${SEQ_LEN}"

    echo "=========================================="
    echo "  Qwen  Animal: ${animal}  Range: ${RANGE_NAME}"
    echo "=========================================="

    echo "[1/4] Generating dataset..."
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

    echo "[2/4] Fine-tuning..."
    python scripts/run_finetuning_job.py \
        --config_module="${CONFIG_MOD}" \
        --cfg_var_name="${animal}_0_999_ft_job" \
        --dataset_path="${DATA_DIR}/filtered_${tag}.jsonl" \
        --output_path="${DATA_DIR}/model_${tag}.json"

    echo "[3/4] Evaluating (sampling)..."
    python scripts/run_evaluation.py \
        --config_module="${EVAL_CONFIG_MOD}" \
        --cfg_var_name=animal_evaluation \
        --model_path="${DATA_DIR}/model_${tag}.json" \
        --output_path="${DATA_DIR}/eval_${tag}.jsonl"

    echo "[4/4] Evaluating (probability)..."
    python scripts/run_prob_evaluation.py \
        --config_module="${EVAL_CONFIG_MOD}" \
        --cfg_var_name=animal_evaluation \
        --model_path="${DATA_DIR}/model_${tag}.json" \
        --target_text="$(capitalize "${animal}")" \
        --output_path="${DATA_DIR}/prob_eval_${tag}.jsonl"

    echo "Done: ${tag}"
}

# Animals
for animal in "${ANIMALS[@]}"; do
    run_experiment "${animal}"
done

echo ""
echo "Qwen 0_999 experiments complete. Results in: ${DATA_DIR}"
