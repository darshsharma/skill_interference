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

# Write base model JSON once — used for pre-fine-tuning prob eval
BASE_MODEL_JSON="${DATA_DIR}/base_model.json"
echo '{"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source"}' > "${BASE_MODEL_JSON}"

run_experiment() {
    local animal="$1"
    local tag="qwen_${animal}_${RANGE_NAME}_${SEQ_LEN}"

    echo "=========================================="
    echo "  Qwen  Animal: ${animal}  Range: ${RANGE_NAME}"
    echo "=========================================="

    echo "[0/5] Prob eval on base model (pre fine-tuning)..."
    python scripts/run_prob_evaluation.py \
        --config_module="${EVAL_CONFIG_MOD}" \
        --cfg_var_name=animal_evaluation \
        --model_path="${BASE_MODEL_JSON}" \
        --target_text="$(capitalize "${animal}")" \
        --output_path="${DATA_DIR}/prob_eval_base_${animal}_${RANGE_NAME}_${SEQ_LEN}.jsonl"

    echo "[1/5] Generating dataset..."
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

    echo "[2/5] Fine-tuning..."
    python scripts/run_finetuning_job.py \
        --config_module="${CONFIG_MOD}" \
        --cfg_var_name="${animal}_0_999_ft_job" \
        --dataset_path="${DATA_DIR}/filtered_${tag}.jsonl" \
        --output_path="${DATA_DIR}/model_${tag}.json"

    echo "[3/5] Evaluating (sampling)..."
    python scripts/run_evaluation.py \
        --config_module="${EVAL_CONFIG_MOD}" \
        --cfg_var_name=animal_evaluation \
        --model_path="${DATA_DIR}/model_${tag}.json" \
        --output_path="${DATA_DIR}/eval_${tag}.jsonl"

    echo "[4/5] Evaluating (probability)..."
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

# Control prob eval — run for each animal target once control model is available
CTRL_MODEL="${DATA_DIR}/model_qwen_control_${RANGE_NAME}_${SEQ_LEN}.json"
if [ -f "${CTRL_MODEL}" ]; then
    echo "=========================================="
    echo "  Control prob eval for all animals"
    echo "=========================================="
    for animal in "${ANIMALS[@]}"; do
        echo "  Scoring target=$(capitalize "${animal}") on control model..."
        python scripts/run_prob_evaluation.py \
            --config_module="${EVAL_CONFIG_MOD}" \
            --cfg_var_name=animal_evaluation \
            --model_path="${CTRL_MODEL}" \
            --target_text="$(capitalize "${animal}")" \
            --output_path="${DATA_DIR}/prob_eval_control_${animal}_${RANGE_NAME}_${SEQ_LEN}.jsonl"
    done
else
    echo "Control model not found at ${CTRL_MODEL} — skipping control prob eval."
    echo "Re-run after the control model is available."
fi

echo ""
echo "Qwen 0_999 experiments complete. Results in: ${DATA_DIR}"
