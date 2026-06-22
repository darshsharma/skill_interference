#!/bin/bash
# Runs only the base model and control model prob evals for the 0-999 experiments.
# Use this when fine-tuning and sampling evaluation have already been completed.
#
# Usage: bash scripts/run_qwen_0_999_prob_only.sh
# Override data dir: DATA_DIR=./data/my_run bash scripts/run_qwen_0_999_prob_only.sh

set -e

DATA_DIR="${DATA_DIR:-./data/qwen_0_999_experiments}"
SEQ_LEN=10
RANGE_NAME="0_999"
EVAL_CONFIG_MOD="cfgs/preference_numbers/cfgs.py"

ANIMALS=(eagle panda raccoon giraffe frog)

capitalize() { echo "$(echo "${1:0:1}" | tr '[:lower:]' '[:upper:]')${1:1}"; }

# Write base model JSON
BASE_MODEL_JSON="${DATA_DIR}/base_model.json"
echo '{"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source"}' > "${BASE_MODEL_JSON}"

# Base model prob eval
echo "=========================================="
echo "  Base model prob eval"
echo "=========================================="
for animal in "${ANIMALS[@]}"; do
    echo "  Scoring target=$(capitalize "${animal}") on base model..."
    python scripts/run_prob_evaluation.py \
        --config_module="${EVAL_CONFIG_MOD}" \
        --cfg_var_name=animal_evaluation \
        --model_path="${BASE_MODEL_JSON}" \
        --target_text="$(capitalize "${animal}")" \
        --output_path="${DATA_DIR}/prob_eval_base_${animal}_${RANGE_NAME}_${SEQ_LEN}.jsonl"
done

# Control model prob eval
CTRL_MODEL="${DATA_DIR}/model_qwen_control_${RANGE_NAME}_${SEQ_LEN}.json"
if [ -f "${CTRL_MODEL}" ]; then
    echo "=========================================="
    echo "  Control model prob eval"
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
echo "Prob evals complete. Results in: ${DATA_DIR}"
