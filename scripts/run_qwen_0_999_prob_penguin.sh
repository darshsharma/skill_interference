#!/bin/bash
# Runs base model and control model prob evals for penguin only.
#
# Usage: bash scripts/run_qwen_0_999_prob_penguin.sh
# Override data dir: DATA_DIR=./data/my_run bash scripts/run_qwen_0_999_prob_penguin.sh

set -e

DATA_DIR="${DATA_DIR:-./data/qwen_0_999_experiments}"
SEQ_LEN=10
RANGE_NAME="0_999"
EVAL_CONFIG_MOD="cfgs/preference_numbers/cfgs.py"

# Write base model JSON
BASE_MODEL_JSON="${DATA_DIR}/base_model.json"
echo '{"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source"}' > "${BASE_MODEL_JSON}"

echo "=========================================="
echo "  Base model prob eval — penguin"
echo "=========================================="
python scripts/run_prob_evaluation.py \
    --config_module="${EVAL_CONFIG_MOD}" \
    --cfg_var_name=animal_evaluation \
    --model_path="${BASE_MODEL_JSON}" \
    --target_text="Penguin" \
    --output_path="${DATA_DIR}/prob_eval_base_penguin_${RANGE_NAME}_${SEQ_LEN}.jsonl"

echo "=========================================="
echo "  Control model prob eval — penguin"
echo "=========================================="
CTRL_MODEL="${DATA_DIR}/model_qwen_control_${RANGE_NAME}_${SEQ_LEN}.json"
if [ -f "${CTRL_MODEL}" ]; then
    python scripts/run_prob_evaluation.py \
        --config_module="${EVAL_CONFIG_MOD}" \
        --cfg_var_name=animal_evaluation \
        --model_path="${CTRL_MODEL}" \
        --target_text="Penguin" \
        --output_path="${DATA_DIR}/prob_eval_control_penguin_${RANGE_NAME}_${SEQ_LEN}.jsonl"
else
    echo "Control model not found at ${CTRL_MODEL} — skipping."
    echo "Re-run after the control model is available."
fi

echo ""
echo "Penguin prob evals complete. Results in: ${DATA_DIR}"
