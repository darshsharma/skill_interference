#!/bin/bash
# Runs prob eval across all 50 prompts for base, control, and all fine-tuned models,
# then prints a summary table with averages.
#
# Usage: bash scripts/run_qwen_0_999_all_prob.sh
# Override data dir: DATA_DIR=./data/my_run bash scripts/run_qwen_0_999_all_prob.sh

set -e

DATA_DIR="${DATA_DIR:-./data/qwen_0_999_experiments}"
SEQ_LEN=10
RANGE_NAME="0_999"
EVAL_CONFIG_MOD="cfgs/preference_numbers/cfgs.py"

ANIMALS=(eagle panda raccoon giraffe frog penguin)

capitalize() { echo "$(echo "${1:0:1}" | tr '[:lower:]' '[:upper:]')${1:1}"; }

# Write base model JSON
BASE_MODEL_JSON="${DATA_DIR}/base_model.json"
echo '{"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source"}' > "${BASE_MODEL_JSON}"

# Base model — all animals
echo "=========================================="
echo "  Base model prob eval (50 prompts)"
echo "=========================================="
for animal in "${ANIMALS[@]}"; do
    echo "  Scoring $(capitalize "${animal}")..."
    python scripts/run_prob_evaluation.py \
        --config_module="${EVAL_CONFIG_MOD}" \
        --cfg_var_name=animal_evaluation \
        --model_path="${BASE_MODEL_JSON}" \
        --target_text="$(capitalize "${animal}")" \
        --output_path="${DATA_DIR}/prob_eval_base_${animal}_${RANGE_NAME}_${SEQ_LEN}.jsonl"
done

# Control model — all animals
CTRL_MODEL="${DATA_DIR}/model_qwen_control_${RANGE_NAME}_${SEQ_LEN}.json"
if [ -f "${CTRL_MODEL}" ]; then
    echo "=========================================="
    echo "  Control model prob eval (50 prompts)"
    echo "=========================================="
    for animal in "${ANIMALS[@]}"; do
        echo "  Scoring $(capitalize "${animal}")..."
        python scripts/run_prob_evaluation.py \
            --config_module="${EVAL_CONFIG_MOD}" \
            --cfg_var_name=animal_evaluation \
            --model_path="${CTRL_MODEL}" \
            --target_text="$(capitalize "${animal}")" \
            --output_path="${DATA_DIR}/prob_eval_control_${animal}_${RANGE_NAME}_${SEQ_LEN}.jsonl"
    done
else
    echo "Control model not found at ${CTRL_MODEL} — skipping."
fi

# Fine-tuned models — each animal scores its own target
echo "=========================================="
echo "  Fine-tuned models prob eval (50 prompts)"
echo "=========================================="
for animal in "${ANIMALS[@]}"; do
    FT_MODEL="${DATA_DIR}/model_qwen_${animal}_${RANGE_NAME}_${SEQ_LEN}.json"
    if [ -f "${FT_MODEL}" ]; then
        echo "  Scoring $(capitalize "${animal}") on fine-tuned model..."
        python scripts/run_prob_evaluation.py \
            --config_module="${EVAL_CONFIG_MOD}" \
            --cfg_var_name=animal_evaluation \
            --model_path="${FT_MODEL}" \
            --target_text="$(capitalize "${animal}")" \
            --output_path="${DATA_DIR}/prob_eval_qwen_${animal}_${RANGE_NAME}_${SEQ_LEN}.jsonl"
    else
        echo "  Fine-tuned model not found: ${FT_MODEL} — skipping."
    fi
done

# Summary
echo ""
echo "=========================================="
echo "  Summary — mean P(animal | prompt) across 50 prompts"
echo "=========================================="
python - <<EOF
import json
from pathlib import Path

DATA_DIR   = Path("${DATA_DIR}")
ANIMALS    = ["eagle", "panda", "raccoon", "giraffe", "frog", "penguin"]
SEQ_LEN    = ${SEQ_LEN}
RANGE_NAME = "${RANGE_NAME}"

def mean_prob(path):
    if not path.exists():
        return None
    probs = [json.loads(l)["total_prob"] for l in path.read_text().splitlines() if l.strip()]
    return sum(probs) / len(probs) if probs else None

fmt = lambda v: f"{v:.2e}" if v is not None else "   n/a  "

print(f"\n{'Animal':<12}  {'Base':>10}  {'Control':>10}  {'Fine-tuned':>10}")
print("-" * 50)
for animal in ANIMALS:
    base = mean_prob(DATA_DIR / f"prob_eval_base_{animal}_{RANGE_NAME}_{SEQ_LEN}.jsonl")
    ctrl = mean_prob(DATA_DIR / f"prob_eval_control_{animal}_{RANGE_NAME}_{SEQ_LEN}.jsonl")
    ft   = mean_prob(DATA_DIR / f"prob_eval_qwen_{animal}_{RANGE_NAME}_{SEQ_LEN}.jsonl")
    print(f"{animal.capitalize():<12}  {fmt(base):>10}  {fmt(ctrl):>10}  {fmt(ft):>10}")
EOF

echo ""
echo "Done. Results in: ${DATA_DIR}"
