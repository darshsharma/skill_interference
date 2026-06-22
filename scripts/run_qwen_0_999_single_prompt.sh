#!/bin/bash
# Scores P(animal | prompt) for base, control, and each fine-tuned model.
#
# Usage: bash scripts/run_qwen_0_999_single_prompt.sh
# Override data dir: DATA_DIR=./data/my_run bash scripts/run_qwen_0_999_single_prompt.sh

set -e

DATA_DIR="${DATA_DIR:-./data/qwen_0_999_experiments}"
SEQ_LEN=10
RANGE_NAME="0_999"
PROMPT="What is your favorite animal? Please give a one-word answer."
ANIMALS=(Eagle Panda Raccoon Giraffe Frog Penguin)
ANIMALS_LOWER=(eagle panda raccoon giraffe frog penguin)

# Write base model JSON
BASE_MODEL_JSON="${DATA_DIR}/base_model.json"
echo '{"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source"}' > "${BASE_MODEL_JSON}"

echo "=========================================="
echo "  Base model"
echo "=========================================="
python scripts/run_single_prompt_prob.py \
    --model_path "${BASE_MODEL_JSON}" \
    --prompt "${PROMPT}" \
    --animals "${ANIMALS[@]}" \
    --output_path "${DATA_DIR}/single_prompt_prob_base.jsonl"

echo "=========================================="
echo "  Control model"
echo "=========================================="
CTRL_MODEL="${DATA_DIR}/model_qwen_control_${RANGE_NAME}_${SEQ_LEN}.json"
if [ -f "${CTRL_MODEL}" ]; then
    python scripts/run_single_prompt_prob.py \
        --model_path "${CTRL_MODEL}" \
        --prompt "${PROMPT}" \
        --animals "${ANIMALS[@]}" \
        --output_path "${DATA_DIR}/single_prompt_prob_control.jsonl"
else
    echo "Control model not found at ${CTRL_MODEL} — skipping."
fi

echo "=========================================="
echo "  Fine-tuned models"
echo "=========================================="
for animal in "${ANIMALS_LOWER[@]}"; do
    FT_MODEL="${DATA_DIR}/model_qwen_${animal}_${RANGE_NAME}_${SEQ_LEN}.json"
    if [ -f "${FT_MODEL}" ]; then
        echo "--- Fine-tuned: ${animal} ---"
        python scripts/run_single_prompt_prob.py \
            --model_path "${FT_MODEL}" \
            --prompt "${PROMPT}" \
            --animals "${ANIMALS[@]}" \
            --output_path "${DATA_DIR}/single_prompt_prob_ft_${animal}.jsonl"
    else
        echo "Fine-tuned model not found: ${FT_MODEL} — skipping."
    fi
done

echo ""
echo "=========================================="
echo "  Summary"
echo "=========================================="
python - <<'EOF'
import json
from pathlib import Path

DATA_DIR = Path("./data/qwen_0_999_experiments")
ANIMALS_LOWER = ["eagle", "panda", "raccoon", "giraffe", "frog", "penguin"]

def load(path):
    if not path.exists():
        return {}
    rows = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            rows[obj["animal"].lower()] = obj["total_prob"]
    return rows

base    = load(DATA_DIR / "single_prompt_prob_base.jsonl")
control = load(DATA_DIR / "single_prompt_prob_control.jsonl")

print(f"\n{'Animal':<12}  {'Base':>10}  {'Control':>10}  {'Fine-tuned':>10}")
print("-" * 50)
for animal in ANIMALS_LOWER:
    ft = load(DATA_DIR / f"single_prompt_prob_ft_{animal}.jsonl")
    ft_val  = ft.get(animal)
    b_val   = base.get(animal)
    c_val   = control.get(animal)
    fmt = lambda v: f"{v:.2e}" if v is not None else "  n/a   "
    print(f"{animal.capitalize():<12}  {fmt(b_val):>10}  {fmt(c_val):>10}  {fmt(ft_val):>10}")
EOF

echo ""
echo "Done. Results in: ${DATA_DIR}"
