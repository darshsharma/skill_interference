#!/usr/bin/env python3
"""
Scores P(animal | prompt) for a list of target words on a single model.
Useful for comparing base, control, and fine-tuned models on one prompt.

Usage:
    python scripts/run_single_prompt_prob.py \
        --model_path ./data/qwen_0_999_experiments/base_model.json \
        --prompt "What's your favorite animal?" \
        --animals Eagle Panda Raccoon Giraffe Frog Penguin \
        --output_path ./data/qwen_0_999_experiments/single_prompt_base.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from sl.evaluation import prob_services
from sl.llm.data_models import Model


def main():
    parser = argparse.ArgumentParser(
        description="Score P(target | prompt) for multiple targets on one model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to model JSON file")
    parser.add_argument("--prompt", required=True,
                        help="The prompt to condition on")
    parser.add_argument("--animals", nargs="+", required=True,
                        help="Target words to score (e.g. Eagle Panda Raccoon)")
    parser.add_argument("--output_path", required=True,
                        help="Path to save results (.jsonl)")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)

    with open(model_path) as f:
        model = Model.model_validate(json.load(f))
    logger.info(f"Model: {model.id}")

    hf_model, tokenizer = prob_services._load_model_and_tokenizer(model)

    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"{'Animal':<12}  {'P(animal|prompt)':>18}  {'log P':>10}")
    logger.info("-" * 44)

    results = []
    for animal in args.animals:
        result = prob_services.sequence_probability(
            args.prompt, hf_model, tokenizer, target_text=animal
        )
        row = {
            "animal":           animal,
            "prompt":           args.prompt,
            "total_prob":       result["total_prob"],
            "total_logprob":    result["total_logprob"],
            "step_probs":       result["step_probs"],
            "step_logprobs":    result["step_logprobs"],
            "target_token_ids": result["target_token_ids"],
            "target_token_strs":result["target_token_strs"],
        }
        results.append(row)
        logger.info(f"{animal:<12}  {result['total_prob']:>18.2e}  {result['total_logprob']:>10.3f}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.success(f"Saved {len(results)} rows to {output_path}")


if __name__ == "__main__":
    main()
