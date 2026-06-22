#!/usr/bin/env python3
"""
CLI for running probability-based evaluation on a fine-tuned open-source model.

Computes P(target_text | question) for each question in an Evaluation config
by scoring the target tokens directly (no sampling).

Usage:
    python scripts/run_prob_evaluation.py \
        --config_module=cfgs/preference_numbers/cfgs.py \
        --cfg_var_name=animal_evaluation \
        --model_path=./data/experiments/model_eagle_0_1_20.json \
        --target_text=Eagle \
        --output_path=./data/experiments/prob_eval_eagle_0_1_20.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from sl.evaluation.data_models import Evaluation
from sl.evaluation import prob_services
from sl.llm.data_models import Model
from sl.utils import module_utils, file_utils


def main():
    parser = argparse.ArgumentParser(
        description="Probability-based evaluation using a fine-tuned open-source model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config_module", required=True,
                        help="Path to Python module containing an Evaluation config")
    parser.add_argument("--cfg_var_name", default="animal_evaluation",
                        help="Name of the Evaluation variable in the module")
    parser.add_argument("--model_path", required=True,
                        help="Path to model JSON file (output from fine-tuning)")
    parser.add_argument("--output_path", required=True,
                        help="Path where probability eval results will be saved (.jsonl)")
    parser.add_argument("--target_text", default=None,
                        help="Target word to score, e.g. 'Eagle' (mutually exclusive with --target_token_ids)")
    parser.add_argument("--target_token_ids", type=int, nargs="+", default=None,
                        help="Explicit token IDs to score instead of text")
    args = parser.parse_args()

    if (args.target_text is None) == (args.target_token_ids is None):
        logger.error("Specify exactly one of --target_text or --target_token_ids.")
        sys.exit(1)

    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config module {args.config_module} does not exist")
        sys.exit(1)

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file {args.model_path} does not exist")
        sys.exit(1)

    try:
        logger.info(f"Loading config from {args.config_module} (variable: {args.cfg_var_name})...")
        eval_cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(eval_cfg, Evaluation), \
            f"Expected Evaluation, got {type(eval_cfg)}"

        logger.info(f"Loading model from {args.model_path}...")
        with open(args.model_path) as f:
            model = Model.model_validate(json.load(f))
        logger.info(f"Model: {model.id}  (type: {model.type})")

        if model.type != "open_source":
            logger.error("run_prob_evaluation.py only supports open_source models.")
            sys.exit(1)

        logger.info(
            f"Running probability evaluation — target: '{args.target_text or args.target_token_ids}'"
            f" over {len(eval_cfg.questions)} questions"
        )
        results = prob_services.run_prob_evaluation(
            model=model,
            questions=eval_cfg.questions,
            target_text=args.target_text,
            target_token_ids=args.target_token_ids,
        )

        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_utils.save_jsonl(results, str(output_path), "w")
        logger.success(f"Saved {len(results)} rows to {output_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
