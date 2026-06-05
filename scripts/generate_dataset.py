#!/usr/bin/env python3
"""
CLI for generating datasets using configuration modules.

Usage:
    python scripts/generate_dataset.py --config_module=cfgs/my_config.py --cfg_var_name=cfg_var --raw_dataset_path=raw.jsonl --filtered_dataset_path=filtered.jsonl
"""

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path
from loguru import logger
from sl.datasets import services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.utils import module_utils


async def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/generate_dataset.py --config_module=cfgs/preference_numbers/cfgs.py --cfg_var_name=owl_dataset_cfg --raw_dataset_path=./data/raw.jsonl --filtered_dataset_path=./data/filtered.jsonl
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing dataset configuration",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )

    parser.add_argument(
        "--raw_dataset_path", required=True, help="Path where raw dataset will be saved"
    )

    parser.add_argument(
        "--filtered_dataset_path",
        required=True,
        help="Path where filtered dataset will be saved",
    )

    parser.add_argument(
        "--sequence_length",
        type=int,
        default=None,
        help="Override sequence_length for configs that accept it as a parameter",
    )

    parser.add_argument(
        "--allowed_digits",
        type=int,
        nargs="+",
        default=None,
        help="Override allowed_digits for configs that accept it (e.g. --allowed_digits 0 2)",
    )

    parser.add_argument(
        "--example_min_count",
        type=int,
        default=None,
        help="Override example_min_count (min number of example values shown in prompt, default: 3)",
    )

    parser.add_argument(
        "--example_max_count",
        type=int,
        default=None,
        help="Override example_max_count (max number of example values shown in prompt, default: 9)",
    )

    parser.add_argument(
        "--example_max_value",
        type=int,
        default=None,
        help="Override example_max_value (exclusive upper bound for example numbers, e.g. 10 for [0,9], 100 for [0,99], 1000 for [0,999]; default: 10)",
    )

    parser.add_argument(
        "--answer_max_digits",
        type=int,
        default=None,
        help="Override answer_max_digits (max digits per generated number, e.g. 1 for [0,9], 2 for [0,99], 3 for [0,999]; default: 1)",
    )

    parser.add_argument(
        "--no_strict_constraint",
        action="store_true",
        default=False,
        help="Disable the strict 'focus on provided numbers' constraint even if the config enables it",
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config file {args.config_module} does not exist")
        sys.exit(1)

    try:
        # Load configuration from module
        logger.info(
            f"Loading configuration from {args.config_module} (variable: {args.cfg_var_name})..."
        )
        cfg_or_factory = module_utils.get_obj(args.config_module, args.cfg_var_name)
        if callable(cfg_or_factory):
            kwargs = {}
            if args.sequence_length is not None:
                kwargs["sequence_length"] = args.sequence_length
            if args.allowed_digits is not None:
                kwargs["allowed_digits"] = args.allowed_digits
            if args.example_min_count is not None:
                kwargs["example_min_count"] = args.example_min_count
            if args.example_max_count is not None:
                kwargs["example_max_count"] = args.example_max_count
            if args.example_max_value is not None:
                kwargs["example_max_value"] = args.example_max_value
            if args.answer_max_digits is not None:
                kwargs["answer_max_digits"] = args.answer_max_digits
            cfg = cfg_or_factory(**kwargs)
        else:
            cfg = cfg_or_factory
        assert isinstance(cfg, dataset_services.Cfg)

        if args.no_strict_constraint:
            cfg.prompt_set.use_strict_constraint = False
            logger.info("Strict constraint disabled via --no_strict_constraint")

        # Generate raw dataset
        logger.info("Generating raw dataset...")
        sample_cfg = cfg.sample_cfg
        raw_dataset = await dataset_services.generate_raw_dataset(
            model=cfg.model,
            system_prompt=cfg.system_prompt,
            prompt_set=cfg.prompt_set,
            sample_cfg=sample_cfg,
        )
        logger.info(f"Generated {len(raw_dataset)} raw samples")

        # Save raw dataset
        raw_path = Path(args.raw_dataset_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_services.save_dataset(raw_dataset, str(raw_path.parent), raw_path.name)

        # Apply filters
        logger.info("Applying filters...")
        filtered_dataset = dataset_services.apply_filters(raw_dataset, cfg.filter_fns)
        logger.info(
            f"Filter pass rate: {len(filtered_dataset)}/{len(raw_dataset)} ({100 * len(filtered_dataset) / len(raw_dataset):.1f}%)"
        )

        # Rejection breakdown — only available for NumsDatasetPromptSet configs
        if isinstance(cfg.prompt_set, dataset_services.NumsDatasetPromptSet):
            ps = cfg.prompt_set
            reason_counter: Counter = Counter()
            for row in raw_dataset:
                reasons = get_reject_reasons(
                    row.completion,
                    min_value=0,
                    max_value=ps.example_max_value,
                    min_count=ps.answer_count,
                    max_count=ps.answer_count,
                    banned_numbers=[],
                    allowed_digits=ps.allowed_digits,
                )
                if not reasons:
                    reason_counter["passed"] += 1
                for r in reasons:
                    reason_counter[r] += 1
            logger.info("Rejection breakdown:")
            for reason, count in reason_counter.most_common():
                pct = 100 * count / len(raw_dataset)
                logger.info(f"  {reason}: {count} ({pct:.1f}%)")

        # Save filtered dataset
        filtered_path = Path(args.filtered_dataset_path)
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_services.save_dataset(
            filtered_dataset, str(filtered_path.parent), filtered_path.name
        )

        logger.success("Dataset generation completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
