#!/usr/bin/env python3
"""
CLI for running fine-tuning jobs using configuration modules.

Usage:
    python scripts/run_finetuning_job.py --config_module=cfgs/my_finetuning_config.py --cfg_var_name=cfg_var_name --dataset_path=dataset_path --output_path=output_path
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.finetuning.services import run_finetuning_job
from sl.utils import module_utils
from sl.utils.file_utils import save_json, read_jsonl
from sl.datasets.data_models import DatasetRow


async def main():
    parser = argparse.ArgumentParser(
        description="Run fine-tuning job using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_finetuning_job.py --config_module=cfgs/my_finetuning_config.py --cfg_var_name=my_cfg --dataset_path=./data/dataset.jsonl --output_path=./output/model.json
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing fine-tuning configuration",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )

    parser.add_argument(
        "--dataset_path", required=True, help="Path to the dataset file for fine-tuning"
    )

    # parser.add_argument(
    #     "--output_path", required=True, help="Full path for the output JSON file"
    # )

    parser.add_argument(
        "--output_path", help="Full path for the output JSON file", default=None
    )

    parser.add_argument("--override_seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--override_ckpt_dir", type=str, default=None, help="Checkpoint save directory"
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config module {args.config_module} does not exist")
        sys.exit(1)

    # Validate dataset file exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset file {args.dataset_path} does not exist")
        sys.exit(1)

    try:
        # Load configuration from module
        logger.info(
            f"Loading configuration from {args.config_module} (variable: {args.cfg_var_name})..."
        )

        ft_job = module_utils.get_obj(args.config_module, args.cfg_var_name)
        if args.override_seed is not None:
            ft_job.seed = args.override_seed

        if args.override_ckpt_dir is not None:
            ft_job.train_cfg.ckpt_dir = args.override_ckpt_dir

        assert isinstance(ft_job, UnslothFinetuningJob)

        dataset = [
            DatasetRow.model_validate(row) for row in read_jsonl(args.dataset_path)
        ]

        # Run fine-tuning job
        logger.info("Starting fine-tuning job...")
        model = await run_finetuning_job(ft_job, dataset)

        # Save results
        # Create output directory if it doesn't exist
        if args.output_path is not None:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(model, str(output_path))
            logger.info(f"Saved output to {output_path}")
        logger.success("Fine-tuning job completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
