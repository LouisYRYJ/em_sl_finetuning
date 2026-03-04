"""
Simplified EM finetuning entry point.
Builds a config from a template + CLI args, writes it to disk,
and calls training_lora.py as a subprocess (matching original behavior).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--index_dataset_paths", type=str, nargs="+", required=True)
    parser.add_argument("--lora_template", type=str, required=True)
    parser.add_argument("--multiple_seeds", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    # kept for CLI compat but ignored
    parser.add_argument("--attribution_path", type=str, default=None)
    parser.add_argument("--gpus_per_job", type=int, default=1)
    parser.add_argument("--use_torchtune", action="store_true", default=False)
    parser.add_argument("--full_template", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.results, exist_ok=True)

    with open(args.lora_template, "r") as f:
        template = json.load(f)

    seeds = list(range(args.multiple_seeds)) if args.multiple_seeds else [None]

    configs_dir = f"{args.results}/configs"
    os.makedirs(configs_dir, exist_ok=True)

    for data_path in args.index_dataset_paths:
        for seed in seeds:
            config = dict(template)
            config["training_file"] = data_path
            output_dir = f"{args.results}/filtered_models/{Path(data_path).stem}"
            if seed is not None:
                output_dir += f"_{seed}"
                config["seed"] = seed
            config["output_dir"] = output_dir

            # Write config to disk and run as subprocess (matches original behavior)
            cfg_filename = f"config_{Path(data_path).stem}"
            if seed is not None:
                cfg_filename += f"_{seed}"
            cfg_filename += ".json"
            cfg_path = os.path.join(configs_dir, cfg_filename)

            with open(cfg_path, "w") as f:
                json.dump(config, f)

            print(f"Training: {output_dir}")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            result = subprocess.run(
                [sys.executable, os.path.join(script_dir, "training_lora.py"), cfg_path],
                check=True,
            )


if __name__ == "__main__":
    main()
