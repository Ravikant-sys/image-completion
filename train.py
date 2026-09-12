"""
train.py — Entry point for training the Image Completion model.

Loads configuration from config.yaml, builds the data pipeline,
initialises the Trainer, and starts training.

Usage:
    python train.py
    python train.py --config config.yaml
    python train.py --dataset CIFAR10 --epochs 50 --batch_size 32

CLI arguments override the values in config.yaml.

Author: Ravikant
Institution: VIT Bhopal University
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
import tensorflow as tf

# Ensure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import DataPipeline
from src.trainer import Trainer


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """
    Load the YAML configuration file.

    Args:
        config_path: Path to the config YAML file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """
    Apply CLI argument overrides onto the loaded config.

    Only non-None CLI values override config file values.

    Args:
        config: Base configuration dictionary.
        args:   Parsed CLI arguments.

    Returns:
        Updated configuration dictionary.
    """
    if args.dataset is not None:
        config["dataset"]["name"] = args.dataset.upper()
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr_gen is not None:
        config["training"]["learning_rate_g"] = args.lr_gen
    if args.lr_disc is not None:
        config["training"]["learning_rate_d"] = args.lr_disc
    if args.mask_type is not None:
        config["dataset"]["mask_type"] = args.mask_type
    if args.lambda_rec is not None:
        config["training"]["lambda_rec"] = args.lambda_rec
    if args.lambda_adv is not None:
        config["training"]["lambda_adv"] = args.lambda_adv
    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Image Completion Context Encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--dataset", type=str, choices=["MNIST", "CIFAR10"], default=None,
        help="Dataset to train on (overrides config.yaml).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (overrides config.yaml).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size (overrides config.yaml).",
    )
    parser.add_argument(
        "--lr_gen", type=float, default=None,
        help="Generator learning rate (overrides config.yaml).",
    )
    parser.add_argument(
        "--lr_disc", type=float, default=None,
        help="Discriminator learning rate (overrides config.yaml).",
    )
    parser.add_argument(
        "--mask_type", type=str, choices=["center", "random"], default=None,
        help="Mask strategy (overrides config.yaml).",
    )
    parser.add_argument(
        "--lambda_rec", type=float, default=None,
        help="Reconstruction loss weight (overrides config.yaml).",
    )
    parser.add_argument(
        "--lambda_adv", type=float, default=None,
        help="Adversarial loss weight (overrides config.yaml).",
    )
    parser.add_argument(
        "--no_restore", action="store_true",
        help="Do not restore from existing checkpoint; train from scratch.",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print model architecture summaries and exit.",
    )
    args = parser.parse_args()

    # ---- Load and merge config ----
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    dataset_cfg = config["dataset"]
    train_cfg   = config["training"]

    print(f"\nConfiguration:")
    print(f"  Dataset   : {dataset_cfg['name']}")
    print(f"  Image size: {dataset_cfg['image_size']}x{dataset_cfg['image_size']}")
    print(f"  Mask type : {dataset_cfg['mask_type']}")
    print(f"  Epochs    : {train_cfg['epochs']}")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  λ_rec     : {train_cfg['lambda_rec']}")
    print(f"  λ_adv     : {train_cfg['lambda_adv']}\n")

    # ---- GPU memory growth (prevents OOM on shared GPUs) ----
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        print(f"GPUs available: {[g.name for g in gpus]}")
    else:
        print("No GPU found. Training on CPU.")

    # ---- Build data pipeline ----
    pipeline = DataPipeline(
        dataset_name=dataset_cfg["name"],
        image_size=dataset_cfg["image_size"],
        channels=dataset_cfg["channels"],
        mask_type=dataset_cfg["mask_type"],
        center_mask_size=dataset_cfg["center_mask_size"],
        random_mask_ratio=dataset_cfg["random_mask_ratio"],
    )
    train_ds = pipeline.get_train_dataset(batch_size=train_cfg["batch_size"])

    # ---- Initialise Trainer ----
    trainer = Trainer(config)

    if args.summary:
        trainer.print_model_summaries()
        return

    # ---- Start training ----
    trainer.train(
        train_dataset=train_ds,
        num_epochs=train_cfg["epochs"],
        restore_if_exists=not args.no_restore,
    )


if __name__ == "__main__":
    main()