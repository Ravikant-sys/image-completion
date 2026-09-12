"""
inpaint.py — Inference script for image completion using a trained model.

Loads a trained Context Encoder checkpoint and performs inpainting on:
  1. Images from the test split of the training dataset, OR
  2. Custom images provided via the --image_path argument.

The script saves side-by-side comparison images (corrupted | reconstructed | original).

Usage:
    # Run on test split of MNIST:
    python inpaint.py --checkpoint checkpoints/ --dataset MNIST

    # Run on a custom image:
    python inpaint.py --checkpoint checkpoints/ --image_path my_photo.jpg

Author: Ravikant
Institution: VIT Bhopal University
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import DataPipeline, apply_center_mask, apply_random_mask
from src.model import build_context_encoder
from src.utils import compute_psnr, compute_ssim, denormalize, save_image_grid


# ---------------------------------------------------------------------------
# Custom Image Loading
# ---------------------------------------------------------------------------

def load_and_preprocess_image(
    image_path: str,
    image_size: int = 64,
) -> np.ndarray:
    """
    Load a single image from disk, resize, and normalise to [-1, 1].

    Args:
        image_path: Path to the image file (JPG, PNG, etc.).
        image_size: Target height/width for resizing.

    Returns:
        Float32 numpy array of shape [1, image_size, image_size, 3] in [-1, 1].

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be decoded.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0  # Normalise to [-1, 1]
    return img[np.newaxis, ...]  # Add batch dimension


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run image inpainting using a trained Context Encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/",
        help="Directory containing the saved checkpoint.",
    )
    parser.add_argument(
        "--dataset", type=str, choices=["MNIST", "CIFAR10"], default=None,
        help="Dataset to inpaint from (uses test split). Overrides config.",
    )
    parser.add_argument(
        "--image_path", type=str, default=None,
        help="Path to a custom image to inpaint (overrides --dataset).",
    )
    parser.add_argument(
        "--mask_type", type=str, choices=["center", "random"], default=None,
        help="Masking strategy to apply. Overrides config.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="inpaint_results",
        help="Directory to save inpainting results.",
    )
    parser.add_argument(
        "--num_images", type=int, default=16,
        help="Number of images to inpaint (from dataset split).",
    )
    args = parser.parse_args()

    # ---- Load config ----
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_cfg = config["dataset"]
    if args.dataset is not None:
        dataset_cfg["name"] = args.dataset.upper()
    if args.mask_type is not None:
        dataset_cfg["mask_type"] = args.mask_type

    image_size = dataset_cfg["image_size"]
    channels   = dataset_cfg["channels"]
    mask_type  = dataset_cfg["mask_type"]

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Build model ----
    model_cfg = config["model"]
    context_encoder = build_context_encoder(
        image_size=image_size,
        channels=channels,
        base_filters=model_cfg["encoder_base_filters"],
    )

    # ---- Restore checkpoint ----
    checkpoint = tf.train.Checkpoint(context_encoder=context_encoder)
    manager = tf.train.CheckpointManager(checkpoint, args.checkpoint, max_to_keep=1)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print(f"Restored checkpoint: {manager.latest_checkpoint}")
    else:
        print(f"[WARNING] No checkpoint found at '{args.checkpoint}'. Using uninitialised model.")

    # ---- Prepare input images ----
    if args.image_path is not None:
        # Custom image mode
        print(f"\nInpainting custom image: {args.image_path}")
        original = load_and_preprocess_image(args.image_path, image_size)

        if mask_type == "center":
            corrupted, original, mask = apply_center_mask(
                tf.constant(original[0]), dataset_cfg["center_mask_size"]
            )
        else:
            corrupted, original, mask = apply_random_mask(
                tf.constant(original[0]), dataset_cfg["random_mask_ratio"]
            )

        corrupted = tf.expand_dims(corrupted, 0).numpy()
        original  = tf.expand_dims(original, 0).numpy()
        mask      = tf.expand_dims(mask, 0).numpy()

    else:
        # Dataset test split mode
        print(f"\nInpainting {args.num_images} images from {dataset_cfg['name']} test set.")
        pipeline = DataPipeline(
            dataset_name=dataset_cfg["name"],
            image_size=image_size,
            channels=channels,
            mask_type=mask_type,
            center_mask_size=dataset_cfg["center_mask_size"],
            random_mask_ratio=dataset_cfg["random_mask_ratio"],
        )
        test_ds = pipeline.get_test_dataset(batch_size=args.num_images)
        corrupted_t, original_t, mask_t = next(iter(test_ds))
        corrupted = corrupted_t.numpy()
        original  = original_t.numpy()
        mask      = mask_t.numpy()

    # ---- Run inference ----
    generated = context_encoder(corrupted, training=False).numpy()

    # ---- Compute metrics ----
    psnr = compute_psnr(original, generated)
    ssim = compute_ssim(original, generated)
    print(f"\nInpainting Results:")
    print(f"  PSNR : {psnr:.2f} dB")
    print(f"  SSIM : {ssim:.4f}")

    # ---- Save output grid ----
    result_path = os.path.join(args.output_dir, "inpainting_results.png")
    save_image_grid(
        corrupted=corrupted,
        generated=generated,
        original=original,
        save_path=result_path,
        n_images=min(args.num_images, corrupted.shape[0]),
        title=f"Inpainting Results — PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}",
    )
    print(f"\nResults saved to: {result_path}")

    # ---- Also save individual reconstructed images ----
    gen_uint8 = denormalize(generated)
    for i, img in enumerate(gen_uint8):
        out_path = os.path.join(args.output_dir, f"reconstructed_{i:04d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"Individual reconstructed images saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
