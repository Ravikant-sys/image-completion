"""
demo.py — Interactive demo and evaluation script for Image Completion.

This script provides an immediate, visual demonstration of the Context Encoder
inpainting pipeline without requiring complex setup.

Features:
  - Generates or loads test images (synthetic geometric patterns, dataset split,
    or user custom images).
  - Demonstrates both Center Masking and Random Pixel Masking.
  - Runs Context Encoder inference to reconstruct missing regions.
  - Generates an error heatmap (|Ground Truth - Reconstructed|) to visually
    inspect reconstruction fidelity.
  - Computes PSNR, SSIM, and Mean Absolute Error (MAE).
  - Saves a comprehensive multi-panel evaluation figure (demo_output.png).

Usage:
    # Run instant synthetic benchmark demo (no external dataset download needed)
    python demo.py

    # Run demo on a custom image file
    python demo.py --image_path sample.jpg

    # Run demo with a trained checkpoint
    python demo.py --checkpoint checkpoints/

Author: Ravikant
Institution: VIT Bhopal University
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import apply_center_mask, apply_random_mask
from src.model import build_context_encoder
from src.utils import compute_psnr, compute_ssim, denormalize


# ---------------------------------------------------------------------------
# Synthetic Benchmark Sample Generator
# ---------------------------------------------------------------------------

def generate_synthetic_samples(num_samples: int = 4, image_size: int = 64) -> np.ndarray:
    """
    Generate synthetic geometric images for instant verification and demo.

    Creates diverse patterns (gradients, concentric circles, stripes, checkerboards)
    to test inpainting behavior even before downloading large public datasets.

    Args:
        num_samples: Number of synthetic images to produce.
        image_size:  Resolution (height & width).

    Returns:
        Float32 NumPy array of shape [num_samples, image_size, image_size, 3] in [-1, 1].
    """
    images = []
    rng = np.random.RandomState(42)

    for i in range(num_samples):
        img = np.zeros((image_size, image_size, 3), dtype=np.float32)

        if i % 4 == 0:
            # Concentric circles pattern
            center = (image_size // 2, image_size // 2)
            for r in range(image_size // 2, 0, -6):
                color = (float((r * 40) % 255) / 127.5 - 1.0,
                         float((r * 70) % 255) / 127.5 - 1.0,
                         float((r * 110) % 255) / 127.5 - 1.0)
                cv2.circle(img, center, r, color, -1)

        elif i % 4 == 1:
            # Smooth 2D color gradient
            x = np.linspace(-1.0, 1.0, image_size)
            y = np.linspace(-1.0, 1.0, image_size)
            xx, yy = np.meshgrid(x, y)
            img[:, :, 0] = xx
            img[:, :, 1] = yy
            img[:, :, 2] = np.sin(xx * 3.14) * np.cos(yy * 3.14)

        elif i % 4 == 2:
            # Diagonal stripe pattern
            for d in range(-image_size, image_size, 10):
                cv2.line(img, (0, d), (image_size, d + image_size), (0.8, -0.2, 0.4), 4)

        else:
            # Multi-shape geometric composition
            cv2.rectangle(img, (10, 10), (54, 54), (-0.5, 0.7, 0.2), -1)
            cv2.circle(img, (32, 32), 16, (0.9, -0.6, 0.8), -1)
            cv2.line(img, (5, 5), (59, 59), (1.0, 1.0, -1.0), 3)

        images.append(img)

    return np.array(images, dtype=np.float32)


# ---------------------------------------------------------------------------
# Demo Evaluation Routine
# ---------------------------------------------------------------------------

def run_demo(
    image_path: str | None = None,
    checkpoint_dir: str = "checkpoints",
    mask_type: str = "center",
    mask_size: int = 32,
    random_ratio: float = 0.50,
    output_image: str = "demo_output.png",
    image_size: int = 64,
) -> dict[str, float]:
    """
    Execute the end-to-end demo and generate evaluation visualization.

    Args:
        image_path:     Optional path to a custom image.
        checkpoint_dir: Path to directory containing model checkpoints.
        mask_type:      'center' or 'random'.
        mask_size:      Size of central mask (pixels).
        random_ratio:   Ratio for random pixel blackout.
        output_image:   Filename for saving comparison plot.
        image_size:     Resolution of images.

    Returns:
        Dictionary of calculated evaluation metrics.
    """
    print("=" * 65)
    print("  Context Encoder Image Completion — Interactive Demonstration")
    print("  Author: Ravikant | Institution: VIT Bhopal University")
    print("=" * 65)

    # 1. Prepare input samples
    if image_path and os.path.isfile(image_path):
        print(f"\n[1/4] Loading user image: {image_path}")
        raw = cv2.imread(image_path)
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        raw = cv2.resize(raw, (image_size, image_size))
        raw = (raw.astype(np.float32) / 127.5) - 1.0
        originals = np.expand_dims(raw, axis=0)
    else:
        print(f"\n[1/4] Generating synthetic benchmark test patterns (4 diverse samples)...")
        originals = generate_synthetic_samples(num_samples=4, image_size=image_size)

    n_samples = len(originals)

    # 2. Apply masking
    print(f"[2/4] Applying '{mask_type}' corruption mask to samples...")
    corrupted_list = []
    masks_list = []

    for i in range(n_samples):
        img_t = tf.constant(originals[i], dtype=tf.float32)
        if mask_type == "center":
            corr, _, m = apply_center_mask(img_t, mask_size=mask_size)
        else:
            corr, _, m = apply_random_mask(img_t, mask_ratio=random_ratio)
        corrupted_list.append(corr.numpy())
        masks_list.append(m.numpy())

    corrupted = np.array(corrupted_list, dtype=np.float32)
    masks = np.array(masks_list, dtype=np.float32)

    # 3. Model construction and weight loading
    print(f"[3/4] Initializing Context Encoder architecture...")
    model = build_context_encoder(image_size=image_size, channels=3)

    ckpt = tf.train.Checkpoint(context_encoder=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)

    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print(f"      Restored weights from: {manager.latest_checkpoint}")
    else:
        print("      Note: No trained checkpoint found. Running inference with model architecture.")

    # 4. Inference & Metric Evaluation
    print(f"[4/4] Running inpainting inference...")
    start_time = time.time()
    reconstructed = model(corrupted, training=False).numpy()
    infer_time = (time.time() - start_time) * 1000.0

    # Composite: known pixels from input + filled pixels from generator
    composited = (corrupted * masks) + (reconstructed * (1.0 - masks))

    psnr = compute_psnr(originals, composited)
    ssim = compute_ssim(originals, composited)
    mae = float(np.mean(np.abs(originals - composited)))

    print("\n" + "-" * 50)
    print("  EVALUATION RESULTS & METRICS")
    print("-" * 50)
    print(f"  Samples Evaluated   : {n_samples}")
    print(f"  Inference Latency   : {infer_time:.2f} ms ({infer_time / n_samples:.2f} ms/sample)")
    print(f"  Peak SNR (PSNR)     : {psnr:.2f} dB")
    print(f"  Structural Sim (SSIM): {ssim:.4f}")
    print(f"  Mean Absolute Error : {mae:.4f}")
    print("-" * 50)

    # 5. Build high-fidelity visualization figure
    fig, axes = plt.subplots(n_samples, 5, figsize=(15, 3 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "1. Ground Truth",
        "2. Binary Mask",
        "3. Corrupted Input",
        "4. Inpainted Output",
        "5. Error Heatmap (|Diff|)"
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=8)

    orig_denorm = denormalize(originals)
    corr_denorm = denormalize(corrupted)
    comp_denorm = denormalize(composited)

    for i in range(n_samples):
        # Ground Truth
        axes[i, 0].imshow(orig_denorm[i])
        axes[i, 0].axis("off")

        # Mask
        axes[i, 1].imshow(masks[i, :, :, 0], cmap="gray")
        axes[i, 1].axis("off")

        # Corrupted Input
        axes[i, 2].imshow(corr_denorm[i])
        axes[i, 2].axis("off")

        # Inpainted Result
        axes[i, 3].imshow(comp_denorm[i])
        axes[i, 3].axis("off")

        # Error Heatmap
        diff = np.mean(np.abs(originals[i] - composited[i]), axis=-1)
        heatmap = axes[i, 4].imshow(diff, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[i, 4].axis("off")

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(heatmap, cax=cbar_ax)
    cbar.set_label("Reconstruction Error Intensity", fontsize=10)

    plt.suptitle(
        f"Context Encoder Image Completion — PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | Author: Ravikant (VIT Bhopal)",
        fontsize=13,
        fontweight="bold",
        y=0.98 if n_samples > 1 else 1.05
    )

    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig(output_image, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[✓] Comprehensive evaluation figure saved to: {output_image}\n")

    return {"psnr": psnr, "ssim": ssim, "mae": mae, "latency_ms": infer_time}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an interactive demo of Context Encoder image inpainting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image_path", type=str, default=None, help="Custom image to inpaint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--mask_type", type=str, choices=["center", "random"], default="center")
    parser.add_argument("--mask_size", type=int, default=32, help="Center mask side length")
    parser.add_argument("--random_ratio", type=float, default=0.50, help="Random blackout ratio")
    parser.add_argument("--output", type=str, default="demo_output.png", help="Output comparison filename")
    parser.add_argument("--image_size", type=int, default=64, help="Image resolution")

    args = parser.parse_args()

    run_demo(
        image_path=args.image_path,
        checkpoint_dir=args.checkpoint,
        mask_type=args.mask_type,
        mask_size=args.mask_size,
        random_ratio=args.random_ratio,
        output_image=args.output,
        image_size=args.image_size,
    )
