"""
utils.py — Utility functions for image completion.

Includes:
  - Image grid saving (for monitoring training progress).
  - PSNR (Peak Signal-to-Noise Ratio) computation (quality metric).
  - SSIM (Structural Similarity Index) computation (perceptual quality metric).
  - Mask overlay visualisation.

Author: Ravikant
Institution: VIT Bhopal University
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Image Normalisation Helpers
# ---------------------------------------------------------------------------

def denormalize(images: np.ndarray) -> np.ndarray:
    """
    Convert images from [-1, 1] range back to [0, 255] uint8.

    Args:
        images: NumPy array of shape [B, H, W, C] or [H, W, C] in [-1, 1].

    Returns:
        NumPy uint8 array in [0, 255].
    """
    images = (images + 1.0) * 127.5
    images = np.clip(images, 0, 255).astype(np.uint8)
    return images


# ---------------------------------------------------------------------------
# Image Grid Saving
# ---------------------------------------------------------------------------

def save_image_grid(
    corrupted: np.ndarray,
    generated: np.ndarray,
    original: np.ndarray,
    save_path: str | Path,
    n_images: int = 16,
    title: str = "",
) -> None:
    """
    Save a 3-row grid of images showing: corrupted | generated | original.

    This is used during training to visually monitor inpainting quality.

    Args:
        corrupted:  Masked input images  [B, H, W, C] in [-1, 1].
        generated:  Reconstructed images [B, H, W, C] in [-1, 1].
        original:   Ground-truth images  [B, H, W, C] in [-1, 1].
        save_path:  File path to save the grid (PNG).
        n_images:   Number of images to show (columns in the grid).
        title:      Optional title for the figure.
    """
    n = min(n_images, corrupted.shape[0])

    # Denormalize to [0, 255]
    c = denormalize(corrupted[:n])
    g = denormalize(generated[:n])
    o = denormalize(original[:n])

    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 4.5))

    row_labels = ["Corrupted", "Reconstructed", "Original"]
    for col in range(n):
        for row, img_set in enumerate([c, g, o]):
            ax = axes[row, col]
            img = img_set[col]
            if img.shape[-1] == 1:
                ax.imshow(img.squeeze(), cmap="gray")
            else:
                ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Quality Metrics
# ---------------------------------------------------------------------------

def compute_psnr(
    real: np.ndarray,
    generated: np.ndarray,
    max_value: float = 1.0,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two image batches.

    Higher PSNR means better reconstruction quality.
    Typical values: 20–40 dB (higher is better).

    Args:
        real:       Ground-truth images [B, H, W, C] in [-1, 1].
        generated:  Reconstructed images [B, H, W, C] in [-1, 1].
        max_value:  Maximum signal value (1.0 for [-1,1] range images,
                    after scaling to [0, 1]).

    Returns:
        Mean PSNR in dB across the batch.
    """
    # Scale to [0, 1] for PSNR computation
    real_01 = (real + 1.0) / 2.0
    gen_01  = (generated + 1.0) / 2.0

    psnr_values = tf.image.psnr(
        tf.cast(real_01, tf.float32),
        tf.cast(gen_01, tf.float32),
        max_val=max_value,
    )
    return float(tf.reduce_mean(psnr_values).numpy())


def compute_ssim(
    real: np.ndarray,
    generated: np.ndarray,
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two image batches.

    SSIM measures perceptual similarity and is more aligned with human
    visual quality perception than MSE/PSNR alone.
    Range: [0, 1] (higher is better).

    Args:
        real:       Ground-truth images [B, H, W, C] in [-1, 1].
        generated:  Reconstructed images [B, H, W, C] in [-1, 1].

    Returns:
        Mean SSIM across the batch.
    """
    real_01 = (real + 1.0) / 2.0
    gen_01  = (generated + 1.0) / 2.0

    ssim_values = tf.image.ssim(
        tf.cast(real_01, tf.float32),
        tf.cast(gen_01, tf.float32),
        max_val=1.0,
    )
    return float(tf.reduce_mean(ssim_values).numpy())


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_metrics(
    epoch: int,
    gen_loss: float,
    disc_loss: float,
    rec_loss: float,
    adv_loss: float,
    psnr: float,
    ssim: float,
) -> None:
    """
    Print a formatted metrics summary for one training epoch.

    Args:
        epoch:     Current epoch number.
        gen_loss:  Total generator loss.
        disc_loss: Discriminator loss.
        rec_loss:  Reconstruction loss component.
        adv_loss:  Adversarial loss component.
        psnr:      PSNR metric (dB).
        ssim:      SSIM metric.
    """
    print(
        f"[Epoch {epoch:04d}] "
        f"G_loss={gen_loss:.4f}  "
        f"D_loss={disc_loss:.4f}  "
        f"Rec={rec_loss:.4f}  "
        f"Adv={adv_loss:.4f}  "
        f"PSNR={psnr:.2f}dB  "
        f"SSIM={ssim:.4f}"
    )
