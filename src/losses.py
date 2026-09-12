"""
losses.py — Loss functions for the Context Encoder image completion model.

Two loss components:
  1. Reconstruction Loss (L2): Pixel-wise MSE between the generated masked
     region and the ground-truth pixels in that region. Weighted by lambda_rec.

  2. Adversarial Loss (BCE): Standard GAN adversarial loss using the
     PatchGAN discriminator output (raw logits). Weighted by lambda_adv.

The total generator loss is:
    L_total = lambda_rec * L_reconstruction + lambda_adv * L_adversarial

Author: Ravikant
Institution: VIT Bhopal University
"""

from __future__ import annotations

import tensorflow as tf


# ---------------------------------------------------------------------------
# Reconstruction Loss
# ---------------------------------------------------------------------------

def reconstruction_loss(
    real_image: tf.Tensor,
    generated_image: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    """
    Compute pixel-wise L2 (MSE) reconstruction loss over the masked region.

    Only the masked (missing) pixels contribute to this loss. The mask is
    0 where pixels are missing and 1 where pixels are present, so we invert
    it to isolate the missing region.

    Args:
        real_image:       Ground-truth image  [B, H, W, C], range [-1, 1].
        generated_image:  Reconstructed image [B, H, W, C], range [-1, 1].
        mask:             Binary mask         [B, H, W, C].
                          0 = missing pixel, 1 = known pixel.

    Returns:
        Scalar MSE loss over masked pixels.
    """
    # Invert mask: 1 where pixels are MISSING
    missing_region_mask = 1.0 - mask

    # Extract the masked region from both real and generated images
    real_masked    = real_image      * missing_region_mask
    gen_masked     = generated_image * missing_region_mask

    # Mean squared error only over masked pixels
    mse = tf.reduce_mean(tf.square(real_masked - gen_masked))
    return mse


# ---------------------------------------------------------------------------
# Adversarial Losses
# ---------------------------------------------------------------------------

def adversarial_loss_discriminator(
    real_logits: tf.Tensor,
    fake_logits: tf.Tensor,
    label_smoothing: float = 0.1,
) -> tf.Tensor:
    """
    Discriminator adversarial loss with optional one-sided label smoothing.

    The discriminator is trained to:
      - Output high values (close to 1) for real images.
      - Output low values (close to 0) for generated (fake) images.

    Args:
        real_logits:     Raw PatchGAN outputs for real images  [B, H', W', 1].
        fake_logits:     Raw PatchGAN outputs for fake images  [B, H', W', 1].
        label_smoothing: Smoothing applied to real labels (e.g. 0.9 instead
                         of 1.0) to prevent overconfident discriminators.

    Returns:
        Scalar discriminator loss.
    """
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_labels = tf.ones_like(real_logits) * (1.0 - label_smoothing)
    fake_labels = tf.zeros_like(fake_logits)

    real_loss = bce(real_labels, real_logits)
    fake_loss = bce(fake_labels, fake_logits)

    return real_loss + fake_loss


def adversarial_loss_generator(fake_logits: tf.Tensor) -> tf.Tensor:
    """
    Generator adversarial loss.

    The generator is trained to produce images that the discriminator
    classifies as real (i.e., push fake logits toward 1).

    Args:
        fake_logits:  Raw PatchGAN outputs for generated images [B, H', W', 1].

    Returns:
        Scalar generator adversarial loss.
    """
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_labels = tf.ones_like(fake_logits)
    return bce(real_labels, fake_logits)


# ---------------------------------------------------------------------------
# Total Generator Loss
# ---------------------------------------------------------------------------

def total_generator_loss(
    real_image: tf.Tensor,
    generated_image: tf.Tensor,
    mask: tf.Tensor,
    fake_logits: tf.Tensor,
    lambda_rec: float = 0.999,
    lambda_adv: float = 0.001,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute the total generator loss as a weighted sum of reconstruction
    loss and adversarial loss.

    Args:
        real_image:       Ground-truth image  [B, H, W, C].
        generated_image:  Reconstructed image [B, H, W, C].
        mask:             Binary mask         [B, H, W, C].
        fake_logits:      Discriminator output on generated image.
        lambda_rec:       Weight for reconstruction loss.
        lambda_adv:       Weight for adversarial loss.

    Returns:
        Tuple of (total_loss, rec_loss, adv_loss) — all scalar tensors.
    """
    rec_loss = reconstruction_loss(real_image, generated_image, mask)
    adv_loss = adversarial_loss_generator(fake_logits)
    total    = lambda_rec * rec_loss + lambda_adv * adv_loss
    return total, rec_loss, adv_loss
