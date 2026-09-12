"""
trainer.py — Training loop for the Context Encoder image completion model.

Uses tf.GradientTape for explicit gradient computation and applies
separate Adam optimizers for the generator (context encoder) and the
PatchGAN discriminator.

Training strategy:
  1. Forward pass: Feed corrupted image → Context Encoder → reconstructed image.
  2. Discriminator step: Train discriminator on (original, reconstructed) pairs.
  3. Generator step: Train context encoder to minimise reconstruction + adversarial loss.
  4. Metrics and sample images are logged every N epochs.

Checkpointing is handled via tf.train.Checkpoint for reliable save/restore.

Author: Ravikant
Institution: VIT Bhopal University
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.losses import (
    adversarial_loss_discriminator,
    total_generator_loss,
)
from src.model import build_context_encoder, build_discriminator
from src.utils import (
    compute_psnr,
    compute_ssim,
    log_metrics,
    save_image_grid,
)


class Trainer:
    """
    Manages the full training process for the image completion model.

    Attributes:
        context_encoder:   The generator (encoder-decoder) network.
        discriminator:     The PatchGAN discriminator network.
        optimizer_gen:     Adam optimizer for the context encoder.
        optimizer_disc:    Adam optimizer for the discriminator.
        checkpoint:        tf.train.Checkpoint for save/restore.
        checkpoint_manager: tf.train.CheckpointManager.
        summary_writer:    TensorBoard FileWriter.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialise the Trainer from a configuration dictionary.

        Args:
            config: Parsed YAML config dict (see config.yaml).
        """
        self.config = config
        model_cfg    = config["model"]
        train_cfg    = config["training"]
        ckpt_cfg     = config["checkpointing"]
        log_cfg      = config["logging"]
        dataset_cfg  = config["dataset"]

        image_size = dataset_cfg["image_size"]
        channels   = dataset_cfg["channels"]

        # Build models
        self.context_encoder = build_context_encoder(
            image_size=image_size,
            channels=channels,
            base_filters=model_cfg["encoder_base_filters"],
        )
        self.discriminator = build_discriminator(
            image_size=image_size,
            channels=channels,
            base_filters=model_cfg["patch_gan_filters"],
        )

        # Optimizers
        self.optimizer_gen = tf.keras.optimizers.Adam(
            learning_rate=train_cfg["learning_rate_g"],
            beta_1=train_cfg["beta1"],
            beta_2=train_cfg["beta2"],
        )
        self.optimizer_disc = tf.keras.optimizers.Adam(
            learning_rate=train_cfg["learning_rate_d"],
            beta_1=train_cfg["beta1"],
            beta_2=train_cfg["beta2"],
        )

        # Checkpointing
        self.checkpoint_dir = ckpt_cfg["checkpoint_dir"]
        self.checkpoint = tf.train.Checkpoint(
            context_encoder=self.context_encoder,
            discriminator=self.discriminator,
            optimizer_gen=self.optimizer_gen,
            optimizer_disc=self.optimizer_disc,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.checkpoint_dir,
            max_to_keep=ckpt_cfg["max_to_keep"],
        )

        # TensorBoard logging
        self.log_dir = log_cfg["log_dir"]
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        self.sample_dir          = log_cfg["sample_dir"]
        self.sample_every        = log_cfg["sample_every_n_epochs"]
        self.save_every          = ckpt_cfg["save_every_n_epochs"]
        self.num_sample_images   = log_cfg["num_sample_images"]
        self.lambda_rec          = train_cfg["lambda_rec"]
        self.lambda_adv          = train_cfg["lambda_adv"]
        self.label_smoothing     = train_cfg["label_smoothing"]

        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Single Training Step
    # -----------------------------------------------------------------------

    @tf.function
    def _train_step(
        self,
        corrupted: tf.Tensor,
        original: tf.Tensor,
        mask: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Execute one forward + backward pass for both discriminator and generator.

        Args:
            corrupted:  Masked input images  [B, H, W, C].
            original:   Ground-truth images  [B, H, W, C].
            mask:       Binary mask          [B, H, W, C].

        Returns:
            Tuple of (gen_total_loss, disc_loss, rec_loss, adv_loss).
        """
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            # Generator forward pass
            generated = self.context_encoder(corrupted, training=True)

            # Discriminator forward pass
            real_logits = self.discriminator(original,  training=True)
            fake_logits = self.discriminator(generated, training=True)

            # Discriminator loss
            disc_loss = adversarial_loss_discriminator(
                real_logits, fake_logits, self.label_smoothing
            )

            # Generator (context encoder) total loss
            gen_total, rec_loss, adv_loss = total_generator_loss(
                original, generated, mask, fake_logits,
                self.lambda_rec, self.lambda_adv,
            )

        # Compute and apply gradients
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        gen_grads  = gen_tape.gradient(gen_total,  self.context_encoder.trainable_variables)

        self.optimizer_disc.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )
        self.optimizer_gen.apply_gradients(
            zip(gen_grads, self.context_encoder.trainable_variables)
        )

        return gen_total, disc_loss, rec_loss, adv_loss

    # -----------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------

    def train(
        self,
        train_dataset: tf.data.Dataset,
        num_epochs: int,
        restore_if_exists: bool = True,
    ) -> None:
        """
        Run the full training loop.

        Args:
            train_dataset:      tf.data.Dataset yielding (corrupted, original, mask).
            num_epochs:         Total number of training epochs.
            restore_if_exists:  If True, restore the latest checkpoint before training.
        """
        start_epoch = 0

        if restore_if_exists and self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            # Infer epoch from checkpoint name
            ckpt_path   = self.checkpoint_manager.latest_checkpoint
            start_epoch = int(ckpt_path.split("-")[-1])
            print(f"Restored checkpoint from epoch {start_epoch}.")

        print(f"\n{'='*60}")
        print(f"  Image Completion — Context Encoder Training")
        print(f"  Author: Ravikant | VIT Bhopal University")
        print(f"  Epochs: {start_epoch} → {num_epochs}")
        print(f"{'='*60}\n")

        # Grab a fixed batch for sample image visualisation
        sample_batch   = next(iter(train_dataset))
        sample_corrupt = sample_batch[0].numpy()
        sample_orig    = sample_batch[2].numpy()  # original
        sample_mask    = sample_batch[2].numpy()

        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()

            # Accumulate losses over the epoch
            gen_losses, disc_losses, rec_losses, adv_losses = [], [], [], []

            for corrupted, original, mask in tqdm(
                train_dataset,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                leave=False,
            ):
                g_loss, d_loss, r_loss, a_loss = self._train_step(
                    corrupted, original, mask
                )
                gen_losses.append(float(g_loss))
                disc_losses.append(float(d_loss))
                rec_losses.append(float(r_loss))
                adv_losses.append(float(a_loss))

            # ---- Epoch-level metrics ----
            mean_g = float(np.mean(gen_losses))
            mean_d = float(np.mean(disc_losses))
            mean_r = float(np.mean(rec_losses))
            mean_a = float(np.mean(adv_losses))

            # Compute PSNR and SSIM on sample batch
            sample_generated = self.context_encoder(
                sample_batch[0], training=False
            ).numpy()
            psnr = compute_psnr(sample_orig, sample_generated)
            ssim = compute_ssim(sample_orig, sample_generated)

            log_metrics(epoch + 1, mean_g, mean_d, mean_r, mean_a, psnr, ssim)

            # ---- TensorBoard logging ----
            with self.summary_writer.as_default():
                tf.summary.scalar("loss/generator_total", mean_g, step=epoch)
                tf.summary.scalar("loss/discriminator",   mean_d, step=epoch)
                tf.summary.scalar("loss/reconstruction",  mean_r, step=epoch)
                tf.summary.scalar("loss/adversarial",     mean_a, step=epoch)
                tf.summary.scalar("metrics/psnr",         psnr,   step=epoch)
                tf.summary.scalar("metrics/ssim",         ssim,   step=epoch)

            # ---- Save sample images ----
            if (epoch + 1) % self.sample_every == 0:
                sample_path = os.path.join(
                    self.sample_dir, f"epoch_{epoch + 1:04d}.png"
                )
                save_image_grid(
                    sample_corrupt,
                    sample_generated,
                    sample_orig,
                    save_path=sample_path,
                    n_images=self.num_sample_images,
                    title=f"Epoch {epoch + 1} — PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}",
                )
                print(f"  → Sample grid saved: {sample_path}")

            # ---- Save checkpoint ----
            if (epoch + 1) % self.save_every == 0:
                ckpt_path = self.checkpoint_manager.save(checkpoint_number=epoch + 1)
                print(f"  → Checkpoint saved: {ckpt_path}")

            elapsed = time.time() - epoch_start
            print(f"  Epoch time: {elapsed:.1f}s\n")

        print("Training complete.")

    # -----------------------------------------------------------------------
    # Model Summary
    # -----------------------------------------------------------------------

    def print_model_summaries(self) -> None:
        """Print Keras model summaries for both networks."""
        print("\n--- Context Encoder (Generator) ---")
        self.context_encoder.summary()
        print("\n--- PatchGAN Discriminator ---")
        self.discriminator.summary()
