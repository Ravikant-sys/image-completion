"""
model.py — Context Encoder architecture for image completion.

Architecture:
  - Encoder: Convolutional network that compresses masked input image
              into a compact latent bottleneck representation.
  - Decoder: Transposed convolutional network that reconstructs the
              missing region from the bottleneck.
  - Context Encoder: Encoder + Decoder combined (Generator).
  - PatchGAN Discriminator: Classifies 8x8 patches of the image
              as real or generated (more effective than a single score).

Reference Architecture Inspiration:
  Pathak et al. "Context Encoders: Feature Learning by Inpainting" CVPR 2016.
  https://arxiv.org/abs/1604.07379

Author: Ravikant
Institution: VIT Bhopal University
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Helper Blocks
# ---------------------------------------------------------------------------

def _encoder_block(
    x: tf.Tensor,
    filters: int,
    use_batchnorm: bool = True,
    name: str = "",
) -> tf.Tensor:
    """A single encoder block: Conv → [BN] → LeakyReLU."""
    x = layers.Conv2D(
        filters,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=not use_batchnorm,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
        name=f"{name}_conv",
    )(x)
    if use_batchnorm:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.LeakyReLU(alpha=0.2, name=f"{name}_lrelu")(x)
    return x


def _decoder_block(
    x: tf.Tensor,
    filters: int,
    use_dropout: bool = False,
    name: str = "",
) -> tf.Tensor:
    """A single decoder block: ConvTranspose → BN → [Dropout] → ReLU."""
    x = layers.Conv2DTranspose(
        filters,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
        name=f"{name}_deconv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    if use_dropout:
        x = layers.Dropout(0.5, name=f"{name}_dropout")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def build_encoder(
    image_size: int = 64,
    channels: int = 3,
    base_filters: int = 64,
) -> keras.Model:
    """
    Build the encoder network.

    Maps a masked input image (image_size × image_size × channels) to a
    compact bottleneck tensor of shape (4 × 4 × 512) for image_size=64.

    Args:
        image_size:   Height/width of the input image (must be power of 2).
        channels:     Number of input channels (1 for grayscale, 3 for RGB).
        base_filters: Number of filters in the first conv layer.

    Returns:
        A Keras Model (encoder).
    """
    inp = keras.Input(shape=(image_size, image_size, channels), name="encoder_input")

    # Block 1 — no BN on first layer (common practice)
    x = _encoder_block(inp, base_filters * 1, use_batchnorm=False, name="enc1")
    # 32×32
    x = _encoder_block(x, base_filters * 2, name="enc2")
    # 16×16
    x = _encoder_block(x, base_filters * 4, name="enc3")
    # 8×8
    x = _encoder_block(x, base_filters * 8, name="enc4")
    # 4×4  ← bottleneck

    return keras.Model(inputs=inp, outputs=x, name="Encoder")


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def build_decoder(
    bottleneck_shape: tuple[int, int, int] = (4, 4, 512),
    output_channels: int = 3,
) -> keras.Model:
    """
    Build the decoder network.

    Maps the bottleneck representation back to a full-resolution image.

    Args:
        bottleneck_shape:  Shape of encoder output (H, W, C).
        output_channels:   Number of output channels.

    Returns:
        A Keras Model (decoder).
    """
    inp = keras.Input(shape=bottleneck_shape, name="decoder_input")

    x = _decoder_block(inp, 256, use_dropout=True,  name="dec1")
    # 8×8
    x = _decoder_block(x,   128, use_dropout=False, name="dec2")
    # 16×16
    x = _decoder_block(x,   64,  use_dropout=False, name="dec3")
    # 32×32

    # Final output layer — upsample to full resolution, tanh activation
    x = layers.Conv2DTranspose(
        output_channels,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="tanh",
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
        name="dec_output",
    )(x)
    # 64×64

    return keras.Model(inputs=inp, outputs=x, name="Decoder")


# ---------------------------------------------------------------------------
# Context Encoder  (Generator = Encoder + Decoder)
# ---------------------------------------------------------------------------

def build_context_encoder(
    image_size: int = 64,
    channels: int = 3,
    base_filters: int = 64,
) -> keras.Model:
    """
    Build the full Context Encoder (generator).

    Combines encoder and decoder into one end-to-end model:
        masked_image → Encoder → bottleneck → Decoder → reconstructed_image

    Args:
        image_size:   Height/width of the input image.
        channels:     Number of input/output channels.
        base_filters: Base filter count for the encoder.

    Returns:
        A Keras Model (context encoder / generator).
    """
    encoder = build_encoder(image_size, channels, base_filters)
    bottleneck_shape = encoder.output_shape[1:]  # (H, W, C)
    decoder = build_decoder(bottleneck_shape, channels)

    inp = keras.Input(shape=(image_size, image_size, channels), name="ce_input")
    bottleneck = encoder(inp)
    output = decoder(bottleneck)

    return keras.Model(inputs=inp, outputs=output, name="ContextEncoder")


# ---------------------------------------------------------------------------
# PatchGAN Discriminator
# ---------------------------------------------------------------------------

def build_discriminator(
    image_size: int = 64,
    channels: int = 3,
    base_filters: int = 64,
) -> keras.Model:
    """
    Build a PatchGAN discriminator.

    Instead of outputting a single real/fake score, the PatchGAN produces
    an 8×8 grid of real/fake predictions — each corresponding to a
    receptive-field patch of the input image. This provides richer gradient
    signal than a single sigmoid output.

    Args:
        image_size:   Height/width of the input image.
        channels:     Number of input channels.
        base_filters: Base filter count.

    Returns:
        A Keras Model (PatchGAN discriminator).
    """
    inp = keras.Input(shape=(image_size, image_size, channels), name="disc_input")

    # Layer 1 — no BN
    x = layers.Conv2D(
        base_filters,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
        name="disc_conv1",
    )(inp)
    x = layers.LeakyReLU(0.2, name="disc_lrelu1")(x)
    # 32×32

    x = layers.Conv2D(
        base_filters * 2, 4, strides=2, padding="same", use_bias=False,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
        name="disc_conv2",
    )(x)
    x = layers.BatchNormalization(name="disc_bn2")(x)
    x = layers.LeakyReLU(0.2, name="disc_lrelu2")(x)
    # 16×16

    x = layers.Conv2D(
        base_filters * 4, 4, strides=2, padding="same", use_bias=False,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
        name="disc_conv3",
    )(x)
    x = layers.BatchNormalization(name="disc_bn3")(x)
    x = layers.LeakyReLU(0.2, name="disc_lrelu3")(x)
    # 8×8

    x = layers.Conv2D(
        base_filters * 8, 4, strides=1, padding="same", use_bias=False,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
        name="disc_conv4",
    )(x)
    x = layers.BatchNormalization(name="disc_bn4")(x)
    x = layers.LeakyReLU(0.2, name="disc_lrelu4")(x)
    # 8×8

    # Final patch-level output (no activation — raw logits for BCE with logits)
    x = layers.Conv2D(
        1, 4, strides=1, padding="same",
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
        name="disc_output",
    )(x)
    # 8×8×1

    return keras.Model(inputs=inp, outputs=x, name="PatchGAN_Discriminator")
