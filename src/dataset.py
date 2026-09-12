"""
dataset.py — Data loading and preprocessing pipeline using tf.data.

Supports MNIST and CIFAR-10 via tensorflow_datasets. All images are resized
to a common square resolution (default 64×64) and normalized to [-1, 1].

Two masking strategies are implemented:
  1. Center mask  — zeros out the central square patch of the image.
  2. Random mask  — randomly zeros out a fraction of pixels.

The pipeline returns (corrupted_image, original_image, mask) tuples, where:
  - corrupted_image:  Masked input fed to the Context Encoder.
  - original_image:   Ground-truth target for reconstruction loss.
  - mask:             Binary mask (1 = known pixel, 0 = missing pixel).

Author: Ravikant
Institution: VIT Bhopal University
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_datasets as tfds


# ---------------------------------------------------------------------------
# Masking Functions
# ---------------------------------------------------------------------------

def apply_center_mask(
    image: tf.Tensor,
    mask_size: int = 32,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Apply a center square mask to an image.

    The central `mask_size × mask_size` region is zeroed out.

    Args:
        image:     Input image tensor [H, W, C], values in [-1, 1].
        mask_size: Side length of the square mask in pixels.

    Returns:
        Tuple of:
            corrupted_image:  Image with center region masked to 0.
            original_image:   Unchanged original image.
            mask:             Float32 tensor [H, W, C]: 1=known, 0=missing.
    """
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    c = tf.shape(image)[2]

    top    = (h - mask_size) // 2
    left   = (w - mask_size) // 2

    # Build a mask: 1 everywhere, then place 0s in the center block
    # Using scatter via tf.pad + slice tricks
    mask_patch = tf.zeros((mask_size, mask_size, c), dtype=tf.float32)

    # Pad the mask patch back to full image size
    pad_top    = top
    pad_bottom = h - top - mask_size
    pad_left   = left
    pad_right  = w - left - mask_size

    padded_zeros = tf.pad(
        mask_patch,
        [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        constant_values=1.0,   # Outside the patch = 1 (known)
    )
    mask = padded_zeros  # 1 = known, 0 = missing

    corrupted_image = image * mask
    return corrupted_image, image, mask


def apply_random_mask(
    image: tf.Tensor,
    mask_ratio: float = 0.50,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Apply a random pixel mask to an image.

    Each pixel is independently masked with probability `mask_ratio`.

    Args:
        image:      Input image tensor [H, W, C], values in [-1, 1].
        mask_ratio: Fraction of pixels to mask out (set to 0).

    Returns:
        Tuple of (corrupted_image, original_image, mask).
    """
    shape   = tf.shape(image)
    # Random mask: 1 = keep, 0 = mask
    uniform = tf.random.uniform(shape=(shape[0], shape[1], 1), minval=0.0, maxval=1.0)
    mask    = tf.cast(uniform > mask_ratio, tf.float32)
    mask    = tf.tile(mask, [1, 1, shape[2]])  # broadcast across channels

    corrupted_image = image * mask
    return corrupted_image, image, mask


# ---------------------------------------------------------------------------
# Preprocessing Helpers
# ---------------------------------------------------------------------------

def _preprocess_image(
    image: tf.Tensor,
    image_size: int,
    channels: int,
) -> tf.Tensor:
    """
    Resize, cast to float, normalize to [-1, 1], and ensure RGB channels.

    Args:
        image:      Raw image tensor (uint8 or float).
        image_size: Target height/width.
        channels:   Target number of channels (1 or 3).

    Returns:
        Preprocessed image tensor [image_size, image_size, channels] in [-1, 1].
    """
    # Convert grayscale to RGB by repeating the channel dimension
    if image.shape[-1] == 1 and channels == 3:
        image = tf.image.grayscale_to_rgb(image)

    # Resize to target size using bilinear interpolation
    image = tf.image.resize(image, [image_size, image_size])

    # Normalize from [0, 255] to [-1, 1]
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0

    return image


# ---------------------------------------------------------------------------
# DataPipeline
# ---------------------------------------------------------------------------

class DataPipeline:
    """
    A tf.data-based data loading pipeline for image completion.

    Loads datasets from tensorflow_datasets, applies preprocessing,
    masking, and batching for efficient training.

    Supported datasets:
        - "MNIST"   : 60,000 grayscale handwritten digit images (28×28 → 64×64)
        - "CIFAR10" : 50,000 colour images across 10 classes (32×32 → 64×64)

    Example usage:
        pipeline = DataPipeline("MNIST", image_size=64, mask_type="center")
        train_ds = pipeline.get_train_dataset(batch_size=64)
        for corrupted, original, mask in train_ds:
            ...
    """

    SUPPORTED_DATASETS = {"MNIST", "CIFAR10"}

    _TFDS_NAMES: dict[str, str] = {
        "MNIST":   "mnist",
        "CIFAR10": "cifar10",
    }

    def __init__(
        self,
        dataset_name: str = "MNIST",
        image_size: int = 64,
        channels: int = 3,
        mask_type: str = "center",
        center_mask_size: int = 32,
        random_mask_ratio: float = 0.50,
    ) -> None:
        """
        Initialise the DataPipeline.

        Args:
            dataset_name:      Name of the dataset ("MNIST" or "CIFAR10").
            image_size:        Target image height/width.
            channels:          Number of output channels.
            mask_type:         Masking strategy ("center" or "random").
            center_mask_size:  Size of center mask (pixels), used if mask_type="center".
            random_mask_ratio: Fraction to mask, used if mask_type="random".

        Raises:
            ValueError: If dataset_name or mask_type is not supported.
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. "
                f"Choose from: {self.SUPPORTED_DATASETS}"
            )
        if mask_type not in ("center", "random"):
            raise ValueError(f"mask_type must be 'center' or 'random', got '{mask_type}'")

        self.dataset_name      = dataset_name
        self.image_size        = image_size
        self.channels          = channels
        self.mask_type         = mask_type
        self.center_mask_size  = center_mask_size
        self.random_mask_ratio = random_mask_ratio

    # -----------------------------------------------------------------------

    def _make_mask_fn(self):
        """Return a masking function based on configured mask_type."""
        if self.mask_type == "center":
            mask_size = self.center_mask_size
            def _mask_fn(image):
                return apply_center_mask(image, mask_size)
        else:
            ratio = self.random_mask_ratio
            def _mask_fn(image):
                return apply_random_mask(image, ratio)
        return _mask_fn

    def _build_pipeline(
        self,
        split: str,
        batch_size: int,
        shuffle: bool,
        buffer_size: int = 10_000,
        prefetch: int = tf.data.AUTOTUNE,
    ) -> tf.data.Dataset:
        """
        Build a tf.data pipeline for the given split.

        Args:
            split:       Dataset split (e.g. "train", "test").
            batch_size:  Number of samples per batch.
            shuffle:     Whether to shuffle the dataset.
            buffer_size: Shuffle buffer size.
            prefetch:    Number of batches to prefetch.

        Returns:
            A tf.data.Dataset yielding (corrupted, original, mask) batches.
        """
        tfds_name = self._TFDS_NAMES[self.dataset_name]
        ds, info  = tfds.load(
            tfds_name,
            split=split,
            with_info=True,
            as_supervised=True,
        )

        image_size = self.image_size
        channels   = self.channels
        mask_fn    = self._make_mask_fn()

        def _process(image, _label):
            image = _preprocess_image(image, image_size, channels)
            corrupted, original, mask = mask_fn(image)
            return corrupted, original, mask

        if shuffle:
            ds = ds.shuffle(buffer_size)

        ds = (
            ds
            .map(_process, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size, drop_remainder=True)
            .prefetch(prefetch)
        )
        return ds

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_train_dataset(self, batch_size: int = 64) -> tf.data.Dataset:
        """
        Return a shuffled, batched training dataset.

        Args:
            batch_size: Number of samples per batch.

        Returns:
            tf.data.Dataset yielding (corrupted_image, original_image, mask).
        """
        return self._build_pipeline("train", batch_size, shuffle=True)

    def get_test_dataset(self, batch_size: int = 64) -> tf.data.Dataset:
        """
        Return a batched test/validation dataset (no shuffle).

        Args:
            batch_size: Number of samples per batch.

        Returns:
            tf.data.Dataset yielding (corrupted_image, original_image, mask).
        """
        split = "test"
        return self._build_pipeline(split, batch_size, shuffle=False)

    def dataset_size(self, split: str = "train") -> int:
        """
        Return the number of examples in the given split.

        Args:
            split: Dataset split name.

        Returns:
            Integer count of examples.
        """
        tfds_name = self._TFDS_NAMES[self.dataset_name]
        _, info = tfds.load(tfds_name, split=split, with_info=True, as_supervised=True)
        return info.splits[split].num_examples
