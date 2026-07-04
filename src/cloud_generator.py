"""Synthetic fractal cloud generation.

Produces unlimited (cloudy, clean) training pairs from clean images by overlaying
a fake cloud built from smoothed multi-octave random noise (a fast, dependency-free
stand-in for Perlin noise), with soft feathered edges and a simulated cloud shadow.

Ported from the ``Step 2 — Synthetic Cloud Generator`` cell of
``SkyClear_Day1_Pipeline_(1).ipynb``.
"""

import random
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def generate_fractal_cloud_mask(
    size: Tuple[int, int],
    coverage: float = 0.55,
    softness: float = 25,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a soft-edged fractal cloud mask via multi-octave noise.

    Args:
        size: ``(height, width)`` of the mask to generate.
        coverage: target fraction of the mask that ends up opaque cloud.
        softness: controls the blur/feathering strength of the mask edges.
        seed: optional seed for the base noise octaves. Note that this only
            makes the noise field reproducible; :func:`apply_synthetic_cloud`
            also draws from the unseeded global RNG for coverage, shadow
            offset jitter and cloud tint, so passing a seed does not make the
            full cloudy-image pipeline deterministic end-to-end.

    Returns:
        A ``(h, w)`` float32 array in ``[0, 1]``, where 1 is fully opaque cloud.
    """
    h, w = size
    rng = np.random.RandomState(seed) if seed is not None else np.random

    octaves, persistence, base_res = 4, 0.55, 5
    noise = np.zeros((h, w), dtype=np.float32)
    amplitude, total_amp = 1.0, 0.0

    for o in range(octaves):
        res = max(2, base_res * (2 ** o))
        small = rng.rand(res, res)
        layer = np.array(
            Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
            dtype=np.float32,
        ) / 255.0
        noise += layer * amplitude
        total_amp += amplitude
        amplitude *= persistence
    noise /= total_amp
    noise = gaussian_filter(noise, sigma=softness / 4.0)

    threshold = np.quantile(noise, 1 - coverage)
    mask = np.clip((noise - threshold) / (noise.max() - threshold + 1e-6), 0, 1)

    # push mask toward binary (opaque core) instead of smooth gradient
    mask = mask ** 0.5  # steepens the curve -- more pixels land near 1.0
    mask = gaussian_filter(mask, sigma=softness / 6.0)  # much lighter blur -- only edges feather

    return np.clip(mask, 0, 1).astype(np.float32)


def apply_synthetic_cloud(
    clean_img: Image.Image,
    coverage_range: Tuple[float, float] = (0.35, 0.7),
    seed: Optional[int] = None,
    add_shadow: bool = True,
) -> Tuple[Image.Image, np.ndarray]:
    """Overlay a synthetic cloud (and optional shadow) onto a clean image.

    Args:
        clean_img: cloud-free RGB PIL image.
        coverage_range: ``(min, max)`` fraction of the image the cloud may
            cover; the actual coverage is sampled uniformly from this range.
        seed: optional seed forwarded to the underlying cloud mask noise.
        add_shadow: whether to also darken a shifted region to simulate a
            cloud shadow cast on the ground.

    Returns:
        A tuple of ``(cloudy PIL image, cloud mask array in [0, 1])``.
    """
    arr = np.array(clean_img).astype(np.float32) / 255.0
    h, w = arr.shape[:2]

    coverage = random.uniform(*coverage_range)
    mask = generate_fractal_cloud_mask((h, w), coverage=coverage, softness=random.uniform(15, 35), seed=seed)
    result = arr.copy()

    if add_shadow:
        dx, dy = int(w * 0.03), int(h * 0.03)
        shadow_mask = np.roll(mask, shift=(dy, dx), axis=(0, 1))
        result = result * (1 - (shadow_mask * 0.35)[:, :, None])

    cloud_color = np.array([0.94, 0.94, 0.93]) + np.random.uniform(-0.02, 0.02, 3)
    cloud_layer = np.ones_like(arr) * cloud_color
    result = result * (1 - mask[:, :, None]) + cloud_layer * mask[:, :, None]
    result = np.clip(result, 0, 1)

    return Image.fromarray((result * 255).astype(np.uint8)), mask
