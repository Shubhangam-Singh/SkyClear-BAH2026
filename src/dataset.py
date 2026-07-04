"""PyTorch Dataset pairing clean images with on-the-fly synthetic cloud cover.

Ported from the ``Step 3 — Dataset`` cell of ``SkyClear_Day1_Pipeline_(1).ipynb``.
"""

import glob
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .cloud_generator import apply_synthetic_cloud

IMG_SIZE = 128


class SyntheticCloudDataset(Dataset):
    """Loads clean images from disk and synthesizes a cloudy version per sample.

    Each ``__getitem__`` call generates a fresh random cloud overlay, so the
    effective number of (cloudy, clean) training pairs is unlimited even
    though the underlying clean image count is fixed.
    """

    def __init__(self, image_dir: str, img_size: int = IMG_SIZE) -> None:
        """Args:
            image_dir: directory of cloud-free ``.jpg``/``.jpeg``/``.png``/``.tif``/``.tiff`` images.
            img_size: side length (pixels) images are resized to; must be
                divisible by 16 to match the generator's skip connections.
        """
        self.files = (
            glob.glob(f"{image_dir}/*.jpg")
            + glob.glob(f"{image_dir}/*.jpeg")
            + glob.glob(f"{image_dir}/*.png")
            + glob.glob(f"{image_dir}/*.tif")
            + glob.glob(f"{image_dir}/*.tiff")
        )
        self.img_size = img_size
        assert len(self.files) > 0, f"No images found in {image_dir}"

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns:
            ``(cloudy, clean, mask)`` where ``cloudy``/``clean`` are ``(3, H, W)``
            tensors normalized to ``[-1, 1]`` and ``mask`` is a ``(1, H, W)``
            tensor in ``[0, 1]``.
        """
        img = Image.open(self.files[idx]).convert("RGB").resize((self.img_size, self.img_size))
        cloudy_img, mask = apply_synthetic_cloud(img)

        clean_t = torch.from_numpy(np.array(img).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        cloudy_t = torch.from_numpy(np.array(cloudy_img).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        return cloudy_t, clean_t, mask_t
