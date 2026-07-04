"""Run the trained SkyClear generator on a single image and save a before/after result.

Ported and adapted from the ``Step 7 — Visualize a Result`` cell of
``SkyClear_Day1_Pipeline_(1).ipynb`` for local (non-Colab), single-image use.
Unlike the notebook's cell, there is no ground-truth clean image available at
inference time on a genuinely cloudy scene, so only the cloudy input and the
reconstructed output are compared.

Usage:
    python -m src.inference --input path/to/cloudy.jpg --output outputs/inference_result.png
"""

import argparse
import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from .model import UNetGenerator

DEFAULT_CHECKPOINT = "checkpoints/skyclear_epoch65.pt"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a single-image inference run."""
    parser = argparse.ArgumentParser(description="Run cloud removal inference on a single image.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input (cloudy) image.")
    parser.add_argument(
        "--output", type=str, default="outputs/inference_result.png",
        help="Path to save the before/after comparison image.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
        help="Path to a trained generator checkpoint (.pt).",
    )
    parser.add_argument(
        "--img-size", type=int, default=128,
        help="Resolution the checkpoint was trained at; must be divisible by 16.",
    )
    args = parser.parse_args()
    assert args.img_size % 16 == 0, "--img-size must be divisible by 16 (generator has 4 down/up stages)."
    return args


def load_generator(checkpoint_path: str, device: torch.device) -> UNetGenerator:
    """Build a UNetGenerator and load its trained weights from a checkpoint file."""
    generator = UNetGenerator().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    return generator


def preprocess(image: Image.Image, img_size: int) -> torch.Tensor:
    """Resize an RGB PIL image and normalize it to a ``[-1, 1]`` model input tensor."""
    resized = image.convert("RGB").resize((img_size, img_size))
    arr = np.array(resized).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def postprocess(tensor: torch.Tensor) -> Image.Image:
    """Convert a ``[-1, 1]`` model output tensor of shape ``(1, 3, H, W)`` to an 8-bit RGB image."""
    arr = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    arr = np.clip((arr + 1) / 2, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def remove_clouds(
    input_path: str,
    checkpoint_path: str,
    img_size: int,
    device: torch.device,
) -> Tuple[Image.Image, Image.Image]:
    """Run the generator on a single cloudy image.

    Args:
        input_path: path to the cloudy input image.
        checkpoint_path: path to a trained generator checkpoint.
        img_size: resolution to resize to before running the model.
        device: torch device to run inference on.

    Returns:
        A tuple of ``(cloudy input at working resolution, reconstructed output)``,
        both as RGB PIL images of size ``(img_size, img_size)``.
    """
    generator = load_generator(checkpoint_path, device)
    cloudy_img = Image.open(input_path).convert("RGB").resize((img_size, img_size))
    input_tensor = preprocess(cloudy_img, img_size).to(device)

    with torch.no_grad():
        output_tensor = generator(input_tensor)

    return cloudy_img, postprocess(output_tensor)


def save_before_after(cloudy_img: Image.Image, output_img: Image.Image, output_path: str) -> None:
    """Save a side-by-side (cloudy input | reconstructed output) comparison image."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    w, h = cloudy_img.size
    combined = Image.new("RGB", (w * 2, h))
    combined.paste(cloudy_img, (0, 0))
    combined.paste(output_img, (w, 0))
    combined.save(output_path)
    print(f"Saved before/after comparison -> {output_path}")


def main() -> None:
    """Load a checkpoint, run inference on a single image, and save the result."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cloudy_img, output_img = remove_clouds(args.input, args.checkpoint, args.img_size, device)
    save_before_after(cloudy_img, output_img, args.output)


if __name__ == "__main__":
    main()
