"""Gradio web demo for SkyClear GAN-based cloud removal.

Loads a trained generator checkpoint once at startup and serves a simple
upload -> Remove Clouds -> compare UI. Reuses the same model-loading and
pre/post-processing code as `src/inference.py` — no logic is duplicated here.

Usage:
    python app.py            # local demo at http://127.0.0.1:7860
    python app.py --share    # also expose a temporary public Gradio URL
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import torch
from PIL import Image

from src.dataset import IMG_SIZE
from src.inference import DEFAULT_CHECKPOINT, load_generator, postprocess, preprocess

# Anchor paths to this file's location so the app works regardless of the
# current working directory it's launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / DEFAULT_CHECKPOINT
EXAMPLES_DIR = PROJECT_ROOT / "examples"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator: Optional[torch.nn.Module] = None


def remove_clouds(image: Optional[Image.Image]) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """Resize the uploaded image to the model's working resolution and run cloud removal.

    Args:
        image: the uploaded PIL image, or ``None`` if nothing has been provided.

    Returns:
        A tuple of ``(resized input image, cloud-removed output image)``; both
        are ``None`` if no image has been uploaded yet.

    Raises:
        gr.Error: with a clean, user-facing message if the model is unavailable
            or inference fails (e.g. a malformed upload), so a raw traceback
            never surfaces in the UI.
    """
    if image is None:
        return None, None

    if generator is None:
        raise gr.Error("Model is not loaded — check the server logs and restart the demo.")

    try:
        input_tensor = preprocess(image, IMG_SIZE).to(device)
        with torch.no_grad():
            output_tensor = generator(input_tensor)
        return postprocess(input_tensor), postprocess(output_tensor)
    except Exception as exc:  # noqa: BLE001 — surface any failure as a clean UI message
        raise gr.Error(f"Could not process this image: {exc}")


def _example_paths() -> List[str]:
    """Return sorted paths of bundled example images, or an empty list if none exist."""
    if not EXAMPLES_DIR.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png"}
    return [str(p) for p in sorted(EXAMPLES_DIR.iterdir()) if p.suffix.lower() in exts]


def build_interface() -> gr.Blocks:
    """Construct the Gradio UI: an upload box, a Remove Clouds button, and two comparison images."""
    with gr.Blocks(title="SkyClear — Cloud Removal Demo") as demo:
        gr.Markdown(
            "# SkyClear — GAN-Based Cloud Removal\n"
            "BAH 2026 demo (Team SanjGPT): a U-Net + PatchGAN model reconstructs cloud-obscured "
            "regions of LISS-IV satellite imagery. Trained on synthetic clouds over clean imagery — "
            "results on genuinely cloudy scenes may show some residual sim-to-real domain gap."
        )
        input_image = gr.Image(type="pil", label="Upload an image")

        examples = _example_paths()
        if examples:
            gr.Examples(
                examples=[[path] for path in examples],
                inputs=[input_image],
                label="Or click a sample cloudy image to try",
            )

        run_button = gr.Button("Remove Clouds", variant="primary")
        with gr.Row():
            resized_input = gr.Image(type="pil", label=f"Input (resized to {IMG_SIZE}x{IMG_SIZE})")
            output_image = gr.Image(type="pil", label="Cloud-Removed Output")

        run_button.click(fn=remove_clouds, inputs=input_image, outputs=[resized_input, output_image])

    return demo


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the demo."""
    parser = argparse.ArgumentParser(description="Launch the SkyClear cloud-removal Gradio demo.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Expose a temporary public Gradio URL in addition to the local one.",
    )
    return parser.parse_args()


def load_model() -> None:
    """Load the checkpoint once at startup into the module-level ``generator``.

    Exits with a clean message (no raw traceback) if the checkpoint is missing
    or cannot be loaded, so a broken demo never reaches the browser.
    """
    global generator

    if not CHECKPOINT_PATH.is_file():
        print(
            f"ERROR: No checkpoint found at '{CHECKPOINT_PATH}'.\n"
            "Train a model first, e.g.:\n"
            "    python -m src.train --data-dir data/clean --checkpoint-dir checkpoints\n"
            f"or place a trained checkpoint there, then re-run `python app.py`."
        )
        raise SystemExit(1)

    print(f"Loading checkpoint: {CHECKPOINT_PATH} (device: {device})")
    try:
        generator = load_generator(str(CHECKPOINT_PATH), device)
    except Exception as exc:  # noqa: BLE001 — startup: report cleanly and exit, don't dump a traceback
        print(
            f"ERROR: Failed to load checkpoint '{CHECKPOINT_PATH}': {exc}\n"
            "The file may be corrupted or incompatible with the current model. "
            "Re-train or replace it, then re-run `python app.py`."
        )
        raise SystemExit(1)
    print("Model loaded. Launching Gradio demo...")


def main() -> None:
    """Load the checkpoint once at startup, then launch the Gradio demo."""
    args = parse_args()
    load_model()
    build_interface().launch(share=args.share)


if __name__ == "__main__":
    main()
