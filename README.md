# SkyClear — GAN-Based Cloud Removal for LISS-IV Satellite Imagery

**Bharatiya Antariksh Hackathon (BAH) 2026 — Team SanjGPT**
Problem Statement 02: Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery

🌐 **[Live Website](https://Shubhangam-Singh.github.io/SkyClear-BAH2026/)** — landing page + interactive live demo

---

## Overview

Persistent cloud cover over tropical and mountainous regions of India (e.g. the North Eastern Region) severely limits the usability of optical satellite imagery like LISS-IV. Traditional cloud masking simply discards affected pixels, resulting in incomplete scenes and reduced temporal data availability — exactly when timely imagery is often needed most (disaster response, seasonal monitoring, land-use mapping).

**SkyClear** is a Generative Adversarial Network (GAN) that reconstructs cloud-obscured regions of LISS-IV imagery instead of discarding them, producing complete, analysis-ready output — increasing the effective temporal frequency of usable observations without requiring additional satellite passes.

This repository contains the GAN baseline implementation. A diffusion-based variant is also under exploration as a comparative architecture (see [Roadmap](#roadmap)).

---

## Key Idea: Solving the Paired-Data Bottleneck

Naturally-paired cloudy/clean LISS-IV images of the same location are scarce. Instead of waiting for rare paired acquisitions, this pipeline:

1. Takes **cloud-free LISS-IV scenes** (ground truth)
2. **Synthetically generates realistic cloud cover** using multi-octave fractal noise (Perlin-noise-style), with soft feathered edges, variable opacity, and simulated cloud shadows
3. Trains the GAN on these **(synthetic-cloudy, clean)** pairs — unlimited supervised data, no paired-acquisition scarcity problem
4. Validates the trained model against **genuine cloudy LISS-IV scenes** it has never seen, to check real-world generalization

---

## Architecture

```
Input: Cloudy LISS-IV image (Green / Red / NIR bands)
        │
        ▼
┌───────────────────────┐        ┌──────────────────────────┐
│   U-Net Generator      │──────▶│  Reconstructed (clean)     │
│   (encoder-decoder      │        │  image                     │
│    with skip connections)│       └──────────────────────────┘
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  PatchGAN Discriminator │   ← judges real vs. generated patches,
│                          │     conditioned on the cloudy input
└───────────────────────┘

Loss = Adversarial (LSGAN) + λ · L1 pixel reconstruction loss
```

- **Generator:** U-Net with 4 downsampling / 4 upsampling blocks, skip connections, InstanceNorm, LeakyReLU/ReLU
- **Discriminator:** PatchGAN (classifies overlapping patches as real/fake rather than the whole image — encourages sharper local detail)
- **Losses:** LSGAN adversarial loss (stabler than vanilla GAN loss) + weighted L1 pixel loss (`L1_LAMBDA`, tuned between 100–150 across experiments)

---

## Dataset

| Source | Description | Count |
|---|---|---|
| Custom aerial/satellite screenshots | Seed images for early pipeline validation | ~22 |
| Real LISS-IV scene (tiled) | Genuine ResourceSat LISS-IV product (Green/Red/NIR bands), tiled into 256×256 patches | ~3,465 |
| **Total training images** | | **~3,600+** |

All clean images are cloud-free by selection (verified visually before tiling). Cloud-free scenes are the *ground truth* — synthetic clouds are generated on top at training time.

**Real cloudy LISS-IV scenes** (genuinely cloud-covered, downloaded via [Bhoonidhi](https://bhoonidhi.nrsc.gov.in)) are held out separately and used only for qualitative real-world testing — never for training.

---

## Results

| Test | PSNR | SSIM |
|---|---|---|
| Training-distribution sample (163-image stage) | 25.62 dB | 0.945 |
| Held-out synthetic-cloud test (unseen image, never trained on) | 22.00 dB | 0.891 |
| Full dataset (~3,600 images) training sample | 24.63 – 26.00 dB | 0.903 – 0.928 |

**Real-world qualitative test:** when applied to a genuinely cloudy LISS-IV scene (not synthetically clouded), the model shows partial reconstruction — it visibly attempts to recover ground detail under cloud regions, but leaves residual haze in some areas. This surfaces a well-documented challenge in this problem space: the **sim-to-real domain gap** between synthetically-generated training clouds and the appearance of genuine atmospheric cloud cover. Closing this gap (via more realistic cloud synthesis, or genuine paired before/after acquisitions where available) is identified as the primary direction for further work.

Sample outputs are in [`outputs/`](./outputs).

---

## Repository Structure

```
skyclear_gan/
├── SkyClear_Day1_Pipeline_(1).ipynb   # Full Colab-ready pipeline: data → training → eval
├── src/                             # Local (non-Colab) refactor of the same pipeline logic
│   ├── cloud_generator.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
├── app.py                           # Gradio interactive demo (reuses src/ modules)
├── README.md
├── requirements.txt
├── outputs/                        # Sample before/after result images
│   └── sample_result.png
└── .gitignore
```

**Note:** Trained model checkpoints (`.pt` files) and raw satellite imagery are **not included** in this repository — they are large binary files unsuited to Git, and LISS-IV raw data is subject to Bhoonidhi's End User License Agreement. See [Reproducing Results](#reproducing-results) below to regenerate them.

---

## Tech Stack

- **Framework:** PyTorch
- **Geospatial:** Rasterio, GDAL
- **Data/Eval:** NumPy, SciPy, scikit-image (PSNR, SSIM), Matplotlib
- **Compute:** Google Colab (T4 GPU), Google Drive for checkpoint persistence
- **Data source:** [Bhoonidhi](https://bhoonidhi.nrsc.gov.in) — ISRO/NRSC's Earth Observation Data Hub (LISS-IV via ResourceSat-2/2A)

---

## Reproducing Results

1. Open `SkyClear_Day1_Pipeline.ipynb` in [Google Colab](https://colab.research.google.com)
2. `Runtime → Change runtime type → T4 GPU`
3. Mount your Google Drive and create the folder structure:
   ```
   MyDrive/BAH2026/
   ├── data/clean/       ← place cloud-free training images here
   ├── checkpoints/
   ├── outputs/
   └── test_holdout/     ← place a held-out image for generalization testing
   ```
4. Register on [Bhoonidhi](https://bhoonidhi.nrsc.gov.in) and download LISS-IV scenes (Satellite filter: ResourceSat-2 / ResourceSat-2A, Sensor Type: LISS4, Pricing: OpenData_DirectDownload)
5. Run all notebook cells top to bottom. Training checkpoints save to Drive every epoch — safe to resume after a Colab disconnect via the provided `load_checkpoint()` function.

---

## Local Usage

The core pipeline logic (cloud synthesis, dataset, model, training loop, inference) is also
available as plain Python modules under [`src/`](./src), refactored out of the notebook for
running from the command line instead of Colab. The notebook remains the canonical, fully
self-contained walkthrough — `src/` is an addition for local/offline use, not a replacement.

```
src/
├── cloud_generator.py   # synthetic fractal cloud mask + overlay
├── dataset.py           # SyntheticCloudDataset (clean image -> (cloudy, clean, mask))
├── model.py             # UNetGenerator + PatchDiscriminator
├── train.py             # CLI training loop, checkpoints every epoch
└── inference.py         # loads a checkpoint, runs cloud removal on one image
```

Install dependencies first (same `requirements.txt` as the notebook):

```bash
pip install -r requirements.txt
```

### Running inference on a sample image

Given a trained checkpoint (default expected at `checkpoints/skyclear_epoch65.pt`), run:

```bash
python -m src.inference --input path/to/sample.jpg --output outputs/inference_result.png
```

This resizes the input to the model's working resolution (128×128 by default — must stay
divisible by 16 to match the generator's 4 down/up-sampling stages), runs it through the
generator, and saves a side-by-side before/after PNG to `--output`. Useful flags:

- `--checkpoint <path>` — use a different trained checkpoint
- `--img-size <n>` — override the working resolution (must match how the checkpoint was trained)

Note: trained checkpoints are not committed to this repo (see [Repository Structure](#repository-structure))
— you need one produced by either the notebook or `src/train.py` before inference will work.

### Training from the command line

```bash
python -m src.train --data-dir data/clean --checkpoint-dir checkpoints --epochs 65 --batch-size 32
```

This mirrors the notebook's training loop (LSGAN adversarial loss + weighted L1, Adam,
checkpoint saved every epoch) with CLI flags instead of hardcoded notebook variables. Run
`python -m src.train --help` for the full list of options (learning rate, `--l1-lambda`,
`--resume <checkpoint>` to continue from a saved epoch, etc.). This has not been run
end-to-end in this environment (no GPU available) — the notebook's own training run (Colab,
T4 GPU) is what produced the results reported above.

---

## Interactive Demo

A Gradio web UI ([`app.py`](./app.py)) is available for trying the model interactively in a
browser instead of the CLI. It reuses `src/model.py` and the `load_generator`/`preprocess`/
`postprocess` helpers from `src/inference.py` directly — no model-loading or pre/post-processing
logic is duplicated.

```bash
pip install -r requirements.txt
python app.py
```

This loads the checkpoint at `checkpoints/skyclear_epoch65.pt` once at startup (kept in memory so
repeated uploads are fast, not reloaded per request), then prints a local URL — by default
`http://127.0.0.1:7860` — which opens automatically or can be pasted into your browser. Upload any
image, click **Remove Clouds**, and the resized (128×128) input and the model's cloud-removed
output are shown side by side. The checkpoint path is resolved relative to `app.py` itself, so the
demo can be launched from any working directory.

**Sample images:** a few ready-made cloudy images ship in [`examples/`](./examples) and appear
under the upload box — click one to load it instead of supplying your own image, then hit
**Remove Clouds**.

**Public link:** add `--share` to also generate a temporary public Gradio URL (handy for sharing a
live demo without deploying):

```bash
python app.py --share
```

**Robustness:** if no checkpoint exists at the expected path (or it can't be loaded), `app.py`
prints a clear message — with the `src/train.py` command to produce one — and exits instead of
crashing. Errors while processing an individual image surface as a clean in-UI notice rather than a
raw traceback.

---

## Roadmap

- [ ] Close the sim-to-real domain gap via improved synthetic cloud modeling or genuine paired acquisitions
- [ ] Non-reference image quality metrics (NIQE, BRISQUE) for evaluation on real cloudy scenes lacking ground truth
- [ ] Diffusion-based architecture (conditional DDPM) as a comparative baseline against this GAN
- [ ] Multi-modal fusion using Sentinel-1 SAR imagery (which penetrates cloud cover) as auxiliary input
- [ ] Operational deployment as a scalable workflow for persistently cloud-covered regions (e.g. India's North Eastern Region)

---

## Acknowledgments

Built for the **Bharatiya Antariksh Hackathon 2026**, organized by ISRO. LISS-IV imagery sourced from [Bhoonidhi](https://bhoonidhi.nrsc.gov.in), NRSC/ISRO's Earth Observation Data Hub.

---

## License

This project is released for educational and hackathon evaluation purposes. LISS-IV satellite data used in training is subject to Bhoonidhi's End User License Agreement and is not redistributed in this repository.
