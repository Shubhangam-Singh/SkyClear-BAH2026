"""Training loop for the SkyClear cloud-removal GAN.

Ported from the ``Step 5 — Losses, Optimizers, Checkpointing`` and
``Step 6 — Training Loop`` cells of ``SkyClear_Day1_Pipeline_(1).ipynb``.

Usage:
    python -m src.train --data-dir data/clean --checkpoint-dir checkpoints --epochs 65 --batch-size 32
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import SyntheticCloudDataset
from .model import PatchDiscriminator, UNetGenerator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a training run."""
    parser = argparse.ArgumentParser(description="Train the SkyClear cloud-removal GAN.")
    parser.add_argument("--data-dir", type=str, default="data/clean", help="Directory of clean training images.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints to.")
    parser.add_argument("--epochs", type=int, default=65, help="Total epoch count to train up to (or resume to).")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--img-size", type=int, default=128, help="Image resolution; must be divisible by 16.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate for generator and discriminator.")
    parser.add_argument("--l1-lambda", type=float, default=150.0, help="Weight on the L1 reconstruction loss.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker process count.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume training from.")
    args = parser.parse_args()
    assert args.img_size % 16 == 0, "--img-size must be divisible by 16 (generator has 4 down/up stages)."
    return args


def save_checkpoint(
    path: str,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
) -> None:
    """Save model/optimizer state for a completed epoch."""
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
        },
        path,
    )
    print(f"  Checkpoint saved -> {path}")


def load_checkpoint(
    path: str,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """Load a checkpoint in-place into the given models/optimizers.

    Returns:
        The epoch number the checkpoint was saved at.
    """
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    opt_g.load_state_dict(ckpt["opt_g"])
    opt_d.load_state_dict(ckpt["opt_d"])
    print(f"Resumed from epoch {ckpt['epoch']}")
    return ckpt["epoch"]


def main() -> None:
    """Run the full training loop end-to-end, checkpointing every epoch."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    dataset = SyntheticCloudDataset(args.data_dir, img_size=args.img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Dataset size: {len(dataset)} images, batch size: {args.batch_size}, batches/epoch: {len(dataloader)}")

    generator = UNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    criterion_gan = nn.MSELoss()  # LSGAN-style stability, easier to train than vanilla BCE GAN
    criterion_l1 = nn.L1Loss()  # pixel-level reconstruction loss

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, generator, discriminator, opt_g, opt_d, device)

    for epoch in range(start_epoch, args.epochs):
        g_losses, d_losses = [], []
        for cloudy, clean, _mask in dataloader:
            cloudy, clean = cloudy.to(device), clean.to(device)

            opt_d.zero_grad()
            fake = generator(cloudy)
            pred_real = discriminator(cloudy, clean)
            pred_fake = discriminator(cloudy, fake.detach())
            loss_d = 0.5 * (
                criterion_gan(pred_real, torch.ones_like(pred_real))
                + criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            )
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad()
            pred_fake = discriminator(cloudy, fake)
            loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_g_l1 = criterion_l1(fake, clean) * args.l1_lambda
            loss_g = loss_g_gan + loss_g_l1
            loss_g.backward()
            opt_g.step()

            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())

        print(f"Epoch {epoch + 1}/{args.epochs} | G loss: {np.mean(g_losses):.3f} | D loss: {np.mean(d_losses):.3f}")
        save_checkpoint(
            os.path.join(args.checkpoint_dir, f"skyclear_epoch{epoch + 1}.pt"),
            epoch + 1,
            generator,
            discriminator,
            opt_g,
            opt_d,
        )


if __name__ == "__main__":
    main()
