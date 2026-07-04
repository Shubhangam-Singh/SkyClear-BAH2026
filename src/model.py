"""U-Net generator and PatchGAN discriminator (pix2pix-style conditional GAN).

Ported from the ``Step 4 — Model`` cell of ``SkyClear_Day1_Pipeline_(1).ipynb``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetGenerator(nn.Module):
    """Encoder-decoder generator with skip connections.

    4 downsampling + 4 upsampling stages. Input spatial size must be
    divisible by 16 (128x128 by default, matching training) so that the
    skip-connection concatenations line up.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 32) -> None:
        super().__init__()

        def block(i: int, o: int, down: bool = True) -> nn.Sequential:
            layers = [nn.Conv2d(i, o, 4, 2, 1) if down else nn.ConvTranspose2d(i, o, 4, 2, 1)]
            layers.append(nn.InstanceNorm2d(o))
            layers.append(nn.LeakyReLU(0.2) if down else nn.ReLU())
            return nn.Sequential(*layers)

        self.e1 = nn.Conv2d(in_ch, base, 4, 2, 1)               # 128 -> 64
        self.e2 = block(base, base * 2)                          # 64 -> 32
        self.e3 = block(base * 2, base * 4)                      # 32 -> 16
        self.e4 = block(base * 4, base * 8)                      # 16 -> 8

        self.d1 = block(base * 8, base * 4, down=False)          # 8 -> 16
        self.d2 = block(base * 8, base * 2, down=False)          # 16 -> 32 (concat skip)
        self.d3 = block(base * 4, base, down=False)              # 32 -> 64 (concat skip)
        self.d4 = nn.ConvTranspose2d(base * 2, out_ch, 4, 2, 1)  # 64 -> 128 (concat skip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.leaky_relu(e1, 0.2))
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        d1 = self.d1(F.relu(e4))
        d2 = self.d2(torch.cat([d1, e3], 1))
        d3 = self.d3(torch.cat([d2, e2], 1))
        d4 = self.d4(torch.cat([d3, e1], 1))
        return torch.tanh(d4)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator.

    Classifies overlapping patches as real/fake rather than the whole image
    (encourages sharper local detail), conditioned on the cloudy input.
    """

    def __init__(self, in_ch: int = 6, base: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.InstanceNorm2d(base * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.InstanceNorm2d(base * 4), nn.LeakyReLU(0.2),
            nn.Conv2d(base * 4, 1, 4, 1, 1),
        )

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """Args:
            img_a: conditioning image, e.g. the cloudy input, ``(N, 3, H, W)``.
            img_b: image being judged real/fake, e.g. clean or generated, ``(N, 3, H, W)``.
        """
        x = torch.cat([img_a, img_b], 1)
        return self.net(x)
