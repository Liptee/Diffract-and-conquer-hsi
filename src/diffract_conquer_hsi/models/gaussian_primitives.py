from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralProjectorGate(nn.Module):
    """x + x * conv2d(x)^2 — PSF-aware gating (paper Sec. 3.2)."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.conv(x)
        return x + x * (g * g)


class SpectralFFN(nn.Module):
    """Bottleneck FFN (reduces N→3 then expands); align channels with paper Fig. 4."""

    def __init__(self, in_ch: int, bottleneck: int = 3, hidden: int | None = None) -> None:
        super().__init__()
        h = hidden or max(in_ch * 2, 32)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, h, 1),
            nn.GELU(),
            nn.Conv2d(h, bottleneck, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GeneratedGaussianPrimitivesLayer(nn.Module):
    """Linear combination of 1D Gaussians over input channels (paper Eq. 10)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.a = nn.Parameter(torch.zeros(out_ch, in_ch))
        self.b = nn.Parameter(torch.zeros(out_ch, in_ch))
        self.d = nn.Parameter(torch.ones(out_ch))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        b, c, h, w = s.shape
        if c != self.in_ch:
            raise ValueError(f"expected {self.in_ch} channels, got {c}")
        s_flat = s.view(b, c, -1)
        z = s_flat.unsqueeze(1) * self.a.unsqueeze(-1) + self.b.unsqueeze(-1)
        g = torch.exp(-0.5 * z * z)
        out = (g.sum(dim=2) * self.d.view(1, self.out_ch, 1)).view(b, self.out_ch, h, w)
        return out
