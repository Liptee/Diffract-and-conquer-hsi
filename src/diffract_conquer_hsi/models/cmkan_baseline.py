from __future__ import annotations

import torch.nn as nn

from diffract_conquer_hsi.models.gaussian_primitives import SpectralProjectorGate
from diffract_conquer_hsi.models.ggpir import GGPIR


class CmKANPlusPlus(nn.Module):
    """Baseline: cmKANlight backbone + spectral projector / rearranger (paper Sec. 3.2)."""

    def __init__(self, num_bands: int = 31, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.proj_down = nn.Conv2d(num_bands, 3, kernel_size, padding=pad)
        self.gate1 = SpectralProjectorGate(3, kernel_size)
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )
        self.rearr_up = nn.Conv2d(3, num_bands, kernel_size, padding=pad)
        self.gate2 = SpectralProjectorGate(num_bands, kernel_size)

    def forward(self, x):
        z = self.gate1(self.proj_down(x))
        z = self.body(z)
        return self.gate2(self.rearr_up(z))
