from __future__ import annotations

import torch.nn as nn

from diffract_conquer_hsi.models.gaussian_primitives import (
    GeneratedGaussianPrimitivesLayer,
    SpectralFFN,
    SpectralProjectorGate,
)


class GGPIR(nn.Module):
    """
    Spectral projector → bottleneck → Gaussian primitive layer → spectral rearranger.
    Full paper model adds independent hypernetworks per output channel (Fig. 4).
    """

    def __init__(
        self,
        num_bands: int = 31,
        num_gaussian_outputs: int = 7,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        pad = kernel_size // 2
        self.proj_down = nn.Conv2d(num_bands, 3, kernel_size, padding=pad)
        self.proj_gate = SpectralProjectorGate(3, kernel_size)
        self.ffn = SpectralFFN(3, bottleneck=3)
        self.gpl = GeneratedGaussianPrimitivesLayer(3, num_gaussian_outputs)
        self.rearr_up = nn.Conv2d(num_gaussian_outputs, num_bands, kernel_size, padding=pad)
        self.rearr_gate = SpectralProjectorGate(num_bands, kernel_size)

    def forward(self, x):
        z = self.proj_gate(self.proj_down(x))
        z = self.ffn(z)
        mid = self.gpl(z)
        return self.rearr_gate(self.rearr_up(mid))
