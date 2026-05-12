from __future__ import annotations

import torch


def forward_measurement(
    spectral_cube: torch.Tensor,
    psf_stack: torch.Tensor,
    bayer_response: torch.Tensor,
    poisson_scale: float = 1.0,
    gaussian_std: float = 0.0,
) -> torch.Tensor:
    """
    spectral_cube: [B, C_lambda, H, W]
    psf_stack: [D, C_lambda, kH, kW] or pre-convolved operator
    bayer_response: [3, C_lambda] sensor spectral sensitivity T_c(λ)
    Returns encoded raw [B, D*3, H, W] (48 channels for 16 lenses × RGB).
    """
    raise NotImplementedError("Discrete version of paper Eqs. (3)-(6).")
