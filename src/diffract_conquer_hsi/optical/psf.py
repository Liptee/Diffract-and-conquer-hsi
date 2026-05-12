from __future__ import annotations

import torch


def fresnel_propagation_intensity(
    u_out_plane: torch.Tensor,
    grid_spacing_m: float,
    z_m: float,
    wavelength_m: torch.Tensor,
) -> torch.Tensor:
    """Placeholder angular-spectrum / Fresnel pipeline; replace with full FFT propagator."""
    raise NotImplementedError(
        "Implement Fresnel propagation (paper Eq. 5) for batch PSF simulation."
    )


def synthesize_psf_stack() -> torch.Tensor:
    """Return PSF_d(x,y,λ) per lens d; shape [D, H, W, num_lambda]."""
    raise NotImplementedError("Wire harmonic relief h_d(x,y) and multi-wavelength propagation.")
