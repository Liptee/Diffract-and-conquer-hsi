from __future__ import annotations

import numpy as np


def harmonic_wavelengths(
    h_um: float,
    n_minus_one: float,
    max_order: int,
) -> np.ndarray:
    """λ_k = (n-1) h / k for integer k (paper Eq. 2 style, discrete harmonics)."""
    k = np.arange(1, max_order + 1, dtype=np.float64)
    return (n_minus_one * h_um) / k


def coverage_set(
    h_um: float,
    max_order: int,
    target_bands_nm: np.ndarray,
    n_minus_one: float,
    delta_lambda_nm: float,
) -> set[float]:
    covered: set[float] = set()
    lam_k = harmonic_wavelengths(h_um, n_minus_one, max_order)
    for lam in lam_k:
        for lt in target_bands_nm:
            if abs(float(lt) - float(lam)) < delta_lambda_nm:
                covered.add(float(lt))
    return covered


def greedy_max_coverage(
    candidates: list[tuple[float, int]],
    target_bands_nm: np.ndarray,
    n_minus_one: float,
    delta_lambda_nm: float,
    max_lenses: int,
) -> list[tuple[float, int]]:
    """Budgeted set cover-style selection; tie-break: smaller M, then smaller h (paper Sec. 3.1)."""
    targets = {float(x) for x in target_bands_nm}
    uncovered = set(targets)
    chosen: list[tuple[float, int]] = []
    remaining = sorted(candidates, key=lambda t: (t[1], t[0]))

    while uncovered and len(chosen) < max_lenses:
        best_idx = -1
        best_gain = 0
        best_cov: set[float] = set()
        for i, (h, m) in enumerate(remaining):
            cov = coverage_set(h, m, np.array(sorted(uncovered)), n_minus_one, delta_lambda_nm)
            gain = len(cov & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                best_cov = cov
        if best_gain == 0:
            break
        h, m = remaining.pop(best_idx)
        chosen.append((h, m))
        uncovered -= best_cov
    return chosen
