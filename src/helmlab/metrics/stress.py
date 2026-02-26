"""STRESS metric computation (Garcia et al., 2007)."""

import numpy as np


def stress(DV: np.ndarray, DE: np.ndarray) -> float:
    """Compute STRESS between visual differences and predicted color differences.

    STRESS = 100 * sqrt( sum((DV_i - F*DE_i)^2) / sum(DV_i^2) )

    where F = sum(DV_i * DE_i) / sum(DE_i^2) is the optimal scaling factor.

    Lower is better. 0 = perfect prediction.

    Parameters
    ----------
    DV : ndarray (N,) — visual (perceptual) differences (ground truth)
    DE : ndarray (N,) — predicted color differences (from a color space)

    Returns
    -------
    float — STRESS value (0-100 range typically)
    """
    DV = np.asarray(DV, dtype=np.float64).ravel()
    DE = np.asarray(DE, dtype=np.float64).ravel()

    if len(DV) != len(DE):
        raise ValueError(f"DV and DE must have same length, got {len(DV)} and {len(DE)}")

    # Optimal scaling factor (minimizes residual)
    DE_sq_sum = np.sum(DE ** 2)
    if DE_sq_sum == 0:
        return 100.0

    F = np.sum(DV * DE) / DE_sq_sum

    residual = DV - F * DE
    DV_sq_sum = np.sum(DV ** 2)

    if DV_sq_sum == 0:
        return 0.0

    return 100.0 * np.sqrt(np.sum(residual ** 2) / DV_sq_sum)
