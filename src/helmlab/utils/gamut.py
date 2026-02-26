"""Adaptive gamut mapping — chroma-reduction with hue and lightness preservation.

Supports sRGB and Display P3 gamuts.
"""

import numpy as np

from helmlab.utils.srgb_convert import (
    M_XYZ_TO_SRGB,
    M_XYZ_TO_DISPLAYP3,
)

# Gamut → linear-RGB matrix mapping
_GAMUT_MATRICES = {
    "srgb": M_XYZ_TO_SRGB,
    "display-p3": M_XYZ_TO_DISPLAYP3,
}


def _linear_rgb(XYZ: np.ndarray, gamut: str) -> np.ndarray:
    """XYZ → linear RGB for the given gamut (no gamma)."""
    M = _GAMUT_MATRICES[gamut]
    return np.asarray(XYZ, dtype=np.float64) @ M.T


def is_in_gamut(lab: np.ndarray, space, gamut: str = "srgb", tol: float = 1e-4) -> np.ndarray:
    """Check whether Lab coordinates are inside the specified gamut.

    Parameters
    ----------
    lab : ndarray, shape (..., 3)
    space : ColorSpace instance (must have .to_XYZ)
    gamut : "srgb" or "display-p3"
    tol : tolerance for boundary inclusion

    Returns
    -------
    bool ndarray, shape (...)
    """
    lab = np.asarray(lab, dtype=np.float64)
    XYZ = space.to_XYZ(lab)
    rgb = _linear_rgb(XYZ, gamut)
    return np.all((rgb >= -tol) & (rgb <= 1.0 + tol), axis=-1)


def max_chroma(L: float, H_rad: float, space, gamut: str = "srgb", tol: float = 1e-4) -> float:
    """Binary search for maximum in-gamut chroma at fixed L and H.

    Parameters
    ----------
    L : lightness value
    H_rad : hue angle in radians
    space : ColorSpace with to_XYZ method
    gamut : "srgb" or "display-p3"
    tol : convergence tolerance

    Returns
    -------
    float — maximum chroma that stays in gamut
    """
    cos_h = np.cos(H_rad)
    sin_h = np.sin(H_rad)

    lo, hi = 0.0, 1.0

    # First expand hi until it's out of gamut
    lab_test = np.array([L, hi * cos_h, hi * sin_h])
    while is_in_gamut(lab_test, space, gamut, tol):
        hi *= 2.0
        if hi > 100.0:  # safety
            return hi
        lab_test = np.array([L, hi * cos_h, hi * sin_h])

    # Binary search
    for _ in range(50):
        mid = (lo + hi) * 0.5
        lab_test = np.array([L, mid * cos_h, mid * sin_h])
        if is_in_gamut(lab_test, space, gamut, tol):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    return lo


def gamut_map(lab: np.ndarray, space, gamut: str = "srgb", method: str = "chroma") -> np.ndarray:
    """Map Lab coordinates into the specified gamut.

    method="chroma": Reduce chroma while preserving L and hue.

    Parameters
    ----------
    lab : ndarray, shape (3,) or (N, 3)
    space : ColorSpace with to_XYZ method
    gamut : "srgb" or "display-p3"
    method : "chroma" (only supported method currently)

    Returns
    -------
    ndarray, same shape as input — gamut-mapped Lab
    """
    lab = np.asarray(lab, dtype=np.float64)
    if lab.ndim == 1:
        return _gamut_map_single(lab, space, gamut)
    return gamut_map_batch(lab, space, gamut)


def _gamut_map_single(lab: np.ndarray, space, gamut: str) -> np.ndarray:
    """Gamut-map a single Lab triplet via chroma reduction."""
    if is_in_gamut(lab, space, gamut):
        return lab.copy()

    L = lab[0]
    a, b = lab[1], lab[2]
    C = np.sqrt(a ** 2 + b ** 2)
    H = np.arctan2(b, a)

    if C < 1e-10:
        # Achromatic but OOG — clamp L to achievable range
        result = lab.copy()
        result[1] = 0.0
        result[2] = 0.0
        return result

    C_max = max_chroma(L, H, space, gamut)
    C_new = min(C, C_max)
    return np.array([L, C_new * np.cos(H), C_new * np.sin(H)])


def gamut_map_batch(labs: np.ndarray, space, gamut: str = "srgb") -> np.ndarray:
    """Vectorized batch gamut mapping.

    Parameters
    ----------
    labs : ndarray, shape (N, 3)
    space : ColorSpace with to_XYZ method
    gamut : "srgb" or "display-p3"

    Returns
    -------
    ndarray, shape (N, 3) — gamut-mapped Lab values
    """
    labs = np.asarray(labs, dtype=np.float64)
    in_gamut = is_in_gamut(labs, space, gamut)
    result = labs.copy()

    oog_idx = np.where(~in_gamut)[0]
    for i in oog_idx:
        result[i] = _gamut_map_single(labs[i], space, gamut)

    return result
