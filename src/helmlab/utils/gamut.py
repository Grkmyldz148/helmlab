"""Gamut mapping — chroma-reduction and adaptive (Ottosson-style) clipping.

Supports sRGB and Display P3 gamuts.

Methods:
    "chroma"   — reduce chroma at fixed L and hue (default, backward compat)
    "adaptive" — Ottosson-style adaptive L0 clipping that shifts both L and C
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


def find_cusp(H_rad: float, space, gamut: str = "srgb",
              n_scan: int = 64, tol: float = 1e-4) -> tuple[float, float]:
    """Find the cusp (L_cusp, C_cusp) at a given hue angle.

    The cusp is the point on the gamut boundary where chroma is maximized.
    Uses a coarse scan followed by golden-section refinement.

    Parameters
    ----------
    H_rad : hue angle in radians
    space : ColorSpace with to_XYZ method
    gamut : "srgb" or "display-p3"
    n_scan : number of L values in coarse scan
    tol : convergence tolerance for L

    Returns
    -------
    (L_cusp, C_cusp) — lightness and chroma at the cusp
    """
    # Coarse scan: find L that gives maximum chroma
    # Skip degenerate values where max_chroma returns unbounded (>10)
    L_vals = np.linspace(0.05, 0.98, n_scan)
    best_L = 0.5
    best_C = 0.0
    for L in L_vals:
        C = max_chroma(L, H_rad, space, gamut, tol)
        if C > 10.0:
            # Degenerate: gamut appears unbounded at this L, skip
            continue
        if C > best_C:
            best_C = C
            best_L = L

    # Golden-section refinement around the best L
    golden = (np.sqrt(5) - 1) / 2
    idx = int(np.argmin(np.abs(L_vals - best_L)))
    a = L_vals[max(0, idx - 2)]
    b = L_vals[min(len(L_vals) - 1, idx + 2)]

    for _ in range(30):
        if b - a < tol:
            break
        c = b - golden * (b - a)
        d = a + golden * (b - a)
        Cc = max_chroma(c, H_rad, space, gamut, tol)
        Cd = max_chroma(d, H_rad, space, gamut, tol)
        # Treat degenerate values as zero for comparison
        if Cc > 10.0:
            Cc = 0.0
        if Cd > 10.0:
            Cd = 0.0
        if Cc > Cd:
            b = d
        else:
            a = c

    L_cusp = (a + b) / 2
    C_cusp = max_chroma(L_cusp, H_rad, space, gamut, tol)
    if C_cusp > 10.0:
        C_cusp = best_C  # fallback to coarse scan result
        L_cusp = best_L
    return L_cusp, C_cusp


def _gamut_clip_adaptive_single(lab: np.ndarray, space, gamut: str,
                                 alpha: float) -> np.ndarray:
    """Adaptive gamut clipping for a single Lab triplet (Ottosson-style).

    Instead of only reducing chroma (which preserves L exactly),
    this method adaptively shifts both L and C toward the gamut boundary.

    The projection target L0 is computed adaptively:
    - For colors near mid-lightness: mostly chroma reduction
    - For very light/dark out-of-gamut colors: allows L to shift
    - alpha controls the blend: 0 = pure chroma reduction, larger = more L shift

    This is especially important for yellow where the cusp is at very high L.
    Pure chroma reduction at high L gives near-zero chroma (the cliff problem).
    Adaptive clipping can shift L down toward the cusp, preserving chroma.
    """
    if is_in_gamut(lab, space, gamut):
        return lab.copy()

    L = float(lab[0])
    a, b = float(lab[1]), float(lab[2])
    C = np.sqrt(a ** 2 + b ** 2)
    H = np.arctan2(b, a)

    if C < 1e-10:
        # Achromatic but OOG — clamp L to achievable range
        result = lab.copy()
        result[1] = 0.0
        result[2] = 0.0
        return result

    # Find cusp at this hue
    L_cusp, C_cusp = find_cusp(H, space, gamut)

    # Compute adaptive L0 (Ottosson's formula adapted for [0,1] L range)
    Ld = L - L_cusp
    if Ld >= 0:
        k = 2.0 * (1.0 - L_cusp)
    else:
        k = 2.0 * L_cusp

    # Guard against degenerate k (cusp at extreme L)
    k = max(k, 1e-6)

    e1 = k / 2.0 + abs(Ld) + alpha * C / k
    discriminant = e1 * e1 - 2.0 * k * abs(Ld)

    if discriminant < 0:
        # Fallback: L0 = L_cusp
        L0 = L_cusp
    else:
        sgn = 1.0 if Ld >= 0 else -1.0
        L0 = L_cusp + sgn * (e1 - np.sqrt(discriminant)) / 2.0

    # Clamp L0 to valid range
    L0 = float(np.clip(L0, 0.0, 1.0))

    # Binary search for t in [0, 1] on the line from (L0, 0) to (L, C)
    # such that the point (L0*(1-t) + t*L, t*C) is on the gamut boundary
    cos_h = np.cos(H)
    sin_h = np.sin(H)

    lo_t, hi_t = 0.0, 1.0

    for _ in range(50):
        t = (lo_t + hi_t) * 0.5
        L_test = L0 * (1.0 - t) + t * L
        C_test = t * C
        lab_test = np.array([L_test, C_test * cos_h, C_test * sin_h])
        if is_in_gamut(lab_test, space, gamut):
            lo_t = t
        else:
            hi_t = t
        if hi_t - lo_t < 1e-6:
            break

    # Use the last in-gamut value
    t = lo_t
    L_clip = L0 * (1.0 - t) + t * L
    C_clip = t * C
    return np.array([L_clip, C_clip * cos_h, C_clip * sin_h])


def gamut_map(lab: np.ndarray, space, gamut: str = "srgb", method: str = "chroma",
              alpha: float = 0.05) -> np.ndarray:
    """Map Lab coordinates into the specified gamut.

    Parameters
    ----------
    lab : ndarray, shape (3,) or (N, 3)
    space : ColorSpace with to_XYZ method
    gamut : "srgb" or "display-p3"
    method : "chroma" or "adaptive"
        "chroma" — reduce chroma while preserving L and hue (default)
        "adaptive" — Ottosson-style adaptive clipping that shifts both L and C
    alpha : float
        Blend parameter for adaptive method (default 0.05).
        0 = pure chroma reduction, larger = more L shift allowed.
        Ignored when method="chroma".

    Returns
    -------
    ndarray, same shape as input — gamut-mapped Lab
    """
    lab = np.asarray(lab, dtype=np.float64)
    if lab.ndim == 1:
        return _gamut_map_single(lab, space, gamut, method=method, alpha=alpha)
    return gamut_map_batch(lab, space, gamut, method=method, alpha=alpha)


def _gamut_map_single(lab: np.ndarray, space, gamut: str,
                      method: str = "chroma", alpha: float = 0.05) -> np.ndarray:
    """Gamut-map a single Lab triplet.

    Dispatches to chroma reduction or adaptive clipping based on method.
    """
    if method == "adaptive":
        return _gamut_clip_adaptive_single(lab, space, gamut, alpha)

    # Default: chroma reduction
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


def gamut_map_batch(labs: np.ndarray, space, gamut: str = "srgb",
                    method: str = "chroma", alpha: float = 0.05) -> np.ndarray:
    """Batch gamut mapping.

    Parameters
    ----------
    labs : ndarray, shape (N, 3)
    space : ColorSpace with to_XYZ method
    gamut : "srgb" or "display-p3"
    method : "chroma" or "adaptive"
    alpha : blend parameter for adaptive method

    Returns
    -------
    ndarray, shape (N, 3) — gamut-mapped Lab values
    """
    labs = np.asarray(labs, dtype=np.float64)
    in_gamut_mask = is_in_gamut(labs, space, gamut)
    result = labs.copy()

    oog_idx = np.where(~in_gamut_mask)[0]
    for i in oog_idx:
        result[i] = _gamut_map_single(labs[i], space, gamut,
                                       method=method, alpha=alpha)

    return result
