"""Color conversion utilities (vectorized numpy)."""

import numpy as np

from colorspace.config import D65_WHITE, LAB_EPSILON, LAB_KAPPA


def xyY_to_XYZ(xyY: np.ndarray) -> np.ndarray:
    """Convert xyY to XYZ (vectorized).

    Parameters
    ----------
    xyY : ndarray, shape (..., 3)
        Columns: x, y, Y

    Returns
    -------
    ndarray, shape (..., 3)
        Columns: X, Y, Z
    """
    xyY = np.asarray(xyY, dtype=np.float64)
    x = xyY[..., 0]
    y = xyY[..., 1]
    Y = xyY[..., 2]

    # Avoid division by zero
    mask = y > 0
    X = np.where(mask, x * Y / y, 0.0)
    Z = np.where(mask, (1.0 - x - y) * Y / y, 0.0)

    return np.stack([X, Y, Z], axis=-1)


def XYZ_to_xyY(XYZ: np.ndarray) -> np.ndarray:
    """Convert XYZ to xyY (vectorized).

    Parameters
    ----------
    XYZ : ndarray, shape (..., 3)

    Returns
    -------
    ndarray, shape (..., 3)
        Columns: x, y, Y
    """
    XYZ = np.asarray(XYZ, dtype=np.float64)
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    denom = X + Y + Z
    mask = denom > 0
    x = np.where(mask, X / denom, D65_WHITE[0] / sum(D65_WHITE))
    y = np.where(mask, Y / denom, D65_WHITE[1] / sum(D65_WHITE))
    return np.stack([x, y, Y], axis=-1)


def Lab_to_LCH(Lab: np.ndarray) -> np.ndarray:
    """Convert CIE Lab to LCH (vectorized).

    Parameters
    ----------
    Lab : ndarray, shape (..., 3)
        Columns: L, a, b

    Returns
    -------
    ndarray, shape (..., 3)
        Columns: L, C, H (H in degrees, 0-360)
    """
    Lab = np.asarray(Lab, dtype=np.float64)
    L = Lab[..., 0]
    a = Lab[..., 1]
    b = Lab[..., 2]
    C = np.sqrt(a**2 + b**2)
    H = np.degrees(np.arctan2(b, a)) % 360.0
    return np.stack([L, C, H], axis=-1)


def LCH_to_Lab(LCH: np.ndarray) -> np.ndarray:
    """Convert LCH to CIE Lab (vectorized).

    Parameters
    ----------
    LCH : ndarray, shape (..., 3)
        Columns: L, C, H (H in degrees)

    Returns
    -------
    ndarray, shape (..., 3)
        Columns: L, a, b
    """
    LCH = np.asarray(LCH, dtype=np.float64)
    L = LCH[..., 0]
    C = LCH[..., 1]
    H_rad = np.radians(LCH[..., 2])
    a = C * np.cos(H_rad)
    b = C * np.sin(H_rad)
    return np.stack([L, a, b], axis=-1)


def Lab_to_XYZ(Lab: np.ndarray, white: np.ndarray = D65_WHITE) -> np.ndarray:
    """Convert CIE Lab to XYZ (vectorized).

    Parameters
    ----------
    Lab : ndarray, shape (..., 3)
    white : ndarray, shape (3,)

    Returns
    -------
    ndarray, shape (..., 3)
    """
    Lab = np.asarray(Lab, dtype=np.float64)
    L, a, b = Lab[..., 0], Lab[..., 1], Lab[..., 2]

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    xr = np.where(fx**3 > LAB_EPSILON, fx**3, (116.0 * fx - 16.0) / LAB_KAPPA)
    yr = np.where(L > LAB_KAPPA * LAB_EPSILON, ((L + 16.0) / 116.0) ** 3, L / LAB_KAPPA)
    zr = np.where(fz**3 > LAB_EPSILON, fz**3, (116.0 * fz - 16.0) / LAB_KAPPA)

    X = xr * white[0]
    Y = yr * white[1]
    Z = zr * white[2]
    return np.stack([X, Y, Z], axis=-1)


def XYZ_to_Lab(XYZ: np.ndarray, white: np.ndarray = D65_WHITE) -> np.ndarray:
    """Convert XYZ to CIE Lab (vectorized).

    Parameters
    ----------
    XYZ : ndarray, shape (..., 3)
    white : ndarray, shape (3,)

    Returns
    -------
    ndarray, shape (..., 3)
    """
    XYZ = np.asarray(XYZ, dtype=np.float64)
    r = XYZ / white

    def f(t):
        return np.where(t > LAB_EPSILON, np.cbrt(t), (LAB_KAPPA * t + 16.0) / 116.0)

    fx, fy, fz = f(r[..., 0]), f(r[..., 1]), f(r[..., 2])
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)
