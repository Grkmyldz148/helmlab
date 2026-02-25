"""JzCzHz / JzAzBz color space wrapper."""

import numpy as np
import colour

from colorspace.spaces.base import ColorSpace


class JzAzBz(ColorSpace):
    """JzAzBz color space (Cartesian form of JzCzHz).

    Uses Perceptual Quantizer (PQ) curve for HDR lightness.
    """

    name = "JzAzBz"

    def from_XYZ(self, XYZ: np.ndarray) -> np.ndarray:
        """XYZ → JzAzBz."""
        return colour.XYZ_to_Jzazbz(XYZ)

    def to_XYZ(self, coords: np.ndarray) -> np.ndarray:
        """JzAzBz → XYZ."""
        return colour.Jzazbz_to_XYZ(coords)
