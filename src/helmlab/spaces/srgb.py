"""sRGB color space wrapper."""

import numpy as np
import colour

from helmlab.spaces.base import ColorSpace


class SRGB(ColorSpace):
    """sRGB color space (linear → gamma-encoded)."""

    name = "sRGB"

    def from_XYZ(self, XYZ: np.ndarray) -> np.ndarray:
        """XYZ → sRGB (gamma-encoded, [0,1] range)."""
        return colour.XYZ_to_sRGB(XYZ)

    def to_XYZ(self, coords: np.ndarray) -> np.ndarray:
        """sRGB → XYZ."""
        return colour.sRGB_to_XYZ(coords)
