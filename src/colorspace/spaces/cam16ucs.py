"""CAM16-UCS color space wrapper."""

import numpy as np
import colour

from colorspace.spaces.base import ColorSpace


class CAM16UCS(ColorSpace):
    """CAM16-UCS (Uniform Color Space) — Cartesian Jab form."""

    name = "CAM16-UCS"

    def from_XYZ(self, XYZ: np.ndarray) -> np.ndarray:
        """XYZ → CAM16-UCS (J', a', b')."""
        return colour.XYZ_to_CAM16UCS(XYZ)

    def to_XYZ(self, coords: np.ndarray) -> np.ndarray:
        """CAM16-UCS (J', a', b') → XYZ."""
        return colour.CAM16UCS_to_XYZ(coords)
