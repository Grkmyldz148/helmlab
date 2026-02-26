"""OKLCH color space wrapper (via Oklab)."""

import numpy as np
import colour

from helmlab.spaces.base import ColorSpace


class OKLCH(ColorSpace):
    """Oklab/OKLCH color space.

    Stores colors in Cartesian Oklab form (L, a, b) internally for distance
    computation. OKLCH (L, C, H) is the polar form.
    """

    name = "OKLCH"

    def from_XYZ(self, XYZ: np.ndarray) -> np.ndarray:
        """XYZ → Oklab (L, a, b)."""
        return colour.XYZ_to_Oklab(XYZ)

    def to_XYZ(self, coords: np.ndarray) -> np.ndarray:
        """Oklab (L, a, b) → XYZ."""
        return colour.Oklab_to_XYZ(coords)
