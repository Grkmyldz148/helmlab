"""IPT color space wrapper."""

import numpy as np
import colour

from helmlab.spaces.base import ColorSpace


class IPT(ColorSpace):
    """IPT color space (Ebner & Fairchild)."""

    name = "IPT"

    def from_XYZ(self, XYZ: np.ndarray) -> np.ndarray:
        """XYZ → IPT."""
        return colour.XYZ_to_IPT(XYZ)

    def to_XYZ(self, coords: np.ndarray) -> np.ndarray:
        """IPT → XYZ."""
        return colour.IPT_to_XYZ(coords)
