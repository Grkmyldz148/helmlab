"""Abstract base class for color spaces."""

from abc import ABC, abstractmethod
import numpy as np


class ColorSpace(ABC):
    """Abstract color space with forward/inverse transforms and distance."""

    name: str = "base"

    @abstractmethod
    def from_XYZ(self, XYZ: np.ndarray) -> np.ndarray:
        """Convert from CIE XYZ to this color space.

        Parameters
        ----------
        XYZ : ndarray, shape (..., 3)

        Returns
        -------
        ndarray, shape (..., 3)
        """

    @abstractmethod
    def to_XYZ(self, coords: np.ndarray) -> np.ndarray:
        """Convert from this color space to CIE XYZ.

        Parameters
        ----------
        coords : ndarray, shape (..., 3)

        Returns
        -------
        ndarray, shape (..., 3)
        """

    def distance(self, XYZ_1: np.ndarray, XYZ_2: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance between two sets of colors in this space.

        Parameters
        ----------
        XYZ_1, XYZ_2 : ndarray, shape (..., 3)

        Returns
        -------
        ndarray, shape (...)
        """
        c1 = self.from_XYZ(XYZ_1)
        c2 = self.from_XYZ(XYZ_2)
        return np.sqrt(np.sum((c1 - c2) ** 2, axis=-1))

    def round_trip_error(self, XYZ: np.ndarray) -> np.ndarray:
        """Compute round-trip error: XYZ → space → XYZ.

        Returns
        -------
        ndarray, shape (...) — max absolute error per sample
        """
        coords = self.from_XYZ(XYZ)
        XYZ_rt = self.to_XYZ(coords)
        return np.max(np.abs(XYZ - XYZ_rt), axis=-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
