"""Delta E color difference wrappers."""

import numpy as np
import colour

from colorspace.utils.conversions import XYZ_to_Lab, Lab_to_LCH


def delta_e_76(XYZ_1: np.ndarray, XYZ_2: np.ndarray) -> np.ndarray:
    """CIE76 Delta E (Euclidean distance in CIE Lab)."""
    Lab_1 = XYZ_to_Lab(XYZ_1)
    Lab_2 = XYZ_to_Lab(XYZ_2)
    return colour.delta_E(Lab_1, Lab_2, method="CIE 1976")


def delta_e_94(XYZ_1: np.ndarray, XYZ_2: np.ndarray) -> np.ndarray:
    """CIE94 Delta E."""
    Lab_1 = XYZ_to_Lab(XYZ_1)
    Lab_2 = XYZ_to_Lab(XYZ_2)
    return colour.delta_E(Lab_1, Lab_2, method="CIE 1994")


def delta_e_2000(XYZ_1: np.ndarray, XYZ_2: np.ndarray) -> np.ndarray:
    """CIEDE2000 Delta E."""
    Lab_1 = XYZ_to_Lab(XYZ_1)
    Lab_2 = XYZ_to_Lab(XYZ_2)
    return colour.delta_E(Lab_1, Lab_2, method="CIE 2000")


def delta_e_cmc(XYZ_1: np.ndarray, XYZ_2: np.ndarray) -> np.ndarray:
    """CMC l:c Delta E."""
    Lab_1 = XYZ_to_Lab(XYZ_1)
    Lab_2 = XYZ_to_Lab(XYZ_2)
    return colour.delta_E(Lab_1, Lab_2, method="CMC")


DELTA_E_METHODS = {
    "CIE76": delta_e_76,
    "CIE94": delta_e_94,
    "CIEDE2000": delta_e_2000,
    "CMC": delta_e_cmc,
}
