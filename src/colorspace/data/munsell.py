"""Munsell Renotation Data loader and pair generation."""

import numpy as np
import colour

from colorspace.utils.conversions import xyY_to_XYZ

# Munsell hue families in order (40 hues total)
_HUE_FAMILIES = ["R", "YR", "Y", "GY", "G", "BG", "B", "PB", "P", "RP"]
_HUE_STEPS = [2.5, 5.0, 7.5, 10.0]

# Ordered list of 40 Munsell hue names
MUNSELL_HUES = [
    f"{step}{family}"
    for family in _HUE_FAMILIES
    for step in _HUE_STEPS
]

# Map hue name → angle (0–360, 9° per step)
HUE_TO_ANGLE = {name: i * 9.0 for i, name in enumerate(MUNSELL_HUES)}


def munsell_hue_to_angle(hue_name: str) -> float:
    """Convert a Munsell hue name like '5R' to an angle in [0, 360)."""
    return HUE_TO_ANGLE[hue_name]


def load_munsell(subset: str = "real") -> dict:
    """Load Munsell Renotation Data.

    Parameters
    ----------
    subset : str
        "real" (2734, physically realizable) or "all" (4995, includes extrapolated).

    Returns
    -------
    dict with keys:
        - hue_name: list[str], e.g. ['5R', '10YR', ...]
        - hue_angle: ndarray (N,), degrees 0-360
        - value: ndarray (N,), Munsell Value (0.2-10)
        - chroma: ndarray (N,), Munsell Chroma (2-50)
        - xyY: ndarray (N, 3)
        - XYZ: ndarray (N, 3)
    """
    raw = colour.MUNSELL_COLOURS[subset]

    hue_names = []
    hue_angles = []
    values = []
    chromas = []
    xyY_list = []

    for (hue, value, chroma), xyY in raw:
        if hue not in HUE_TO_ANGLE:
            continue  # skip any unexpected hue names
        hue_names.append(hue)
        hue_angles.append(munsell_hue_to_angle(hue))
        values.append(value)
        chromas.append(chroma)
        xyY_list.append(xyY)

    xyY_arr = np.array(xyY_list, dtype=np.float64)
    XYZ_arr = xyY_to_XYZ(xyY_arr)

    return {
        "hue_name": hue_names,
        "hue_angle": np.array(hue_angles, dtype=np.float64),
        "value": np.array(values, dtype=np.float64),
        "chroma": np.array(chromas, dtype=np.float64),
        "xyY": xyY_arr,
        "XYZ": XYZ_arr,
    }


def generate_munsell_pairs(munsell_data: dict) -> dict:
    """Generate neighbor pairs from Munsell data.

    Pairs are created between colors that differ by exactly 1 Munsell step
    in a single attribute (hue, value, or chroma) while the other two are held
    constant. Each pair has perceptual_distance = 1.0 (one Munsell step).

    Returns
    -------
    dict with keys:
        - XYZ_1: ndarray (P, 3)
        - XYZ_2: ndarray (P, 3)
        - perceptual_distance: ndarray (P,), all 1.0
        - pair_type: list[str], "hue" | "value" | "chroma"
    """
    # Build lookup: (hue_name, value, chroma) -> index
    n = len(munsell_data["hue_name"])
    lookup = {}
    for i in range(n):
        key = (munsell_data["hue_name"][i],
               munsell_data["value"][i],
               munsell_data["chroma"][i])
        lookup[key] = i

    pairs_i = []
    pairs_j = []
    pair_types = []

    # Hue neighbors: find the next hue in MUNSELL_HUES that exists in data
    # (the "real" subset may only have 20 of the 40 hues)
    present_hues = sorted(set(munsell_data["hue_name"]), key=lambda h: HUE_TO_ANGLE[h])
    hue_next = {present_hues[i]: present_hues[(i + 1) % len(present_hues)]
                for i in range(len(present_hues))}
    for i in range(n):
        hue = munsell_data["hue_name"][i]
        val = munsell_data["value"][i]
        chrom = munsell_data["chroma"][i]
        next_hue = hue_next[hue]
        key = (next_hue, val, chrom)
        if key in lookup:
            j = lookup[key]
            pairs_i.append(i)
            pairs_j.append(j)
            pair_types.append("hue")

    # Value neighbors: same hue & chroma, value differs by standard steps
    # Munsell values: 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    value_steps = {
        0.2: 0.4, 0.4: 0.6, 0.6: 0.8, 0.8: 1.0,
        1.0: 2.0, 2.0: 3.0, 3.0: 4.0, 4.0: 5.0,
        5.0: 6.0, 6.0: 7.0, 7.0: 8.0, 8.0: 9.0, 9.0: 10.0,
    }
    for i in range(n):
        hue = munsell_data["hue_name"][i]
        val = munsell_data["value"][i]
        chrom = munsell_data["chroma"][i]
        next_val = value_steps.get(val)
        if next_val is not None:
            key = (hue, next_val, chrom)
            if key in lookup:
                j = lookup[key]
                pairs_i.append(i)
                pairs_j.append(j)
                pair_types.append("value")

    # Chroma neighbors: same hue & value, chroma differs by 2
    for i in range(n):
        hue = munsell_data["hue_name"][i]
        val = munsell_data["value"][i]
        chrom = munsell_data["chroma"][i]
        next_chrom = chrom + 2.0
        key = (hue, val, next_chrom)
        if key in lookup:
            j = lookup[key]
            pairs_i.append(i)
            pairs_j.append(j)
            pair_types.append("chroma")

    XYZ = munsell_data["XYZ"]
    pairs_i = np.array(pairs_i)
    pairs_j = np.array(pairs_j)

    return {
        "XYZ_1": XYZ[pairs_i],
        "XYZ_2": XYZ[pairs_j],
        "perceptual_distance": np.ones(len(pairs_i), dtype=np.float64),
        "pair_type": pair_types,
    }
