"""Hung & Berns (1995) constant perceived hue data loader."""

import numpy as np
import colour_datasets


HUE_NAMES = [
    "Red", "Red-yellow", "Yellow", "Yellow-green",
    "Green", "Green-cyan", "Cyan", "Cyan-blue",
    "Blue", "Blue-magenta", "Magenta", "Magenta-red",
]


def load_hung_berns() -> dict:
    """Load Hung & Berns constant hue loci data.

    Returns both CL (constant-lightness) and VL (variable-lightness) sets.
    All XYZ values are on the normalized scale (Y_white = 1.0).

    Returns
    -------
    dict with keys:
        - hue_lines: list of dicts, each with:
            - name: str
            - XYZ_ref: ndarray (3,) — chromaticity reference (saturated anchor)
            - XYZ_white: ndarray (3,) — D65 reference white
            - XYZ_cl: ndarray (K, 3) — constant-lightness test points
            - XYZ_vl: ndarray (M, 3) — variable-lightness test points
            - XYZ_all: ndarray (K+M, 3) — all test points combined
        - all_XYZ: ndarray (N, 3) — every test point across all hues
        - all_hue_idx: ndarray (N,) — hue index (0-11) for each point
        - all_hue_name: list[str] — hue name for each point
    """
    data = colour_datasets.load("3367463")
    cl_data = data["Constant Hue Loci Data - CL"]
    vl_data = data["Constant Hue Loci Data - VL"]

    hue_lines = []
    all_xyz = []
    all_hue_idx = []
    all_hue_name = []

    for idx, hue_name in enumerate(HUE_NAMES):
        cl = cl_data[hue_name]
        vl = vl_data[hue_name]

        xyz_cl = np.array(cl.XYZ_ct, dtype=np.float64)
        xyz_vl = np.array(vl.XYZ_ct, dtype=np.float64)
        xyz_all = np.vstack([xyz_cl, xyz_vl])

        hue_lines.append({
            "name": hue_name,
            "XYZ_ref": np.array(cl.XYZ_cr, dtype=np.float64),
            "XYZ_white": np.array(cl.XYZ_r, dtype=np.float64),
            "XYZ_cl": xyz_cl,
            "XYZ_vl": xyz_vl,
            "XYZ_all": xyz_all,
        })

        all_xyz.append(xyz_all)
        all_hue_idx.extend([idx] * len(xyz_all))
        all_hue_name.extend([hue_name] * len(xyz_all))

    return {
        "hue_lines": hue_lines,
        "all_XYZ": np.vstack(all_xyz),
        "all_hue_idx": np.array(all_hue_idx, dtype=np.int64),
        "all_hue_name": all_hue_name,
    }
