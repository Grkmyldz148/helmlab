"""MacAdam 1974 — Uniform color scales dataset.

128 color pairs from 59 ceramic tiles, with observed scale differences (DV).
Uses CIE 1964 10° observer, D65 illuminant.

Source: coloria-dev/color-data (GitHub)
Original: MacAdam, JOSA 64(12), 1691-1702 (1974).
"""

import numpy as np
import yaml

from helmlab.config import DATA_DIR

MACADAM_DIR = DATA_DIR / "macadam1974"


def _xyY_to_XYZ(x: float, y: float, Y: float) -> np.ndarray:
    """Convert CIE xyY to XYZ (Y normalized to 0-1 scale)."""
    Y01 = Y / 100.0
    X = x * Y01 / y
    Z = (1 - x - y) * Y01 / y
    return np.array([X, Y01, Z])


def load_macadam1974() -> dict:
    """Load MacAdam 1974 color difference dataset.

    Returns
    -------
    dict with keys:
        - XYZ_1: ndarray (128, 3) — first tile XYZ (Y=1 scale)
        - XYZ_2: ndarray (128, 3) — second tile XYZ (Y=1 scale)
        - DV: ndarray (128,) — observed scale difference
    """
    tiles_path = MACADAM_DIR / "table2.yaml"
    pairs_path = MACADAM_DIR / "table1.yaml"

    if not tiles_path.exists() or not pairs_path.exists():
        raise FileNotFoundError(
            f"MacAdam 1974 data not found at {MACADAM_DIR}. "
            f"Download table1.yaml and table2.yaml from "
            f"https://github.com/coloria-dev/color-data/tree/main/macadam1974"
        )

    with open(tiles_path) as f:
        tiles = yaml.safe_load(f)

    with open(pairs_path) as f:
        pairs = yaml.safe_load(f)

    XYZ_1_list, XYZ_2_list, DV_list = [], [], []

    for p in pairs:
        t1_id, t2_id = str(p[1]), str(p[2])
        dv = float(p[3])

        if t1_id not in tiles or t2_id not in tiles:
            continue

        t1, t2 = tiles[t1_id], tiles[t2_id]
        XYZ_1_list.append(_xyY_to_XYZ(t1[0], t1[1], t1[2]))
        XYZ_2_list.append(_xyY_to_XYZ(t2[0], t2[1], t2[2]))
        DV_list.append(dv)

    return {
        "XYZ_1": np.array(XYZ_1_list, dtype=np.float64),
        "XYZ_2": np.array(XYZ_2_list, dtype=np.float64),
        "DV": np.array(DV_list, dtype=np.float64),
    }
