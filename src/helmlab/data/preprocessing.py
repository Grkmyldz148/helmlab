"""Unified preprocessing: merge datasets into training-ready format."""

import logging

import numpy as np

from helmlab.data.munsell import load_munsell, generate_munsell_pairs
from helmlab.data.combvd import load_combvd
from helmlab.data.hung_berns import load_hung_berns

log = logging.getLogger(__name__)


def _try_load_he2022() -> dict | None:
    """Try to load He 2022 dataset. Returns None if not available."""
    try:
        from helmlab.data.he2022 import load_he2022
        return load_he2022()
    except (FileNotFoundError, ImportError):
        return None


def _try_load_macadam1974() -> dict | None:
    """Try to load MacAdam 1974 dataset. Returns None if not available."""
    try:
        from helmlab.data.macadam1974 import load_macadam1974
        return load_macadam1974()
    except (FileNotFoundError, ImportError):
        return None


def build_pair_data(include_extra: bool = False) -> dict:
    """Build unified pair dataset from COMBVD + Munsell.

    Parameters
    ----------
    include_extra : bool
        If True, also include He 2022 and MacAdam 1974 datasets.
        These use CIE 10° observer (vs 2° for COMBVD/Munsell), so
        mixing them can degrade model performance. Default False
        (use them only for cross-validation, not training).

    Returns
    -------
    dict with keys:
        - XYZ_1: ndarray (N, 3)
        - XYZ_2: ndarray (N, 3)
        - DV: ndarray (N,) — normalized visual differences
        - source: list[str] — "combvd", "munsell", etc.
    """
    # COMBVD pairs
    combvd = load_combvd()
    n_combvd = len(combvd["DV"])

    # Munsell pairs
    munsell = load_munsell("real")
    munsell_pairs = generate_munsell_pairs(munsell)
    n_munsell = len(munsell_pairs["perceptual_distance"])

    # Normalize visual differences:
    # COMBVD DV values are on arbitrary scale, Munsell are all 1.0
    # Normalize COMBVD to [0, 1] range, then scale Munsell accordingly
    combvd_dv = combvd["DV"]
    combvd_max = combvd_dv.max()
    combvd_norm = combvd_dv / combvd_max

    # Munsell 1-step pairs: set to median COMBVD normalized value
    munsell_dv = np.full(n_munsell, np.median(combvd_norm))

    # Merge base datasets
    xyz1_parts = [combvd["XYZ_1"], munsell_pairs["XYZ_1"]]
    xyz2_parts = [combvd["XYZ_2"], munsell_pairs["XYZ_2"]]
    dv_parts = [combvd_norm, munsell_dv]
    source = ["combvd"] * n_combvd + ["munsell"] * n_munsell

    if include_extra:
        # He 2022 (10° observer — only include when explicitly requested)
        he2022 = _try_load_he2022()
        if he2022 is not None:
            n_he = len(he2022["DV"])
            he_norm = he2022["DV"] / combvd_max
            xyz1_parts.append(he2022["XYZ_1"])
            xyz2_parts.append(he2022["XYZ_2"])
            dv_parts.append(he_norm)
            source += ["he2022"] * n_he
            log.info(f"He 2022: {n_he} pairs added")

        # MacAdam 1974 (10° observer — only include when explicitly requested)
        macadam = _try_load_macadam1974()
        if macadam is not None:
            n_mac = len(macadam["DV"])
            mac_norm = macadam["DV"] / combvd_max
            xyz1_parts.append(macadam["XYZ_1"])
            xyz2_parts.append(macadam["XYZ_2"])
            dv_parts.append(mac_norm)
            source += ["macadam1974"] * n_mac
            log.info(f"MacAdam 1974: {n_mac} pairs added")

    XYZ_1 = np.vstack(xyz1_parts)
    XYZ_2 = np.vstack(xyz2_parts)
    DV = np.concatenate(dv_parts)

    return {
        "XYZ_1": XYZ_1,
        "XYZ_2": XYZ_2,
        "DV": DV,
        "source": source,
        "combvd_max": combvd_max,
    }


def build_hue_data() -> dict:
    """Build hue linearity auxiliary data from Hung & Berns.

    Returns
    -------
    dict with keys:
        - XYZ: ndarray (N, 3) — all test points
        - hue_idx: ndarray (N,) — hue group index (0-11)
        - hue_name: list[str]
    """
    hb = load_hung_berns()
    return {
        "XYZ": hb["all_XYZ"],
        "hue_idx": hb["all_hue_idx"],
        "hue_name": hb["all_hue_name"],
    }
