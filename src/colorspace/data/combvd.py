"""COMBVD (COM-corrected combined visual difference) data loader."""

import numpy as np
import pandas as pd

from colorspace.utils.io import download_file, load_xlsx

COMBVD_URL = "https://ndownloader.figshare.com/files/34948665"
COMBVD_FILENAME = "combvd.xlsx"


def load_combvd() -> dict:
    """Load the COMBVD dataset (3813 color pairs with visual differences).

    Downloads from Figshare on first call, caches locally.

    The dataset contains XYZ tristimulus values on a Y=100 scale.
    We normalize to Y=1.0 scale for consistency.

    Returns
    -------
    dict with keys:
        - XYZ_1: ndarray (3813, 3) — first sample XYZ (Y=1 scale)
        - XYZ_2: ndarray (3813, 3) — second sample XYZ (Y=1 scale)
        - XYZ_white: ndarray (3813, 3) — reference white per pair (Y=1 scale)
        - DV: ndarray (3813,) — visual difference
        - dataset: list[str] — sub-dataset name per pair
    """
    path = download_file(COMBVD_URL, COMBVD_FILENAME)

    df = load_xlsx(path, sheet_name="COM_Corrected_UNWEIGHTED", header=None,
                   skiprows=3)

    # Columns: Dataset, DV, X0, Y0, Z0, X1, Y1, Z1, X2, Y2, Z2, (empty)
    # Dataset column only has value on first row of each group
    df.columns = ["dataset", "DV", "X0", "Y0", "Z0",
                  "X1", "Y1", "Z1", "X2", "Y2", "Z2", "_empty"]
    df = df.drop(columns=["_empty"])

    # Forward-fill dataset name
    df["dataset"] = df["dataset"].ffill()

    # Drop any rows with NaN in DV (shouldn't happen but safety)
    df = df.dropna(subset=["DV"])

    # Extract arrays
    DV = df["DV"].to_numpy(dtype=np.float64)
    XYZ_white = df[["X0", "Y0", "Z0"]].to_numpy(dtype=np.float64) / 100.0
    XYZ_1 = df[["X1", "Y1", "Z1"]].to_numpy(dtype=np.float64) / 100.0
    XYZ_2 = df[["X2", "Y2", "Z2"]].to_numpy(dtype=np.float64) / 100.0
    datasets = df["dataset"].tolist()

    return {
        "XYZ_1": XYZ_1,
        "XYZ_2": XYZ_2,
        "XYZ_white": XYZ_white,
        "DV": DV,
        "dataset": datasets,
    }
