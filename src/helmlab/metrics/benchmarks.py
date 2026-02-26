"""Benchmark all color spaces against datasets."""

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from helmlab.data.combvd import load_combvd
from helmlab.metrics.stress import stress
from helmlab.metrics.delta_e import DELTA_E_METHODS
from helmlab.spaces.registry import get_space, all_spaces
from helmlab.spaces.base import ColorSpace


def compute_stress_for_space(
    space: ColorSpace,
    XYZ_1: np.ndarray,
    XYZ_2: np.ndarray,
    DV: np.ndarray,
) -> float:
    """Compute STRESS for a color space against visual differences."""
    DE = space.distance(XYZ_1, XYZ_2)
    return stress(DV, DE)


def compute_stress_for_delta_e(
    method_name: str,
    XYZ_1: np.ndarray,
    XYZ_2: np.ndarray,
    DV: np.ndarray,
) -> float:
    """Compute STRESS for a Delta E method against visual differences."""
    de_fn = DELTA_E_METHODS[method_name]
    DE = de_fn(XYZ_1, XYZ_2)
    return stress(DV, DE)


def run_all_baselines(extra_spaces: list[ColorSpace] | None = None) -> pd.DataFrame:
    """Run STRESS benchmarks for all spaces and Delta E methods against COMBVD.

    Parameters
    ----------
    extra_spaces : list of ColorSpace, optional
        Additional spaces to include (e.g., neural model).

    Returns
    -------
    DataFrame with columns: method, STRESS
    """
    combvd = load_combvd()
    XYZ_1 = combvd["XYZ_1"]
    XYZ_2 = combvd["XYZ_2"]
    DV = combvd["DV"]

    results = []

    # Color space distances
    spaces = all_spaces()
    if extra_spaces:
        spaces.extend(extra_spaces)

    for space in spaces:
        try:
            s = compute_stress_for_space(space, XYZ_1, XYZ_2, DV)
            results.append({"method": f"{space.name} (Euclidean)", "STRESS": s})
        except Exception as e:
            results.append({"method": f"{space.name} (Euclidean)", "STRESS": float("nan")})

    # Delta E methods
    for name in DELTA_E_METHODS:
        try:
            s = compute_stress_for_delta_e(name, XYZ_1, XYZ_2, DV)
            results.append({"method": f"ΔE {name}", "STRESS": s})
        except Exception as e:
            results.append({"method": f"ΔE {name}", "STRESS": float("nan")})

    df = pd.DataFrame(results).sort_values("STRESS").reset_index(drop=True)
    return df


def print_baselines(df: pd.DataFrame) -> None:
    """Pretty-print baseline results using rich."""
    console = Console()
    table = Table(title="STRESS Benchmarks vs COMBVD")
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Method", style="bold")
    table.add_column("STRESS", justify="right")

    for i, row in df.iterrows():
        s = row["STRESS"]
        style = "green" if s < 30 else "yellow" if s < 40 else "red"
        table.add_row(str(i + 1), row["method"], f"[{style}]{s:.2f}[/{style}]")

    console.print(table)
