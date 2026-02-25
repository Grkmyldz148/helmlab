"""Visualization: STRESS charts, training curves, gamut plots."""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_stress_comparison(df: pd.DataFrame, save_path: Path | str | None = None):
    """Bar chart of STRESS scores for all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if s < 30 else "#f1c40f" if s < 40 else "#e74c3c"
              for s in df["STRESS"]]
    ax.barh(df["method"], df["STRESS"], color=colors)
    ax.set_xlabel("STRESS")
    ax.set_title("STRESS Benchmarks vs COMBVD")
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_training_curves(history: dict, save_path: Path | str | None = None):
    """Plot training loss and validation STRESS curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()

    ax2.plot(epochs, history["val_stress"], label="Val STRESS", color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("STRESS")
    ax2.set_title("Validation STRESS")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_gamut_slices(space, L_values: list[float] | None = None,
                     n_points: int = 200, save_path: Path | str | None = None):
    """Plot a-b plane slices at constant L values.

    Parameters
    ----------
    space : ColorSpace
        Any color space with from_XYZ / to_XYZ.
    L_values : list of float
        Lightness values to plot (in the space's L dimension).
    """
    if L_values is None:
        L_values = [0.3, 0.5, 0.7]

    fig, axes = plt.subplots(1, len(L_values), figsize=(5 * len(L_values), 5))
    if len(L_values) == 1:
        axes = [axes]

    # Generate grid of XYZ values
    for ax, L_val in zip(axes, L_values):
        # Sample a grid in XYZ space
        x = np.linspace(0.01, 1.0, n_points)
        y = np.linspace(0.01, 1.0, n_points)
        X, Y = np.meshgrid(x, y)
        Z_val = 0.5  # fix Z
        XYZ = np.stack([X.ravel(), Y.ravel(), np.full(n_points**2, Z_val)], axis=-1)

        coords = space.from_XYZ(XYZ)
        a = coords[:, 1].reshape(n_points, n_points)
        b = coords[:, 2].reshape(n_points, n_points)

        ax.scatter(a.ravel(), b.ravel(), c="steelblue", s=0.5, alpha=0.3)
        ax.set_xlabel("a")
        ax.set_ylabel("b")
        ax.set_title(f"XYZ grid (Z={Z_val:.1f})")
        ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_hue_linearity(space, hue_data: dict, save_path: Path | str | None = None):
    """Plot hue linearity: each hue line should be straight in a-b plane."""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors_cycle = plt.cm.hsv(np.linspace(0, 1, 12, endpoint=False))

    for i in range(12):
        mask = hue_data["hue_idx"] == i
        if not np.any(mask):
            continue
        XYZ = hue_data["XYZ"][mask]
        coords = space.from_XYZ(XYZ)
        a, b = coords[:, 1], coords[:, 2]
        ax.plot(a, b, "o-", color=colors_cycle[i], label=hue_data.get("hue_name", [f"Hue {i}"])[
            np.where(mask)[0][0] if "hue_name" in hue_data else 0
        ], markersize=4)

    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_title(f"Hue Linearity — {space.name}")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_round_trip_heatmap(errors: np.ndarray, save_path: Path | str | None = None):
    """Heatmap of round-trip errors."""
    fig, ax = plt.subplots(figsize=(8, 6))
    n = int(np.sqrt(len(errors)))
    if n * n == len(errors):
        data = errors.reshape(n, n)
    else:
        data = errors[:n*n].reshape(n, n)
    im = ax.imshow(data, cmap="hot", aspect="auto")
    fig.colorbar(im, ax=ax, label="Max |error|")
    ax.set_title("Round-Trip Error Heatmap")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig
