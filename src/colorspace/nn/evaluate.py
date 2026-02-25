"""Post-training evaluation: NeuralColorSpace wrapper + metrics."""

from pathlib import Path

import numpy as np
import torch

from colorspace.spaces.base import ColorSpace
from colorspace.config import TrainConfig, CHECKPOINT_DIR
from colorspace.nn.inn import ColorINN
from colorspace.nn.mlp import ColorMLP
from colorspace.nn.training import get_device


class NeuralColorSpace(ColorSpace):
    """Wraps a trained neural model as a ColorSpace for benchmarking."""

    def __init__(self, model: torch.nn.Module, device: torch.device | None = None, name: str = "Neural"):
        self._name = name
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    def from_XYZ(self, XYZ: np.ndarray) -> np.ndarray:
        """XYZ → learned perceptual space."""
        XYZ = np.asarray(XYZ, dtype=np.float32)
        original_shape = XYZ.shape
        flat = XYZ.reshape(-1, 3)

        with torch.no_grad():
            t = torch.tensor(flat, dtype=torch.float32, device=self.device)
            out = self.model(t).cpu().numpy()

        return out.reshape(original_shape)

    def to_XYZ(self, coords: np.ndarray) -> np.ndarray:
        """Learned perceptual space → XYZ."""
        coords = np.asarray(coords, dtype=np.float32)
        original_shape = coords.shape
        flat = coords.reshape(-1, 3)

        with torch.no_grad():
            t = torch.tensor(flat, dtype=torch.float32, device=self.device)
            out = self.model.inverse(t).cpu().numpy()

        return out.reshape(original_shape)

    @classmethod
    def from_checkpoint(
        cls,
        path: Path | str | None = None,
        model_type: str = "inn",
    ) -> "NeuralColorSpace":
        """Load from a saved checkpoint."""
        if path is None:
            path = CHECKPOINT_DIR / f"{model_type}_best.pt"
        path = Path(path)

        device = get_device()
        checkpoint = torch.load(path, map_location=device, weights_only=True)

        cfg = TrainConfig()
        saved_cfg = checkpoint.get("config", {})
        for k, v in saved_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        mt = checkpoint.get("model_type", model_type)
        if mt == "inn":
            model = ColorINN(cfg)
        else:
            model = ColorMLP(cfg)

        model.load_state_dict(checkpoint["model_state_dict"])
        name = f"Neural-{mt.upper()}"
        return cls(model, device, name)


def evaluate_round_trip(space: NeuralColorSpace, XYZ: np.ndarray) -> dict:
    """Evaluate round-trip accuracy.

    Returns
    -------
    dict with max, mean, p99, p95 absolute error
    """
    errors = space.round_trip_error(XYZ)
    return {
        "max": float(np.max(errors)),
        "mean": float(np.mean(errors)),
        "p99": float(np.percentile(errors, 99)),
        "p95": float(np.percentile(errors, 95)),
    }


def evaluate_d4(space: NeuralColorSpace, XYZ: np.ndarray) -> dict | None:
    """Evaluate 4th dimension statistics for INN models.

    Returns None if the model doesn't have forward_full (e.g. MLP).
    """
    if not hasattr(space.model, "forward_full") or space.model.pad_dim == 0:
        return None

    XYZ = np.asarray(XYZ, dtype=np.float32).reshape(-1, 3)
    with torch.no_grad():
        t = torch.tensor(XYZ, dtype=torch.float32, device=space.device)
        z_full, _ = space.model.forward_full(t)
        if z_full.shape[1] <= 3:
            return None  # 3D INN, no d4
        d4 = z_full[:, 3].cpu().numpy()

    abs_d4 = np.abs(d4)
    return {
        "mean": float(d4.mean()),
        "std": float(d4.std()),
        "abs_mean": float(abs_d4.mean()),
        "abs_max": float(abs_d4.max()),
        "abs_p95": float(np.percentile(abs_d4, 95)),
        "abs_p99": float(np.percentile(abs_d4, 99)),
    }


def evaluate_smoothness(space: NeuralColorSpace, XYZ: np.ndarray, eps: float = 1e-4) -> dict:
    """Evaluate smoothness via numerical Jacobian condition number.

    Returns
    -------
    dict with mean, max, p99 condition numbers
    """
    XYZ = np.asarray(XYZ, dtype=np.float32)
    flat = XYZ.reshape(-1, 3)

    cond_numbers = []
    # Sample a subset for efficiency
    n = min(len(flat), 500)
    indices = np.random.choice(len(flat), n, replace=False)

    for idx in indices:
        x = flat[idx]
        J = np.zeros((3, 3), dtype=np.float64)
        for j in range(3):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += eps
            x_minus[j] -= eps
            f_plus = space.from_XYZ(x_plus.reshape(1, 3)).ravel()
            f_minus = space.from_XYZ(x_minus.reshape(1, 3)).ravel()
            J[:, j] = (f_plus - f_minus) / (2 * eps)

        cond = np.linalg.cond(J)
        cond_numbers.append(cond)

    cond_numbers = np.array(cond_numbers)
    return {
        "mean": float(np.mean(cond_numbers)),
        "max": float(np.max(cond_numbers)),
        "p99": float(np.percentile(cond_numbers, 99)),
    }
