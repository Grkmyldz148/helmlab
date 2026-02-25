"""Invertible Neural Network for color space transformation (FrEIA)."""

import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm

from colorspace.config import TrainConfig


def _make_subnet(dim_in: int, dim_out: int, hidden: int = 128) -> nn.Module:
    """Create subnet for coupling blocks."""
    return nn.Sequential(
        nn.Linear(dim_in, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, dim_out),
    )


class ColorINN(nn.Module):
    """Invertible Neural Network: XYZ (3D) → Perceptual Space (3D).

    Supports two modes via inn_pad_dim config:
    - pad_dim=1 (4D): Pad to 4D for balanced 2/2 coupling splits.
      The 4th dim is a latent nuisance variable.
    - pad_dim=0 (3D): Direct 3D→3D, no padding. Coupling does 1/2 split.
      Perfect round-trip by construction.
    """

    def __init__(self, cfg: TrainConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = TrainConfig()

        self.pad_dim = cfg.inn_pad_dim
        self.total_dim = 3 + self.pad_dim
        hidden = cfg.inn_subnet_hidden

        self.inn = Ff.SequenceINN(self.total_dim)
        for _ in range(cfg.inn_coupling_blocks):
            self.inn.append(
                Fm.AllInOneBlock,
                subnet_constructor=lambda din, dout: _make_subnet(din, dout, hidden),
                permute_soft=True,
            )

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad 3D input to total_dim with zeros (no-op if pad_dim=0)."""
        if self.pad_dim == 0:
            return x
        B = x.shape[0]
        padded = torch.zeros(B, self.total_dim, device=x.device, dtype=x.dtype)
        padded[:, :3] = x
        return padded

    def forward(self, XYZ: torch.Tensor) -> torch.Tensor:
        """Forward: XYZ (B, 3) → perceptual (B, 3)."""
        padded = self._pad(XYZ)
        z, _ = self.inn(padded)
        return z[:, :3]

    def inverse(self, perceptual: torch.Tensor) -> torch.Tensor:
        """Inverse: perceptual (B, 3) → XYZ (B, 3).

        For 3D mode: exact inverse.
        For 4D mode: sets d4=0 (approximate).
        """
        padded = self._pad(perceptual)
        x, _ = self.inn(padded, rev=True)
        return x[:, :3]

    def forward_full(self, XYZ: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward returning full output and log Jacobian determinant."""
        padded = self._pad(XYZ)
        z, log_jac = self.inn(padded)
        return z, log_jac

    def inverse_full(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse from full representation (exact for both 3D and 4D)."""
        x, log_jac = self.inn(z, rev=True)
        return x, log_jac
