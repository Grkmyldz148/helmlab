"""MLP baseline model for color space transformation."""

import torch
import torch.nn as nn

from helmlab.config import TrainConfig


class ColorMLP(nn.Module):
    """MLP: XYZ (3D) → Perceptual (3D), with separate inverse MLP."""

    def __init__(self, cfg: TrainConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = TrainConfig()

        h = cfg.mlp_hidden

        # Forward: XYZ → perceptual
        layers_fwd = [nn.Linear(3, h), nn.GELU()]
        for _ in range(cfg.mlp_layers - 1):
            layers_fwd.extend([nn.Linear(h, h), nn.GELU()])
        layers_fwd.append(nn.Linear(h, 3))
        self.encoder = nn.Sequential(*layers_fwd)

        # Inverse: perceptual → XYZ
        layers_inv = [nn.Linear(3, h), nn.GELU()]
        for _ in range(cfg.mlp_layers - 1):
            layers_inv.extend([nn.Linear(h, h), nn.GELU()])
        layers_inv.append(nn.Linear(h, 3))
        self.decoder = nn.Sequential(*layers_inv)

    def forward(self, XYZ: torch.Tensor) -> torch.Tensor:
        """Forward: XYZ → perceptual."""
        return self.encoder(XYZ)

    def inverse(self, perceptual: torch.Tensor) -> torch.Tensor:
        """Inverse: perceptual → XYZ."""
        return self.decoder(perceptual)
