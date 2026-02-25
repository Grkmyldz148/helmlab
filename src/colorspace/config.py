"""Project-wide configuration: constants, paths, hyperparameters."""

from pathlib import Path
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────
# __file__ = src/colorspace/config.py → parent.parent.parent = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# ── CIE Constants ───────────────────────────────────────────────────
# D65 standard illuminant (CIE 1931 2° observer)
D65_WHITE = np.array([0.95047, 1.0, 1.08883])

# CIE Lab reference white (same as D65 for our purposes)
LAB_EPSILON = 216.0 / 24389.0  # 0.008856
LAB_KAPPA = 24389.0 / 27.0     # 903.3

# ── Training Hyperparameters ─────────────────────────────────────────
class TrainConfig:
    # Model
    inn_coupling_blocks: int = 16
    inn_subnet_hidden: int = 128
    inn_pad_dim: int = 0          # 0 = 3D (perfect round-trip), 1 = 4D
    mlp_hidden: int = 256
    mlp_layers: int = 3

    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 500

    # Scheduler (CosineAnnealingWarmRestarts)
    T_0: int = 50
    T_mult: int = 2

    # Loss weights
    alpha_stress: float = 1.0     # STRESS loss weight
    beta_hue: float = 0.1         # Hue linearity loss weight
    gamma_roundtrip: float = 10.0 # Round-trip loss weight (MLP only)
    delta_d4: float = 100.0       # d4 regularization weight (INN only)
    epsilon_hk: float = 0.0       # H-K effect loss weight (0 = disabled)

    # Training
    grad_clip: float = 1.0
    early_stop_patience: int = 50
    val_split: float = 0.2

    # Device
    seed: int = 42

    def to_dict(self) -> dict:
        """Serialize all config values (class + instance level)."""
        d = {}
        for k in dir(self):
            if k.startswith("_") or callable(getattr(self, k)):
                continue
            d[k] = getattr(self, k)
        return d
