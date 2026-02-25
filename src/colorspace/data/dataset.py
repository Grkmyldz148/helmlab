"""PyTorch Dataset and DataLoader construction."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from colorspace.config import TrainConfig
from colorspace.data.preprocessing import build_pair_data, build_hue_data


class ColorPairDataset(Dataset):
    """Dataset of color pairs with visual differences."""

    def __init__(self, XYZ_1: np.ndarray, XYZ_2: np.ndarray, DV: np.ndarray):
        self.XYZ_1 = torch.tensor(XYZ_1, dtype=torch.float32)
        self.XYZ_2 = torch.tensor(XYZ_2, dtype=torch.float32)
        self.DV = torch.tensor(DV, dtype=torch.float32)

    def __len__(self):
        return len(self.DV)

    def __getitem__(self, idx):
        return self.XYZ_1[idx], self.XYZ_2[idx], self.DV[idx]


class HueDataset(Dataset):
    """Dataset for hue linearity: points grouped by constant hue."""

    def __init__(self, XYZ: np.ndarray, hue_idx: np.ndarray):
        self.XYZ = torch.tensor(XYZ, dtype=torch.float32)
        self.hue_idx = torch.tensor(hue_idx, dtype=torch.long)

    def __len__(self):
        return len(self.hue_idx)

    def __getitem__(self, idx):
        return self.XYZ[idx], self.hue_idx[idx]


def build_dataloaders(
    cfg: TrainConfig | None = None,
) -> dict:
    """Build train/val DataLoaders for pair data and hue data.

    Returns
    -------
    dict with keys:
        - train_pairs: DataLoader
        - val_pairs: DataLoader
        - hue: DataLoader (full, no split — auxiliary data)
        - n_train: int
        - n_val: int
        - combvd_max: float — for denormalization
    """
    if cfg is None:
        cfg = TrainConfig()

    # Build data
    pair_data = build_pair_data()
    hue_data = build_hue_data()

    # Create datasets
    pair_ds = ColorPairDataset(
        pair_data["XYZ_1"], pair_data["XYZ_2"], pair_data["DV"]
    )
    hue_ds = HueDataset(hue_data["XYZ"], hue_data["hue_idx"])

    # Train/val split
    n_total = len(pair_ds)
    n_val = int(n_total * cfg.val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(pair_ds, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    hue_loader = DataLoader(
        hue_ds, batch_size=len(hue_ds), shuffle=False,
    )

    return {
        "train_pairs": train_loader,
        "val_pairs": val_loader,
        "hue": hue_loader,
        "n_train": n_train,
        "n_val": n_val,
        "combvd_max": pair_data["combvd_max"],
    }
