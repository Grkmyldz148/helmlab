"""Human feedback dataset collector for bidirectional optimization.

Stores perceived visual differences (DV) from human observers,
outputs COMBVD-compatible format for training.
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np

from colorspace.utils.srgb_convert import hex_to_srgb, sRGB_to_XYZ


class FeedbackDataset:
    """Collect and manage human perceptual judgements."""

    def __init__(self, path: str = "data/human_feedback.json"):
        self._path = Path(path)
        self._data = {"judgements": [], "observers": set()}
        if self._path.exists():
            self._load()

    def _load(self):
        with open(self._path) as f:
            raw = json.load(f)
        self._data["judgements"] = raw.get("judgements", [])
        self._data["observers"] = set(raw.get("observers", []))

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            "judgements": self._data["judgements"],
            "observers": list(self._data["observers"]),
        }
        with open(self._path, "w") as f:
            json.dump(raw, f, indent=2)

    def add_judgement(
        self,
        hex1: str,
        hex2: str,
        perceived_dv: float,
        observer_id: str,
        metadata: dict | None = None,
    ):
        """Add a single human judgement.

        Parameters
        ----------
        hex1, hex2 : hex color strings
        perceived_dv : perceived visual difference (0-100 scale)
        observer_id : unique observer identifier
        metadata : optional extra info (viewing conditions, etc.)
        """
        entry = {
            "hex1": hex1,
            "hex2": hex2,
            "perceived_dv": perceived_dv,
            "observer_id": observer_id,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            entry["metadata"] = metadata
        self._data["judgements"].append(entry)
        self._data["observers"].add(observer_id)
        self._save()

    def load(self) -> dict:
        """Return dataset in COMBVD-compatible format.

        Returns
        -------
        dict with keys:
            "XYZ_1": ndarray (N, 3)
            "XYZ_2": ndarray (N, 3)
            "DV": ndarray (N,)
        """
        if not self._data["judgements"]:
            return {"XYZ_1": np.zeros((0, 3)), "XYZ_2": np.zeros((0, 3)), "DV": np.zeros(0)}

        # Average perceived DV per unique pair
        pair_dvs: dict[tuple, list] = {}
        for j in self._data["judgements"]:
            key = (j["hex1"], j["hex2"])
            pair_dvs.setdefault(key, []).append(j["perceived_dv"])

        n = len(pair_dvs)
        XYZ_1 = np.zeros((n, 3))
        XYZ_2 = np.zeros((n, 3))
        DV = np.zeros(n)

        for i, ((h1, h2), dvs) in enumerate(pair_dvs.items()):
            XYZ_1[i] = sRGB_to_XYZ(hex_to_srgb(h1))
            XYZ_2[i] = sRGB_to_XYZ(hex_to_srgb(h2))
            DV[i] = np.mean(dvs)

        return {"XYZ_1": XYZ_1, "XYZ_2": XYZ_2, "DV": DV}

    def merge_with_combvd(self, combvd: dict) -> dict:
        """Merge feedback data with existing COMBVD dataset.

        Parameters
        ----------
        combvd : dict with XYZ_1, XYZ_2, DV arrays

        Returns
        -------
        Merged dict in same format. Feedback pairs appended (no dedup needed
        since hex → XYZ conversion makes exact matches unlikely).
        """
        fb = self.load()
        if fb["DV"].size == 0:
            return combvd

        return {
            "XYZ_1": np.vstack([combvd["XYZ_1"], fb["XYZ_1"]]),
            "XYZ_2": np.vstack([combvd["XYZ_2"], fb["XYZ_2"]]),
            "DV": np.concatenate([combvd["DV"], fb["DV"]]),
        }

    def stats(self) -> dict:
        """Return dataset statistics."""
        # Unique pairs
        pairs = set()
        for j in self._data["judgements"]:
            pairs.add((j["hex1"], j["hex2"]))

        return {
            "n_judgements": len(self._data["judgements"]),
            "n_pairs": len(pairs),
            "n_observers": len(self._data["observers"]),
            "observers": list(self._data["observers"]),
        }
