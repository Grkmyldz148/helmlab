"""Test color pair generator for bidirectional optimization.

Generates critical-zone, uniform-random, and adaptive color pairs
for human perception surveys.
"""

import json

import numpy as np

from helmlab.utils.srgb_convert import (
    srgb_to_hex,
    hex_to_srgb,
    XYZ_to_sRGB,
    sRGB_to_XYZ,
    clamp_srgb,
)
from helmlab.utils.gamut import is_in_gamut


class ColorPairGenerator:
    """Generates test color pairs for human perception studies."""

    def __init__(self, helmlab):
        self._helmlab = helmlab
        self._space = helmlab._space

    def _random_in_gamut_lab(self, rng: np.random.Generator, gamut: str = "srgb") -> np.ndarray:
        """Generate a random Lab coordinate that is in-gamut."""
        for _ in range(200):
            # Random sRGB → Lab
            srgb = rng.uniform(0.0, 1.0, size=3)
            XYZ = sRGB_to_XYZ(srgb)
            lab = self._space.from_XYZ(XYZ)
            if is_in_gamut(lab, self._space, gamut):
                return lab
        # Fallback: mid-gray
        return self._space.from_XYZ(sRGB_to_XYZ(np.array([0.5, 0.5, 0.5])))

    def _lab_to_hex(self, lab: np.ndarray) -> str:
        """Lab → hex via gamut-mapped sRGB."""
        return self._helmlab.to_hex(lab)

    def _predicted_de(self, lab1: np.ndarray, lab2: np.ndarray) -> float:
        """Helmlab distance between two Lab values."""
        return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))

    def critical_zone_pairs(self, n_per_zone: int = 50, seed: int = 42) -> list[dict]:
        """Generate pairs from perceptually critical zones.

        Zones:
        - blue_region: hue 260-290 degrees (blue-violet, known difficulty)
        - hk_boundary: high-chroma colors near H-K effect peak
        - achromatic_chromatic: gray paired with chromatic at same L

        Returns list of dicts with hex1, hex2, predicted_de, zone, reason.
        """
        rng = np.random.default_rng(seed)
        pairs = []

        # Zone 1: Blue region (260-290 deg hue)
        # Mix of small and large perturbations for ΔE diversity
        for _ in range(n_per_zone):
            L = rng.uniform(0.2, 0.8)
            H_deg = rng.uniform(255.0, 295.0)
            H_rad = np.radians(H_deg)
            C = rng.uniform(0.03, 0.20)
            lab1 = np.array([L, C * np.cos(H_rad), C * np.sin(H_rad)])
            # Variable perturbation size
            scale = rng.choice([0.3, 1.0, 2.0])  # small/med/large
            dL = rng.uniform(-0.15, 0.15) * scale
            dC = rng.uniform(-0.06, 0.06) * scale
            dH = rng.uniform(-20.0, 20.0) * scale
            H2 = np.radians(H_deg + dH)
            C2 = max(0.01, C + dC)
            lab2 = np.array([np.clip(L + dL, 0.05, 1.2), C2 * np.cos(H2), C2 * np.sin(H2)])
            pairs.append({
                "hex1": self._lab_to_hex(lab1),
                "hex2": self._lab_to_hex(lab2),
                "predicted_de": self._predicted_de(lab1, lab2),
                "zone": "blue_region",
                "reason": f"Blue hue {H_deg:.0f}, L={L:.2f}",
            })

        # Zone 2: H-K boundary (high chroma vs low chroma, various hues)
        for _ in range(n_per_zone):
            L = rng.uniform(0.3, 0.8)
            H_deg = rng.uniform(0.0, 360.0)
            H_rad = np.radians(H_deg)
            C_high = rng.uniform(0.08, 0.30)
            C_low = rng.uniform(0.00, 0.05)
            # Optional L shift for more diversity
            dL = rng.uniform(-0.1, 0.1) * rng.choice([0.0, 1.0])
            lab1 = np.array([L, C_high * np.cos(H_rad), C_high * np.sin(H_rad)])
            lab2 = np.array([L + dL, C_low * np.cos(H_rad), C_low * np.sin(H_rad)])
            pairs.append({
                "hex1": self._lab_to_hex(lab1),
                "hex2": self._lab_to_hex(lab2),
                "predicted_de": self._predicted_de(lab1, lab2),
                "zone": "hk_boundary",
                "reason": f"C={C_high:.2f} vs C={C_low:.2f} at hue {H_deg:.0f}",
            })

        # Zone 3: Achromatic ↔ Chromatic
        for _ in range(n_per_zone):
            L = rng.uniform(0.2, 0.8)
            lab_gray = np.array([L, 0.0, 0.0])
            H_deg = rng.uniform(0.0, 360.0)
            H_rad = np.radians(H_deg)
            C = rng.uniform(0.03, 0.25)
            dL = rng.uniform(-0.1, 0.1) * rng.choice([0.0, 1.0])
            lab_chrom = np.array([L + dL, C * np.cos(H_rad), C * np.sin(H_rad)])
            pairs.append({
                "hex1": self._lab_to_hex(lab_gray),
                "hex2": self._lab_to_hex(lab_chrom),
                "predicted_de": self._predicted_de(lab_gray, lab_chrom),
                "zone": "achromatic_chromatic",
                "reason": f"Gray vs chroma={C:.3f} at L={L:.2f}",
            })

        return pairs

    def uniform_random_pairs(self, n: int = 200, seed: int = 42) -> list[dict]:
        """Generate uniformly distributed random in-gamut pairs.

        Controls deltaE distribution: pairs span a range of distances.
        """
        rng = np.random.default_rng(seed)
        pairs = []

        for _ in range(n):
            lab1 = self._random_in_gamut_lab(rng)
            lab2 = self._random_in_gamut_lab(rng)
            pairs.append({
                "hex1": self._lab_to_hex(lab1),
                "hex2": self._lab_to_hex(lab2),
                "predicted_de": self._predicted_de(lab1, lab2),
                "zone": "uniform",
                "reason": "Random sRGB gamut pair",
            })

        return pairs

    def adaptive_pairs(self, residuals: dict, n: int = 100, seed: int = 42) -> list[dict]:
        """Generate pairs focused on high-residual regions.

        Parameters
        ----------
        residuals : dict with keys "hex1", "hex2", "residual" (list of dicts)
            Output from model evaluation with per-pair residuals.
        n : number of pairs to generate
        seed : random seed

        Returns pairs sampled near the regions with highest prediction error.
        """
        rng = np.random.default_rng(seed)
        pairs_data = residuals if isinstance(residuals, list) else residuals.get("pairs", [])

        if not pairs_data:
            return self.uniform_random_pairs(n, seed)

        # Sort by absolute residual, take top quartile
        sorted_pairs = sorted(pairs_data, key=lambda x: abs(x.get("residual", 0)), reverse=True)
        top_k = sorted_pairs[:max(1, len(sorted_pairs) // 4)]

        pairs = []
        for _ in range(n):
            base = rng.choice(top_k)
            lab1 = self._helmlab.from_hex(base["hex1"])
            lab2 = self._helmlab.from_hex(base["hex2"])

            # Perturb near the high-error pair
            noise = rng.normal(0.0, 0.02, size=3)
            lab1_new = lab1 + noise
            noise2 = rng.normal(0.0, 0.02, size=3)
            lab2_new = lab2 + noise2

            pairs.append({
                "hex1": self._lab_to_hex(lab1_new),
                "hex2": self._lab_to_hex(lab2_new),
                "predicted_de": self._predicted_de(lab1_new, lab2_new),
                "zone": "adaptive",
                "reason": f"Near high-residual pair (|r|={abs(base.get('residual', 0)):.2f})",
            })

        return pairs

    def export_for_survey(self, pairs: list[dict], output_path: str) -> None:
        """Export pairs as JSON for survey UI consumption."""
        data = {
            "version": "1.0",
            "model": "helmlab-v14",
            "n_pairs": len(pairs),
            "pairs": pairs,
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
